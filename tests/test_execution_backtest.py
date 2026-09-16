"""Real factor subprocess tests plus Qlib boundary tests; no LLM or training calls."""

import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from factor_agent.backtest import QlibBacktester, combine_factors, deduplicate_new_factors
from factor_agent.config import AgentConfig
from factor_agent.errors import ExperimentFailed
from factor_agent.execution import FactorExecutor, child_environment, file_hash, run_process, validate_factor
from factor_agent.models import ExecutionResult, FactorArtifact, FactorSpec
from factor_agent.qlib_worker import REQUIRED_METRICS
from factor_agent.storage import write_json


def sample_frame(name="momentum"):
    index = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=3), ["A", "B", "C"]],
                                      names=["datetime", "instrument"])
    return pd.DataFrame({name: [1., 2., 4.] * 3}, index=index)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        sample_frame("$close").to_hdf(self.data / "daily_pv.h5", key="data")
        self.config = AgentConfig(data_dir=self.data, debug_data_dir=self.data, python_executable=sys.executable,
                                  qlib_python=sys.executable, factor_timeout=60)
        self.spec = FactorSpec("momentum", "fixture", "close", {})
        self.code = self.root / "factor_source.py"
        self.code.write_text('''import pandas as pd
from pathlib import Path
data = pd.read_hdf("daily_pv.h5", key="data")
data.columns = ["momentum"]
data.to_hdf("result.h5", key="data")
counter = Path("counter")
counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else "1")
''')
        self.runtime = patch("factor_agent.execution.runtime_fingerprint", return_value={"python": sys.version})
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def test_factor_execution_cache_depends_on_input_and_rejects_stale_output(self):
        executor = FactorExecutor(self.config)
        directory = self.root / "attempt"
        first = executor.run(self.code, self.spec, directory)
        self.assertTrue(first.success, first.error)
        self.assertEqual(first.stats["rows"], 9)
        self.assertTrue(executor.run(self.code, self.spec, directory).success)
        self.assertEqual((directory / "counter").read_text(), "1")
        frame = sample_frame("$close") * 2
        frame.to_hdf(self.data / "daily_pv.h5", key="data", mode="w")
        self.assertTrue(executor.run(self.code, self.spec, directory).success)
        self.assertEqual((directory / "counter").read_text(), "2")
        self.code.write_text("raise RuntimeError('intentional failure')\n")
        failure = executor.run(self.code, self.spec, directory)
        self.assertFalse(failure.success)
        self.assertFalse((directory / "result.h5").exists())
        self.assertIn("intentional failure", failure.stdout)

    def test_contract_allows_warmup_nan_but_rejects_invalid_outputs(self):
        frame = sample_frame()
        frame.iloc[0] = np.nan
        self.assertEqual(validate_factor(frame, self.spec)["nan_count"], 1)
        broken = [frame.rename(columns={"momentum": "wrong"}), pd.concat([frame, frame]),
                  frame.assign(momentum=np.inf), frame.assign(momentum=np.nan), frame.reset_index()]
        for value in broken:
            with self.subTest(value=str(value.head())):
                with self.assertRaises(ValueError):
                    validate_factor(value, self.spec)

    def test_child_environment_removes_credentials_and_proxies(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture", "OTHER_TOKEN": "fixture",
                                     "HTTPS_PROXY": "fixture", "PATH": os.environ["PATH"]}):
            env = child_environment()
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("OTHER_TOKEN", env)
        self.assertNotIn("HTTPS_PROXY", env)
        self.assertIn("PATH", env)

    def test_interruption_kills_child_tree_before_propagating(self):
        process = Mock(pid=2345)
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with patch("factor_agent.execution.subprocess.Popen", return_value=process), \
                patch("factor_agent.execution.os.killpg") as kill:
            with self.assertRaises(KeyboardInterrupt):
                run_process(["fixture"], self.root, self.root / "output.log", 10)
            kill.assert_called_once_with(2345, signal.SIGKILL)
            self.assertEqual(process.wait.call_count, 2)

    def test_accepted_factor_mutation_is_not_silently_ignored(self):
        executor = FactorExecutor(self.config)
        result = executor.run(self.code, self.spec, self.root / "full", full=True)
        self.assertTrue(result.success, result.error)
        artifact = FactorArtifact(self.spec, str(self.code), result.data_path, implemented=True)
        backtester = QlibBacktester(self.config, executor)
        self.assertEqual(len(backtester._accepted_frame(artifact, result.stats["input_fingerprint"])), 9)
        self.code.write_text(self.code.read_text() + "\n# changed after acceptance\n")
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            backtester._accepted_frame(artifact, result.stats["input_fingerprint"])


class FactorCombinationTests(unittest.TestCase):
    def test_correlation_rule_is_signed_and_compares_only_old_to_new(self):
        old = sample_frame("old")
        new = pd.DataFrame({"same": old.old * 2, "negative": -old.old, "constant": 1.}, index=old.index)
        kept, dropped = deduplicate_new_factors(old, new)
        self.assertEqual(list(kept.columns), ["negative"])
        self.assertEqual(set(dropped), {"same", "constant"})

    def test_first_round_nan_and_later_complete_case_behavior_match_original(self):
        new = sample_frame("new")
        new.iloc[0] = np.nan
        self.assertEqual(len(combine_factors(None, new)), 9)
        joined = combine_factors(sample_frame("old"), new)
        self.assertEqual(len(joined), 8)
        self.assertEqual(list(joined.columns), [("feature", "old"), ("feature", "new")])


class BacktestBoundaryTests(unittest.TestCase):
    def test_only_full_data_survivors_enter_added_factor_library(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "values.h5"
            sample_frame("good").to_hdf(output, key="data")
            factors = []
            for name in ("good", "bad"):
                folder = root / name
                folder.mkdir()
                code = folder / "reused_factor.py"
                code.write_text("# reviewed fixture")
                factor = FactorArtifact(FactorSpec(name, "", ""), str(code), implemented=True)
                write_json(folder / "reuse.json", {"spec_identity": factor.spec.identity, "code_sha256": file_hash(code)})
                factors.append(factor)
            good, bad = factors
            executor = Mock()
            executor.run.side_effect = [ExecutionResult(True, data_path=str(output)),
                                        ExecutionResult(False, error="full-data error")]
            runner = QlibBacktester(AgentConfig(data_dir=root), executor)
            combined, added, dropped = runner._prepare_factors([], [good, bad], root)
            self.assertEqual([factor.spec.name for factor in added], ["good"])
            self.assertEqual(added[0].values_path, str(output))
            self.assertEqual(list(combined.columns), [("feature", "good")])
            self.assertEqual(dropped, {"bad": "full-data error"})

    def test_resume_backtest_rejects_code_changed_after_coding_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "reused_factor.py"
            code.write_text("# originally reviewed code")
            factor = FactorArtifact(FactorSpec("factor", "", ""), str(code), implemented=True)
            write_json(root / "reuse.json", {"spec_identity": factor.spec.identity, "code_sha256": file_hash(code)})
            code.write_text("# modified after coding checkpoint")
            executor = Mock()
            runner = QlibBacktester(AgentConfig(data_dir=root), executor)
            with self.assertRaisesRegex(ValueError, "provenance is invalid"):
                runner._prepare_factors([], [factor], root)
            executor.run.assert_not_called()

    def test_backtest_cache_tracks_provider_and_output_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = root / "provider"
            (provider / "calendars").mkdir(parents=True)
            calendar = provider / "calendars/day.txt"
            calendar.write_text("2020-01-01\n")
            config = AgentConfig(qlib_provider=provider, qlib_python=sys.executable)
            runner = QlibBacktester(config, Mock())
            directory = root / "backtest"

            def fake_qlib(argv, working, log_path, timeout):
                metrics = {name: 0.1 for name in REQUIRED_METRICS}
                for name in ("qlib_res.csv", "ret.pkl", "returns.csv", "equity_curve.svg"):
                    (working / name).write_text("test artifact")
                write_json(working / "worker_result.json", {"status": "FINISHED", "recorder_id": "test-run",
                           "metrics": metrics, "feature_count": 20, "report_rows": 1, "cache_key": argv[-1],
                           "outputs": {name: file_hash(working / name)
                                       for name in ("qlib_res.csv", "ret.pkl", "returns.csv", "equity_curve.svg")}})
                return 0, ""

            with patch("factor_agent.backtest.runtime_fingerprint", return_value={"version": "fixture"}), \
                    patch("factor_agent.backtest.run_process", side_effect=fake_qlib) as process:
                first = runner.run([], [], directory)
                self.assertEqual(first.metadata["total_feature_count"], 20)
                self.assertEqual(runner.run([], [], directory).metadata["recorder_id"], "test-run")
                self.assertEqual(process.call_count, 1)
                (directory / "backtest.json").unlink()
                runner.run([], [], directory)
                self.assertEqual(process.call_count, 1, "Recover completed child without retraining")
                (directory / "ret.pkl").write_text("corrupted")
                runner.run([], [], directory)
                self.assertEqual(process.call_count, 2)
                calendar.write_text("2020-01-01\n2020-01-02\n")
                runner.run([], [], directory)
                self.assertEqual(process.call_count, 3)

    def test_nonzero_worker_exit_cannot_reuse_previous_result_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "provider/calendars").mkdir(parents=True)
            (root / "provider/calendars/day.txt").write_text("2020-01-01\n")
            directory = root / "backtest"
            directory.mkdir()
            (directory / "worker_result.json").write_text('{"status":"FINISHED"}')
            runner = QlibBacktester(AgentConfig(qlib_provider=root / "provider"), Mock())
            with patch("factor_agent.backtest.runtime_fingerprint", return_value={}), \
                    patch("factor_agent.backtest.run_process", return_value=(1, "failed")):
                with self.assertRaisesRegex(ExperimentFailed, "status 1"):
                    runner.run([], [], directory)
            self.assertFalse((directory / "worker_result.json").exists())


if __name__ == "__main__":
    unittest.main()
