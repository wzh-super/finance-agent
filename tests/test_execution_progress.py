"""Observe actual child output before exit; no LLM requests or model training."""

from pathlib import Path
import os
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from factor_agent.execution import FactorExecutor, run_process
from factor_agent.models import FactorSpec


class ProcessProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.log = self.directory / "child.log"
        self.events = []

    def event(self, stage, message, **details):
        self.events.append((stage, message, details))

    def test_streaming_callback_unblocks_child_before_exit(self):
        script = """from pathlib import Path
import time
print('ready for observer', flush=True)
while not Path('observer_ack').exists():
    time.sleep(0.02)
Path('finished').write_text('yes')
print('child completed', flush=True)
"""

        def observer(stage, message, **details):
            self.event(stage, message, **details)
            if details.get("action") == "stdout" and "ready for observer" in message:
                self.assertFalse((self.directory / "finished").exists())
                (self.directory / "observer_ack").touch()

        code, output = run_process([sys.executable, "-c", script], self.directory, self.log, 10,
                                   event=observer, stage="test.child")
        self.assertEqual(code, 0)
        self.assertEqual(output, "ready for observer\nchild completed\n")
        self.assertTrue((self.directory / "finished").is_file())
        self.assertEqual(self.events[0][2]["action"], "start")
        self.assertEqual(self.events[-1][2]["action"], "complete")
        self.assertEqual(self.events[-1][2]["returncode"], 0)
        self.assertTrue(all(stage == "test.child" for stage, _, _ in self.events))

    def test_partial_utf8_carriage_returns_stderr_and_final_fragment(self):
        script = """import os,time
os.write(1, b'\\xe4')
time.sleep(0.7)
os.write(1, b'\\xb8\\xad\\rprogress 10%\\rprogress 100%\\n')
os.write(2, b'stderr line\\n')
os.write(1, b'final fragment')
"""
        code, output = run_process([sys.executable, "-c", script], self.directory, self.log, 10,
                                   event=self.event)
        messages = "\n".join(message for _, message, details in self.events if details["action"] == "stdout")
        self.assertEqual(code, 0)
        self.assertIn("中", messages)
        self.assertNotIn("\ufffd", messages)
        self.assertIn("progress 10%\nprogress 100%", messages)
        self.assertIn("stderr line", messages)
        self.assertTrue(messages.endswith("final fragment"))
        self.assertEqual(self.log.read_bytes(), "中\rprogress 10%\rprogress 100%\nstderr line\nfinal fragment".encode())
        self.assertTrue(output.endswith("final fragment"))

    def test_timeout_streams_remaining_output_and_kills_group(self):
        script = "import time; print('before timeout', flush=True); time.sleep(30)"
        with patch("factor_agent.execution.os.killpg", wraps=os.killpg) as kill:
            code, output = run_process([sys.executable, "-c", script], self.directory, self.log, 1,
                                       event=self.event)
        self.assertEqual(code, -signal.SIGKILL)
        self.assertIn("before timeout", output)
        self.assertIn("child process group terminated", output)
        self.assertEqual(kill.call_args.args[1], signal.SIGKILL)
        self.assertIn("timeout", [details["action"] for _, _, details in self.events])
        self.assertTrue(any("before timeout" in message for _, message, _ in self.events))

    def test_observer_exception_also_cleans_up_child(self):
        script = "import time; print('ready', flush=True); time.sleep(30)"

        def observer(stage, message, **details):
            if details["action"] == "stdout":
                raise RuntimeError("observer failed")

        with patch("factor_agent.execution.os.killpg", wraps=os.killpg) as kill:
            with self.assertRaisesRegex(RuntimeError, "observer failed"):
                run_process([sys.executable, "-c", script], self.directory, self.log, 10, event=observer)
        self.assertEqual(kill.call_count, 1)
        self.assertEqual(kill.call_args.args[1], signal.SIGKILL)

    def test_without_observer_retains_single_wait_contract(self):
        process = Mock(pid=2345)
        process.wait.return_value = 0
        with patch("factor_agent.execution.subprocess.Popen", return_value=process):
            self.assertEqual(run_process(["fixture"], self.directory, self.log, 17), (0, ""))
        process.wait.assert_called_once_with(timeout=17)

    def test_interruption_during_streaming_kills_group_and_propagates(self):
        process = Mock(pid=2345)
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with patch("factor_agent.execution.subprocess.Popen", return_value=process), \
                patch("factor_agent.execution.os.killpg") as kill:
            with self.assertRaises(KeyboardInterrupt):
                run_process(["fixture"], self.directory, self.log, 10, event=self.event)
        kill.assert_called_once_with(2345, signal.SIGKILL)
        self.assertEqual(process.wait.call_count, 2)


class FactorProgressTests(unittest.TestCase):
    def test_factor_events_include_data_summary_validation_and_cache_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "daily_pv.h5").write_bytes(b"fingerprinted fixture input")
            code = root / "source.py"
            code.write_text("# fixture implementation")
            config = SimpleNamespace(data_dir=data, debug_data_dir=data, python_executable=sys.executable,
                                     factor_timeout=10)
            spec = FactorSpec("example", "fixture", "fixture")
            records = []
            event = lambda stage, message, **details: records.append((stage, message, details))
            executor = FactorExecutor(config, event=event)
            invalid = False

            def process(argv, directory, log_path, timeout, **kwargs):
                index = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=2), ["A"]],
                                                  names=["datetime", "instrument"])
                column = "wrong" if invalid else "example"
                pd.DataFrame({column: [None, 1.]}, index=index).to_hdf(directory / "result.h5", key="data")
                log_path.write_text("fixture output\n")
                self.assertIs(kwargs["event"], event)
                return 0, "fixture output\n"

            with patch("factor_agent.execution.runtime_fingerprint", return_value={}), \
                    patch("factor_agent.execution.run_process", side_effect=process) as child:
                first = executor.run(code, spec, root / "execution", full=True)
                self.assertTrue(first.success)
                self.assertTrue(executor.run(code, spec, root / "execution", full=True).success)
                self.assertEqual(child.call_count, 1)
                summary = records[-1][2]
                self.assertEqual(summary["action"], "cache_hit")
                self.assertEqual(summary["rows"], 2)
                self.assertEqual(summary["nan_count"], 1)
                self.assertEqual(summary["dates"], ["2020-01-01 00:00:00", "2020-01-02 00:00:00"])
                self.assertTrue(summary["output_path"].endswith("result.h5"))
                invalid = True
                failure = executor.run(code, spec, root / "invalid", full=True)
                self.assertFalse(failure.success)
                self.assertIn("validation_failed", [entry[2]["action"] for entry in records])
                self.assertTrue(all(entry[0] == "factor.full" for entry in records))


if __name__ == "__main__":
    unittest.main()
