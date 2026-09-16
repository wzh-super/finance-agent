"""离线验收研究闭环：假组件只替代收费服务，使用真实主循环和 JSON 存储。"""

from dataclasses import dataclass
from contextlib import redirect_stdout
from pathlib import Path
import copy
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_agent.errors import ExperimentFailed, ServiceError
from factor_agent.models import BacktestResult, FactorArtifact, FactorSpec, Feedback, Hypothesis
from factor_agent.pipeline import Pipeline
from factor_agent.storage import RunStore


@dataclass
class TinyConfig:
    """编排层仅需配置指纹，不应依赖 Qlib 或 API 环境。"""

    value: str = "offline-test-config"

    def fingerprint(self):
        return self.value


class ScriptedResearcher:
    def __init__(self, decisions=(True, False, True)):
        self.decisions = decisions
        self.calls = {"propose": 0, "design": 0, "review": 0}
        self.proposal_histories = []
        self.design_libraries = []
        self.comparators = []
        self.reviewed_factors = []
        self.fail_once = None

    def _called(self, stage):
        self.calls[stage] += 1
        if self.fail_once == stage:
            self.fail_once = None
            raise ServiceError("temporary service failure")

    def propose(self, history):
        self._called("propose")
        self.proposal_histories.append(copy.deepcopy(history))
        return Hypothesis(f"hypothesis {len(history) + 1}", "a testable financial rationale")

    def design(self, hypothesis, history, accepted):
        self._called("design")
        self.design_libraries.append([a.spec.name for a in accepted])
        number = len(history) + 1
        return [FactorSpec(f"factor_{number}", hypothesis.hypothesis, "$close / lag($close, 5) - 1")]

    def review(self, hypothesis, factors, result, best_result, history):
        self._called("review")
        self.comparators.append(copy.deepcopy(best_result))
        self.reviewed_factors.append(copy.deepcopy(factors))
        return Feedback("observed", "evaluated", "next idea", "reason", self.decisions[len(history)])


class ScriptedCoder:
    def __init__(self):
        self.calls = 0
        self.fail_once = None
        self.failed_names = set()

    def develop(self, specs, directory):
        self.calls += 1
        if self.fail_once:
            error, self.fail_once = self.fail_once, None
            raise error
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        result = []
        for spec in specs:
            code = directory / f"{spec.name}.py"
            code.write_text("# retained implementation evidence\n", encoding="utf-8")
            result.append(FactorArtifact(spec, str(code), implemented=spec.name not in self.failed_names,
                                         error="implementation failed" if spec.name in self.failed_names else ""))
        return result


class ScriptedBacktester:
    def __init__(self):
        self.calls = []
        self.fail_next_candidate = None
        self.drop_names = set()

    def run(self, accepted, candidates, directory):
        self.calls.append(([a.spec.name for a in accepted], [a.spec.name for a in candidates]))
        if candidates and self.fail_next_candidate:
            error, self.fail_next_candidate = self.fail_next_candidate, None
            raise error
        added = [a for a in candidates if a.spec.name not in self.drop_names]
        value = 0.01 * len(self.calls)
        return BacktestResult(
            {"IC": value, "1day.excess_return_with_cost.annualized_return": value,
             "1day.excess_return_with_cost.max_drawdown": -0.1},
            str(directory),
            [a.spec.name for a in [*accepted, *added]],
            added_factors=added,
            dropped_factors={a.spec.name: "duplicate" for a in candidates if a.spec.name in self.drop_names},
        )


class PipelineAcceptanceTests(unittest.TestCase):
    def setUp(self):
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = TinyConfig()
        self.store = RunStore(self.root)
        self.researcher = ScriptedResearcher()
        self.coder = ScriptedCoder()
        self.backtester = ScriptedBacktester()

    def pipeline(self):
        return Pipeline(self.config, self.store, self.researcher, self.coder, self.backtester)

    def test_accept_reject_accept_accumulates_only_accepted_factors(self):
        state = self.pipeline().run(3)
        self.assertEqual([e.feedback.decision for e in state.history], [True, False, True])
        self.assertEqual([a.spec.name for a in state.accepted_factors], ["factor_1", "factor_3"])
        self.assertEqual(self.backtester.calls, [([], []), ([], ["factor_1"]),
                                                (["factor_1"], ["factor_2"]),
                                                (["factor_1"], ["factor_3"])])
        self.assertEqual(self.researcher.design_libraries, [[], ["factor_1"], ["factor_1"]])
        self.assertEqual([len(h) for h in self.researcher.proposal_histories], [0, 1, 2])
        self.assertFalse(self.researcher.proposal_histories[2][1].feedback.decision)
        self.assertEqual([r.factor_names for r in self.researcher.comparators],
                         [[], ["factor_1"], ["factor_1"]])
        self.assertEqual(state.best_result.factor_names, ["factor_1", "factor_3"])
        self.assertIsNone(state.current)
        loaded = self.store.load(self.config.fingerprint())
        self.assertEqual(loaded, state)

    def test_each_stage_resumes_without_repeating_completed_work(self):
        for stage in ("propose", "design", "code", "backtest", "review", "commit"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                store = RunStore(Path(root))
                researcher, coder, backtester = ScriptedResearcher(), ScriptedCoder(), ScriptedBacktester()
                Pipeline(self.config, store, researcher, coder, backtester).run(1, stop_after_stage=stage)
                state = Pipeline(self.config, store, researcher, coder, backtester).run(1)
                self.assertEqual(len(state.history), 1)
                self.assertEqual(researcher.calls, {"propose": 1, "design": 1, "review": 1})
                self.assertEqual(coder.calls, 1)
                self.assertEqual(len(backtester.calls), 2)

    def test_round_count_is_total_target_and_completed_run_does_nothing(self):
        self.pipeline().run(1)
        self.pipeline().run(3)
        before = (copy.deepcopy(self.researcher.calls), self.coder.calls, len(self.backtester.calls))
        state = self.pipeline().run(3)
        self.assertEqual(len(state.history), 3)
        self.assertEqual(before, (self.researcher.calls, self.coder.calls, len(self.backtester.calls)))

    def test_invalid_experiment_is_rejected_and_does_not_pollute_library(self):
        self.backtester.fail_next_candidate = ExperimentFailed("no valid full-sample output")
        state = self.pipeline().run(2)
        failed = state.history[0]
        self.assertFalse(failed.feedback.decision)
        self.assertIn("no valid full-sample output", failed.error + failed.feedback.observations)
        self.assertEqual(self.researcher.calls["review"], 1)
        self.assertEqual(self.backtester.calls[-1], ([], ["factor_2"]))
        self.assertEqual(state.accepted_factors, [])
        self.assertFalse(self.researcher.proposal_histories[1][0].feedback.decision)

    def test_backtest_only_admits_factors_reported_as_added(self):
        original_design = self.researcher.design

        def design_with_duplicate(hypothesis, history, accepted):
            return original_design(hypothesis, history, accepted) + [
                FactorSpec("discarded", "correlated candidate", "$close")]

        self.researcher.design = design_with_duplicate
        self.backtester.drop_names.add("discarded")
        state = self.pipeline().run(1)
        self.assertEqual([f.spec.name for f in state.accepted_factors], ["factor_1"])
        self.assertNotIn("discarded", state.best_result.factor_names)

    def test_service_failure_preserves_completed_stages_and_can_resume(self):
        self.researcher.fail_once = "review"
        with self.assertRaises(ServiceError):
            self.pipeline().run(1)
        paused = self.store.load(self.config.fingerprint())
        self.assertEqual(len(paused.history), 0)
        self.assertEqual(paused.current.stage, "review")
        self.assertIsNotNone(paused.current.result)
        state = self.pipeline().run(1)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(self.coder.calls, 1)
        self.assertEqual(len(self.backtester.calls), 2)
        self.assertEqual(self.researcher.calls, {"propose": 1, "design": 1, "review": 2})

    def test_partial_coding_failure_evaluates_survivors_and_retains_failure_evidence(self):
        original_design = self.researcher.design

        def design_with_failure(hypothesis, history, accepted):
            return original_design(hypothesis, history, accepted) + [
                FactorSpec("broken", "another candidate", "$volume")]

        self.researcher.design = design_with_failure
        self.coder.failed_names.add("broken")
        state = self.pipeline().run(1)
        self.assertEqual(self.backtester.calls[-1], ([], ["factor_1"]))
        reviewed = self.researcher.reviewed_factors[0]
        self.assertEqual([factor.spec.name for factor in reviewed], ["factor_1", "broken"])
        self.assertFalse(reviewed[1].implemented)
        self.assertEqual(state.history[0].factors[1].error, "implementation failed")
        self.assertEqual([factor.spec.name for factor in state.accepted_factors], ["factor_1"])

    def test_interruption_resumes_at_unfinished_stage(self):
        self.coder.fail_once = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.pipeline().run(1)
        paused = self.store.load(self.config.fingerprint())
        self.assertEqual(paused.current.stage, "code")
        self.assertEqual(paused.history, [])
        state = self.pipeline().run(1)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(self.researcher.calls["propose"], 1)
        self.assertEqual(self.researcher.calls["design"], 1)
        self.assertEqual(self.coder.calls, 2)

    def test_baseline_interruption_leaves_a_checkpoint_that_can_resume(self):
        with patch.object(self.backtester, "run", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.pipeline().run(1)
        self.assertTrue(self.store.state_path.is_file())
        paused = self.store.load(self.config.fingerprint())
        self.assertIsNone(paused.best_result)
        self.assertEqual(paused.history, [])
        self.assertEqual(self.researcher.calls["propose"], 0)
        state = self.pipeline().run(1)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(len(self.backtester.calls), 2)

    def test_config_change_cannot_silently_reuse_old_state(self):
        self.pipeline().run(1)
        with self.assertRaises((ValueError, RuntimeError)):
            self.store.load("different-data-or-model-settings")

    def _state_file(self):
        candidates = []
        for path in self.root.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "config_fingerprint" in value and "history" in value:
                candidates.append(path)
        self.assertEqual(len(candidates), 1, "one authoritative JSON checkpoint must exist")
        return candidates[0]

    def test_corrupt_state_and_unknown_schema_fail_instead_of_restarting(self):
        self.pipeline().run(1)
        state_file = self._state_file()
        saved = json.loads(state_file.read_text(encoding="utf-8"))
        state_file.write_text('{"history":', encoding="utf-8")
        with self.assertRaises((ValueError, RuntimeError)):
            self.store.load(self.config.fingerprint())
        saved["version"] = 999
        state_file.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaises((ValueError, RuntimeError)):
            self.store.load(self.config.fingerprint())


if __name__ == "__main__":
    unittest.main()
