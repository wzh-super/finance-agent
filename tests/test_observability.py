"""控制台与轨迹视图验收：仅合成文件和替身，绝不调用 API 或执行研究。"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_agent.models import (
    BacktestResult, CodingAttempt, ExecutionResult, Experiment, FactorArtifact,
    FactorSpec, Feedback, Hypothesis, RunState,
)
from factor_agent.storage import write_json


def snapshot(root):
    """Compare filenames and bytes; ordinary reads may update filesystem atime."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def markdown_code_blocks(text):
    """Read CommonMark backtick/tilde fences, including longer nested literal fences."""
    fence = None
    body = []
    blocks = []
    for line in text.splitlines():
        match = re.match(r"^([`~]{3,})(.*)$", line)
        if fence is None:
            if match and len(set(match.group(1))) == 1:
                fence = match.group(1)
                body = []
        elif match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence) and not match.group(2).strip():
            blocks.append("\n".join(body))
            fence = None
        else:
            body.append(line)
    return blocks


class ProgressTests(unittest.TestCase):
    def test_progress_records_timestamp_context_and_redacts_nested_details(self):
        from factor_agent.progress import ProgressLogger, safe_value

        secret = "fake-observability-api-key-never-record"
        details = {"round": 2, "factor": "momentum", "attempt": 3,
                   "nested": [{"message": "credential " + secret, "count": 7}], "ok": False}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
            root = Path(temporary)
            output = io.StringIO()
            with redirect_stdout(output):
                ProgressLogger(root).event("coding", "Repair code containing " + secret, **details)
            human = (root / "progress.log").read_text()
            record = json.loads((root / "events.jsonl").read_text().strip())
            datetime.fromisoformat(record["time"].replace("Z", "+00:00"))
            self.assertEqual(record["stage"], "coding")
            for text in (output.getvalue(), human, json.dumps(record)):
                self.assertNotIn(secret, text)
                self.assertIn("momentum", text)
            cleaned = safe_value(details)
            self.assertEqual(cleaned["nested"][0]["count"], 7)
            self.assertFalse(cleaned["ok"])
            self.assertNotIn(secret, json.dumps(cleaned))

    def test_failed_reading_view_keeps_authoritative_checkpoint_and_emits_warning(self):
        from factor_agent.storage import RunStore

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = RunStore(root)
            state = RunState("fixture-source", current=Experiment(0, stage="propose"))
            with patch.object(store, "write_trajectory", side_effect=ValueError("view renderer failed")), redirect_stdout(io.StringIO()):
                store.save(state)
            self.assertEqual(store.load("fixture-source"), state)
            self.assertTrue((root / "rounds/000/experiment.json").is_file())
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            warnings = [event for event in events if event["stage"] == "warning"]
            self.assertEqual(len(warnings), 1)
            self.assertIn("view renderer failed", json.dumps(warnings[0]))

    def test_source_snapshot_interruption_never_publishes_partial_tree(self):
        from factor_agent.trajectory import snapshot_source
        import shutil

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_copy = shutil.copyfile
            count = 0

            def interrupted_copy(source, target):
                nonlocal count
                count += 1
                if count == 2:
                    raise KeyboardInterrupt()
                return original_copy(source, target)

            with patch("factor_agent.trajectory.shutil.copyfile", side_effect=interrupted_copy):
                with self.assertRaises(KeyboardInterrupt):
                    snapshot_source(root)
            self.assertFalse((root / "source_snapshot").exists())
            self.assertEqual(list(root.glob(".source_snapshot-*")), [])
            snapshot_source(root)
            manifest = json.loads((root / "source_snapshot/manifest.json").read_text())
            self.assertIn("factor_agent/pipeline.py", manifest["files"])
            for relative, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((root / "source_snapshot" / relative).read_bytes()).hexdigest(), digest)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.secret = "fake-trajectory-secret-never-export"
        environment = patch.dict(os.environ, {"OPENAI_API_KEY": self.secret})
        environment.start()
        self.addCleanup(environment.stop)
        self.spec = FactorSpec("liquidity_signal", "Liquidity hypothesis factor", "$volume / Mean($volume, 5)",
                               {"$volume": "daily trading volume"})
        self.hypothesis = Hypothesis("Liquidity hypothesis", "Economic rationale retained in the trajectory")

    def partial_run(self):
        experiment = Experiment(0, stage="code", hypothesis=self.hypothesis, specs=[self.spec])
        write_json(self.root / "state.json", RunState("old-source-fingerprint", current=experiment))
        write_json(self.root / "run_mode.json", {"mode": "live_api"})
        factor_dir = self.root / "rounds/000/coding" / self.spec.identity
        write_json(factor_dir / "spec.json", self.spec)
        first = factor_dir / "attempt_001"
        first.mkdir(parents=True)
        code = "# FULL_CODE_BEGIN\n" + "\n".join(f"# preserved source line {n:04d}" for n in range(400))
        code += "\ntext = '''\n```\n````\n'''\n# " + self.secret + "\n# FULL_CODE_END\n"
        (first / "factor.py").write_text(code)
        execution = ExecutionResult(False, stdout="Traceback evidence\nValueError: original alignment failure",
                                    error="original alignment failure", returncode=1)
        write_json(first / "coding_execution.json", execution)
        write_json(first / "attempt.json", CodingAttempt(1, str(first / "factor.py"), execution,
                   code_feedback="The first attempt mixed instruments", final_feedback="Implementation rejected",
                   accepted=False))
        second = factor_dir / "attempt_002"
        second.mkdir()
        (second / "factor.py").write_text("# PARTIAL_REPAIR_CODE\n# final reviewer has not replied yet\n")
        write_json(second / "coding_execution.json", ExecutionResult(True, stdout="second execution completed"))
        write_json(second / "code_review.json", "Partial-attempt critic is already available")
        # Neither attempt.json nor factor.json exists for this interrupted repair.
        return first, second

    def test_partial_trajectory_keeps_full_code_errors_and_unfinished_attempt_artifacts(self):
        from factor_agent.trajectory import render_trajectory

        first, second = self.partial_run()
        before = snapshot(self.root)
        rendered = render_trajectory(self.root)
        self.assertEqual(snapshot(self.root), before)
        for retained in ("live_api", self.hypothesis.hypothesis, self.hypothesis.reason,
                         self.spec.formulation, "original alignment failure", "The first attempt mixed instruments",
                         "PARTIAL_REPAIR_CODE", "Partial-attempt critic is already available"):
            self.assertIn(retained, rendered)
        self.assertIn(str(first.relative_to(self.root) / "factor.py"), rendered)
        self.assertIn(str(second.relative_to(self.root) / "factor.py"), rendered)
        blocks = markdown_code_blocks(rendered)
        full = [block for block in blocks if "FULL_CODE_BEGIN" in block]
        self.assertEqual(len(full), 1)
        self.assertIn("FULL_CODE_END", full[0])
        self.assertIn("preserved source line 0399", full[0])
        self.assertIn("\n```\n````\n", full[0])
        self.assertNotIn(self.secret, rendered)

    def test_explicit_export_is_redacted_and_does_not_rewrite_original_evidence(self):
        from factor_agent.trajectory import render_trajectory, write_trajectory

        self.partial_run()
        before = snapshot(self.root)
        output = self.root / "exports/readable.md"
        expected = render_trajectory(self.root, relative_to=output.parent)
        self.assertEqual(write_trajectory(self.root, output), output)
        self.assertEqual(output.read_text(), expected)
        self.assertNotIn(self.secret, output.read_text())
        after = snapshot(self.root)
        self.assertEqual({name: after[name] for name in before}, before)

    def test_batch_timeline_distinguishes_committed_and_partial_evolution_rounds(self):
        from factor_agent.trajectory import render_trajectory

        self.partial_run()
        coding = self.root / "rounds/000/coding"
        write_json(coding / "schedule.json", {"identities": [self.spec.identity], "max_evolution_rounds": 10})
        for number in (1, 2):
            write_json(coding / f"evolution_{number:03d}/plan.json",
                       {"number": number, "scheduled": [self.spec.identity], "blocked": [], "reused": {}})
        write_json(coding / "evolution_001/completed.json", {"number": 1})
        before = snapshot(self.root)
        rendered = render_trajectory(self.root)
        self.assertIn("### Coding Evolution Rounds", rendered)
        self.assertIn("Maximum evolution rounds: 10", rendered)
        self.assertIn("| 1 | liquidity_signal | true |", rendered)
        self.assertIn("| 2 | liquidity_signal | false |", rendered)
        self.assertEqual(snapshot(self.root), before)

    def test_implementation_success_and_investment_rejection_have_distinct_labels(self):
        from factor_agent.trajectory import render_trajectory

        metrics = {"IC": 0.01, "1day.excess_return_with_cost.annualized_return": 0.02,
                   "1day.excess_return_with_cost.max_drawdown": -0.1}
        artifact = FactorArtifact(self.spec, implemented=True)
        feedback = Feedback("Underperformed baseline", "Hypothesis is unconfirmed", "Try another window",
                            "Reject the investment result", False)
        experiment = Experiment(0, stage="complete", hypothesis=self.hypothesis, specs=[self.spec],
                                factors=[artifact], result=BacktestResult(metrics, "fixture"), feedback=feedback)
        write_json(self.root / "state.json", RunState("old-source-fingerprint", history=[experiment]))
        write_json(self.root / "run_mode.json", {"mode": "offline_replay"})
        rendered = render_trajectory(self.root)
        self.assertIn("offline_replay", rendered)
        self.assertIn("implemented: true", rendered)
        self.assertIn("research decision: false", rendered)
        self.assertIn("Underperformed baseline", rendered)
        self.assertIn("Accepted factor library: empty", rendered)

    def test_export_refuses_to_overwrite_checkpoints_code_or_other_reports(self):
        from factor_agent.trajectory import write_trajectory

        first, _ = self.partial_run()
        report = self.root / "report.md"
        report.write_text("# Original investment report\nKeep this evidence.\n")
        targets = [self.root / "state.json", first / "factor.py", report]
        before = snapshot(self.root)
        for target in targets:
            with self.subTest(target=target.relative_to(self.root)):
                with self.assertRaises(ValueError):
                    write_trajectory(self.root, target)
        self.assertEqual(snapshot(self.root), before)

    def test_show_reads_old_run_without_env_config_runtime_or_agent_construction(self):
        from factor_agent.cli import main

        self.partial_run()
        before = snapshot(self.root)
        output = io.StringIO()
        with ExitStack() as stack:
            for target in ("factor_agent.cli.load_dotenv", "factor_agent.config.AgentConfig.from_file",
                           "factor_agent.storage.RunStore.__init__", "factor_agent.llm.LLMClient.__init__",
                           "factor_agent.execution.FactorExecutor.__init__", "factor_agent.backtest.QlibBacktester.__init__"):
                stack.enter_context(patch(target, side_effect=AssertionError("read-only show crossed execution boundary")))
            stack.enter_context(redirect_stdout(output))
            self.assertEqual(main(["show", "--run-dir", str(self.root)]), 0)
        self.assertIn("PARTIAL_REPAIR_CODE", output.getvalue())
        self.assertNotIn(self.secret, output.getvalue())
        self.assertEqual(snapshot(self.root), before)

    def test_show_missing_run_does_not_create_directory(self):
        from factor_agent.cli import main

        missing = self.root / "never-created"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertNotEqual(main(["show", "--run-dir", str(missing)]), 0)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
