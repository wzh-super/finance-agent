"""English framework output must not rewrite historical model or code evidence."""

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_agent.models import (
    BacktestResult, CodingAttempt, ExecutionResult, Experiment, FactorArtifact,
    FactorSpec, Feedback, Hypothesis, RunState,
)
from factor_agent.progress import ProgressLogger, metric_comparison
from factor_agent.storage import write_json
from factor_agent.trajectory import render_trajectory, write_trajectory


PACKAGE = Path(__file__).resolve().parents[1] / "factor_agent"
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U0003134f]")
LEGACY_TITLE = "# Agent 运行轨迹\n"


class FixedOutputLanguageTests(unittest.TestCase):
    def test_python_output_literals_are_english_with_one_narrow_legacy_reader_exception(self):
        violations = []
        legacy_count = 0
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                        docstrings.add(id(node.body[0].value))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or id(node) in docstrings:
                    continue
                if not CJK.search(node.value):
                    continue
                ancestors = []
                current = node
                while current in parents:
                    current = parents[current]
                    ancestors.append(current)
                legacy_reader = (
                    path.name == "trajectory.py" and node.value == LEGACY_TITLE
                    and any(isinstance(parent, ast.FunctionDef) and parent.name == "write_trajectory" for parent in ancestors)
                    and any(isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute)
                            and parent.func.attr == "startswith" for parent in ancestors)
                )
                if legacy_reader:
                    legacy_count += 1
                else:
                    violations.append(f"{path.name}:{node.lineno}")
        self.assertEqual(violations, [])
        self.assertEqual(legacy_count, 1, "Only the exact legacy trajectory-title reader may contain CJK")

    def test_prompt_assets_have_no_fixed_cjk_text(self):
        violations = [path.name for path in (PACKAGE / "prompts").glob("*.txt") if CJK.search(path.read_text())]
        self.assertEqual(violations, [])

    def test_metric_labels_and_progress_truncation_notice_are_english(self):
        current = {"IC": 0.03, "Rank IC": 0.04,
                   "1day.excess_return_with_cost.annualized_return": 0.06,
                   "1day.excess_return_with_cost.information_ratio": 0.9,
                   "1day.excess_return_with_cost.max_drawdown": -0.1}
        previous = {**current, "1day.excess_return_with_cost.annualized_return": 0.04}
        rendered = metric_comparison(current, previous)
        self.assertIn("Metric | Reference | Current | Change", rendered)
        self.assertIn("Annualized excess return after costs", rendered)
        self.assertIn("Excess-return maximum drawdown after costs", rendered)
        self.assertIn("+2.0000 pp", rendered)
        self.assertIsNone(CJK.search(rendered))
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()) as output:
            ProgressLogger(Path(temporary)).event("coding", "Generated factor code", code="x" * 6001)
            self.assertIn("Truncated", output.getvalue())
            self.assertIsNone(CJK.search(output.getvalue()))


class HistoricalEvidenceLanguageTests(unittest.TestCase):
    def save_fixture(self, root, *, historical=False):
        spec = FactorSpec("momentum", "A momentum factor", "$close / lag($close, 5) - 1")
        hypothesis = Hypothesis("历史因子假设" if historical else "Momentum hypothesis", "Economic rationale")
        folder = root / "rounds/000/coding" / spec.identity
        attempt_dir = folder / "attempt_001"
        attempt_dir.mkdir(parents=True)
        code = "# 历史代码必须保持原样\nvalue = 1\n" if historical else "# Original factor code\nvalue = 1\n"
        (attempt_dir / "factor.py").write_text(code)
        write_json(folder / "spec.json", spec)
        attempt = CodingAttempt(1, str(attempt_dir / "factor.py"), ExecutionResult(True),
                                code_feedback="Implementation matches the definition", final_feedback="Accepted", accepted=True)
        write_json(attempt_dir / "attempt.json", attempt)
        artifact = FactorArtifact(spec, str(attempt_dir / "factor.py"), attempts=[attempt], implemented=True)
        metrics = {"IC": 0.03, "1day.excess_return_with_cost.annualized_return": 0.06,
                   "1day.excess_return_with_cost.max_drawdown": -0.1}
        experiment = Experiment(0, stage="complete", hypothesis=hypothesis, specs=[spec], factors=[artifact],
                                result=BacktestResult(metrics, "fixture"),
                                feedback=Feedback("Observed", "Unconfirmed", "Try another window", "Reject", False))
        write_json(root / "state.json", RunState("old-source", history=[experiment]))
        write_json(root / "run_mode.json", {"mode": "live_api"})
        write_json(root / "baseline/worker_result.json", {"metrics": metrics})
        request_dir = root / "llm/requests/original-request"
        request = {"purpose": "code/round_000", "model": "saved-model", "messages": [
            {"role": "user", "content": "历史请求：请保留原始内容" if historical else "Original request"}]}
        response = "原始模型回答：不应翻译或改写。\n第二行证据。" if historical else "Original response"
        write_json(request_dir / "request.json", request)
        write_json(request_dir / "response_001.json", {"text": response, "usage": {"total_tokens": 12}})
        return code, response

    def test_trajectory_framework_headers_are_english(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.save_fixture(root)
            rendered = render_trajectory(root)
            self.assertTrue(rendered.startswith("# Agent Research Trajectory\n"))
            for heading in ("## Timeline and Source", "## Fixed-Model Baseline", "### Hypothesis and Rationale",
                            "### Factor 1: momentum", "### Full-Data Execution and Backtest",
                            "### Research Feedback and Acceptance Decision", "## Original Model Interactions"):
                self.assertIn(heading, rendered)
            self.assertIsNone(CJK.search(rendered))

    def test_reexporting_legacy_view_preserves_all_original_evidence_including_non_english_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, response = self.save_fixture(root, historical=True)
            output = root / "trajectory.md"
            output.write_text(LEGACY_TITLE + "Old generated view\n")
            originals = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in root.rglob("*") if path.is_file() and path != output}
            self.assertEqual(write_trajectory(root), output)
            rendered = output.read_text()
            self.assertTrue(rendered.startswith("# Agent Research Trajectory\n"))
            self.assertIn("历史因子假设", rendered)
            self.assertIn(code, rendered)
            self.assertIn(response, rendered)
            self.assertIn("llm/requests/original-request/request.json", rendered)
            self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}, originals)


if __name__ == "__main__":
    unittest.main()
