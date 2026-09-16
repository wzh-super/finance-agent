"""Behavioral checks for real coding/research classes; no API or training calls."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import tempfile
import unittest

from factor_agent.coding import FactorCoder
from factor_agent.errors import ExperimentFailed, ServiceError
from factor_agent.memory import CodingMemory
from factor_agent.models import CodingAttempt, ExecutionResult, FactorArtifact, FactorSpec, Hypothesis
from factor_agent.research import Researcher


def config(tmp_path, **kwargs):
    return SimpleNamespace(debug_data_dir=tmp_path, max_coding_attempts=3, max_task_failures=20,
                           memory_top_k=1, memory_recent_failures=3, embedding_model="test", **kwargs)


def spec(name="momentum"):
    return FactorSpec(name, "Momentum", "close / close.shift(10) - 1", {"close": "adjusted close"})


class ScriptedLLM:
    def __init__(self, **responses):
        self.responses = responses
        self.calls = Counter()

    def complete(self, system, user, *, purpose, response_json=True):
        stage = purpose.split("/")[0]
        self.calls[stage] += 1
        defaults = {
            "code": {"code": "print('factor')"},
            "format_review": {"output_format_decision": True, "output_format_feedback": "Correct format"},
            "code_review": "No critics found",
            "final_review": {"final_decision": True, "final_feedback": "Implementation is correct"},
        }
        value = self.responses[stage].pop(0) if self.responses.get(stage) else defaults[stage]
        if isinstance(value, Exception):
            raise value
        return value

    def embed(self, texts):
        self.calls["embed"] += 1
        return [[1.0, 0.5] for _ in texts]


class ScriptedExecutor:
    def __init__(self, successes=None):
        self.successes = list(successes or [True])
        self.calls = 0

    def run(self, code_path, factor_spec, directory, *, full=False):
        self.calls += 1
        success = self.successes.pop(0) if self.successes else True
        if not success:
            return ExecutionResult(False, stdout="ValueError: invalid shape", error="factor.py exited with status 1")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "result.h5"
        path.write_text("synthetic test result")
        return ExecutionResult(True, data_path=str(path), stats={"columns": [factor_spec.name]})


def coder(tmp_path, llm, executor, configuration=None):
    configuration = configuration or config(tmp_path)
    memory = CodingMemory(tmp_path / "memory.json", llm, configuration)
    return FactorCoder(llm, executor, memory, configuration), memory


class TestResearchCoding(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name)

    def test_research_deduplicates_accepted_names_and_rejects_malformed_specs(self):
        tmp_path = self.tmp_path
        old = spec("old")
        definition = {"description": "Momentum", "formulation": "close", "variables": {"close": "price"}}
        llm = ScriptedLLM(design=[{"old": definition, "new": definition}, {"bad": {"description": "bad"}}])
        researcher = Researcher(llm, config(tmp_path))
        assert [s.name for s in researcher.design(Hypothesis("h", "r"), [], [FactorArtifact(old)])] == ["new"]
        with self.assertRaisesRegex(ExperimentFailed, "variables"):
            researcher.design(Hypothesis("h", "r"), [], [])


    def test_failed_execution_cannot_be_accepted_by_llm_and_is_repaired(self):
        tmp_path = self.tmp_path
        llm = ScriptedLLM()
        executor = ScriptedExecutor([False, True])
        developer, memory = coder(tmp_path, llm, executor)
        result = developer.develop([spec()], tmp_path / "round")
        assert result[0].implemented
        assert [a.accepted for a in result[0].attempts] == [False, True]
        assert "deterministic" in result[0].attempts[0].final_feedback
        assert memory.failure_count(spec()) == 1
        assert llm.calls["code"] == 2 and llm.calls["format_review"] == 1


    def test_service_error_resumes_inside_attempt_without_reexecuting_completed_stages(self):
        tmp_path = self.tmp_path
        llm = ScriptedLLM(final_review=[ServiceError("quota unavailable")])
        executor = ScriptedExecutor()
        developer, memory = coder(tmp_path, llm, executor)
        with self.assertRaises(ServiceError):
            developer.develop([spec()], tmp_path / "round")
        assert memory.failure_count(spec()) == 0
        developer, memory = coder(tmp_path, llm, executor)
        assert developer.develop([spec()], tmp_path / "round")[0].implemented
        assert executor.calls == 1
        assert llm.calls == Counter(code=1, format_review=1, code_review=1, final_review=2)


    def test_semantically_invalid_code_response_allows_other_factors_to_complete(self):
        tmp_path = self.tmp_path
        llm = ScriptedLLM(code=[{"wrong": "key"}, {"code": "print('valid')"}])
        cfg = config(tmp_path)
        cfg.max_coding_attempts = 1
        developer, memory = coder(tmp_path, llm, ScriptedExecutor(), cfg)
        results = developer.develop([spec("bad"), spec("good")], tmp_path / "round")
        assert [f.implemented for f in results] == [False, True]
        assert "code" in results[0].error
        assert memory.success(spec("good")) is not None


    def test_exact_success_reuses_snapshot_and_does_not_call_model_or_executor(self):
        tmp_path = self.tmp_path
        llm = ScriptedLLM()
        executor = ScriptedExecutor()
        developer, memory = coder(tmp_path, llm, executor)
        first = developer.develop([spec()], tmp_path / "first")[0]
        Path(first.code_path).write_text("mutated historical file")
        call_counts = llm.calls.copy()
        second = developer.develop([spec()], tmp_path / "second")[0]
        assert second.implemented
        assert Path(second.code_path).read_text() == "print('factor')"
        assert llm.calls == call_counts and executor.calls == 1


    def test_memory_returns_failure_to_success_pairs_and_caches_embeddings(self):
        tmp_path = self.tmp_path
        llm = ScriptedLLM()
        memory = CodingMemory(tmp_path / "memory.json", llm, config(tmp_path))
        for name, number, accepted in [("known", 1, False), ("known", 2, True), ("new", 1, False)]:
            code = tmp_path / f"{name}_{number}.py"
            code.write_text("print('fixed')" if accepted else "raise ValueError('broken')")
            execution = ExecutionResult(accepted, stdout="" if accepted else "ValueError: broken")
            attempt = CodingAttempt(number, str(code), execution, final_feedback="checked", accepted=accepted)
            memory.record(spec(name), attempt)
        context = memory.context(spec("new"))
        assert context["similar_error_repairs"][0]["corrected"]["factor"]["name"] == "known"
        assert context["similar_successful_implementations"][0]["factor"]["name"] == "known"
        first_count = llm.calls["embed"]
        reloaded = CodingMemory(tmp_path / "memory.json", llm, config(tmp_path))
        assert reloaded.context(spec("new")) == context
        assert llm.calls["embed"] == first_count

    def test_completed_factor_mutation_is_rejected_on_resume(self):
        tmp_path = self.tmp_path
        llm = ScriptedLLM()
        executor = ScriptedExecutor()
        developer, _ = coder(tmp_path, llm, executor)
        artifact = developer.develop([spec()], tmp_path / "round")[0]
        Path(artifact.code_path).write_text("print('changed after validation')")
        with self.assertRaisesRegex(ValueError, "changed"):
            developer.develop([spec()], tmp_path / "round")

    def test_interrupted_factor_mutation_is_rejected_before_reusing_judgement(self):
        tmp_path = self.tmp_path
        llm = ScriptedLLM(final_review=[ServiceError("pause")])
        executor = ScriptedExecutor()
        developer, _ = coder(tmp_path, llm, executor)
        with self.assertRaises(ServiceError):
            developer.develop([spec()], tmp_path / "round")
        code = tmp_path / "round" / spec().identity / "attempt_001" / "factor.py"
        code.write_text("print('changed after execution')")
        with self.assertRaisesRegex(ValueError, "changed"):
            developer.develop([spec()], tmp_path / "round")
        assert executor.calls == 1 and llm.calls["final_review"] == 1

if __name__ == "__main__":
    unittest.main()
