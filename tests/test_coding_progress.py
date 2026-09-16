"""Observability must preserve repair, cache and interruption behavior; no live API."""

from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from factor_agent.coding import FactorCoder
from factor_agent.errors import ServiceError
from factor_agent.llm import LLMClient, ReplayClient
from factor_agent.memory import CodingMemory
from factor_agent.models import BacktestResult, ExecutionResult, FactorSpec
from factor_agent.research import Researcher


class FakeResearchLLM:
    def __init__(self, pause_final=False):
        self.calls = Counter()
        self.pause_final = pause_final

    def complete(self, system, user, *, purpose, response_json=True):
        stage = purpose.split("/")[0]
        self.calls[stage] += 1
        if stage == "final_review" and self.pause_final:
            self.pause_final = False
            raise ServiceError("scripted pause")
        return {
            "code": {"code": "print('generated factor')"},
            "format_review": {"output_format_decision": True, "output_format_feedback": "Valid single column"},
            "code_review": "Keep the grouping by instrument.",
            "final_review": {"final_decision": True, "final_feedback": "Formula matches the implementation."},
        }[stage]

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeExecutor:
    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    def run(self, code_path, spec, directory, *, full=False):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return ExecutionResult(False, stdout="ValueError: invalid shape", error="invalid shape", returncode=1)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "result.h5"
        output.write_text("fake output")
        return ExecutionResult(True, data_path=str(output), stats={"rows": 12}, returncode=0)


class CodingProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = []
        self.event = lambda stage, message, **details: self.events.append({"stage": stage, "message": message, **details})
        self.config = SimpleNamespace(debug_data_dir=self.root, max_coding_attempts=3, max_task_failures=20,
                                      memory_top_k=1, memory_recent_failures=3, embedding_model="fake-embedding")
        self.spec = FactorSpec("factor_a", "test factor", "close", {"close": "price"})

    def developer(self, llm, executor):
        memory = CodingMemory(self.root / "memory.json", llm, self.config, event=self.event)
        return FactorCoder(llm, executor, memory, self.config, event=self.event), memory

    def test_failure_repair_and_full_review_texts_are_observable(self):
        llm, executor = FakeResearchLLM(), FakeExecutor(fail_first=True)
        coder, _ = self.developer(llm, executor)
        artifact = coder.develop([self.spec], self.root / "round")[0]
        self.assertTrue(artifact.implemented)
        self.assertEqual(executor.calls, 2)
        kinds = [event["kind"] for event in self.events]
        self.assertLess(kinds.index("repair_planned"), len(kinds) - 1)
        debug = [event for event in self.events if event["kind"] == "debug_result"]
        self.assertEqual([event["success"] for event in debug], [False, True])
        self.assertTrue(all(not event["cached"] for event in debug))
        generated = [event for event in self.events if event["kind"] == "code_generated"]
        self.assertEqual([event["attempt"] for event in generated], [1, 2])
        self.assertEqual(generated[0]["code"], "print('generated factor')")
        semantic = [event for event in self.events if event["kind"] == "code_review"]
        self.assertEqual(semantic[-1]["feedback"], "Keep the grouping by instrument.")
        final = [event for event in self.events if event["kind"] == "final_review"]
        self.assertEqual([event["accepted"] for event in final], [False, True])
        self.assertEqual(llm.calls["code"], 2)

    def test_paused_attempt_logs_cached_steps_without_repeating_them(self):
        llm, executor = FakeResearchLLM(pause_final=True), FakeExecutor()
        coder, memory = self.developer(llm, executor)
        with self.assertRaises(ServiceError):
            coder.develop([self.spec], self.root / "round")
        self.assertEqual(memory.failure_count(self.spec), 0)
        self.assertIn("attempt_paused", [event["kind"] for event in self.events])
        self.events.clear()
        coder, _ = self.developer(llm, executor)
        self.assertTrue(coder.develop([self.spec], self.root / "round")[0].implemented)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(llm.calls, Counter(code=1, format_review=1, code_review=1, final_review=2))
        resumed = [event for event in self.events if event["kind"] == "attempt_start"]
        self.assertTrue(resumed[0]["resumed"])
        cached = [event for event in self.events if event["kind"] == "stage_cache"]
        self.assertEqual({event["stage"] for event in cached}, {"format_review", "code_review"})
        self.assertTrue(next(event for event in self.events if event["kind"] == "debug_result")["cached"])
        self.assertNotIn("code_generated", [event["kind"] for event in self.events])

    def client(self):
        cfg = SimpleNamespace(chat_model="openai/fake-model", embedding_model="fake-embedding",
                              api_base="https://private-endpoint.invalid/v1", api_retries=1)
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="fake-response", model="fake-model",
                                   choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'), finish_reason="stop")],
                                   usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5}))

        client = LLMClient(cfg, self.root / "llm", event=self.event)
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return client, calls

    def test_llm_completed_and_cached_events_keep_usage_and_artifact_refs(self):
        client, calls = self.client()
        first = client.complete("full private system prompt", "full user prompt", purpose="hypothesis/0")
        self.assertEqual(client.complete("full private system prompt", "full user prompt", purpose="hypothesis/0"), first)
        self.assertEqual(len(calls), 1)
        self.assertEqual([e["kind"] for e in self.events], ["request_start", "request_completed", "request_cache"])
        completed = self.events[1]
        self.assertEqual(completed["usage"], {"prompt_tokens": 10, "completion_tokens": 5})
        self.assertTrue(Path(completed["response_path"]).is_file())
        text = json.dumps(self.events, default=str)
        self.assertNotIn("private-endpoint", text)
        self.assertNotIn("full private system prompt", text)
        self.assertNotIn("full user prompt", text)

    def test_raw_response_recovery_is_labelled_and_never_reissues_request(self):
        client, calls = self.client()
        client.complete("system", "user", purpose="hypothesis/0")
        next((self.root / "llm").rglob("result.json")).unlink()
        self.events.clear()
        client.complete("system", "user", purpose="hypothesis/0")
        self.assertEqual(len(calls), 1)
        self.assertEqual([event["kind"] for event in self.events], ["response_recovered"])
        self.assertTrue(self.events[0]["cached"])

    def test_transient_retry_event_records_status_without_endpoint(self):
        import httpx
        from openai import InternalServerError

        client, _ = self.client()
        client.config.api_retries = 2
        original = client._client.chat.completions.create
        attempts = []

        def transient(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                response = httpx.Response(503, request=httpx.Request("POST", client.config.api_base))
                raise InternalServerError("temporary unavailable", response=response, body=None)
            return original(**kwargs)

        client._client.chat.completions.create = transient
        with patch("factor_agent.llm.time.sleep"):
            client.complete("system", "user", purpose="hypothesis/0")
        self.assertEqual(len(attempts), 2)
        retry = next(event for event in self.events if event["kind"] == "request_retry")
        self.assertEqual((retry["status"], retry["attempt"], retry["model"]), (503, 1, "fake-model"))
        self.assertTrue(Path(retry["artifact"]).is_file())
        self.assertNotIn("private-endpoint", json.dumps(self.events, default=str))

    def test_replay_events_are_explicitly_offline_including_cache(self):
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps({"responses": [{"purpose": "code", "response": {"code": "pass"}}]}))
        replay = ReplayClient(fixture, self.root / "replay", event=self.event)
        replay.complete("system", "user", purpose="code/0")
        replay.complete("system", "user", purpose="code/0")
        replay.embed(["example"])
        self.assertTrue(all(event["mode"] == "offline_replay" for event in self.events))
        self.assertIn("request_cache", [event["kind"] for event in self.events])
        self.assertIn("embedding_replay", [event["kind"] for event in self.events])

    def test_english_language_rule_reaches_every_research_and_coding_role(self):
        captured = {}
        coding_llm = FakeResearchLLM()

        def complete(system, user, *, purpose, response_json=True):
            stage = purpose.split("/")[0]
            captured[stage] = system
            research_replies = {
                "hypothesis": {"hypothesis": "Momentum predicts returns.", "reason": "Test price persistence."},
                "design": {"momentum": {"description": "Price momentum", "formulation": "close",
                                          "variables": {"close": "Closing price"}}},
                "feedback": {"Observations": "Returns improved.", "Feedback for Hypothesis": "Supported.",
                             "New Hypothesis": "Test a longer window.", "Reasoning": "Compare horizons.",
                             "Replace Best Result": True},
            }
            if stage in research_replies:
                return research_replies[stage]
            return coding_llm.complete(system, user, purpose=purpose, response_json=response_json)

        llm = SimpleNamespace(complete=complete, embed=coding_llm.embed)
        researcher = Researcher(llm, self.config)
        hypothesis = researcher.propose([])
        specs = researcher.design(hypothesis, [], [])
        coder, _ = self.developer(llm, FakeExecutor())
        factors = coder.develop(specs, self.root / "round")
        metrics = {"IC": 0.04, "1day.excess_return_with_cost.annualized_return": 0.08,
                   "1day.excess_return_with_cost.max_drawdown": -0.1}
        result = BacktestResult(metrics, str(self.root), factor_names=[specs[0].name])
        researcher.review(hypothesis, factors, result, result, [])
        self.assertEqual(set(captured), {"hypothesis", "design", "feedback", "code",
                                         "format_review", "code_review", "final_review"})
        for stage, system in captured.items():
            with self.subTest(stage=stage):
                self.assertIn("Write all natural-language responses in English", system)
                self.assertIn("factor names and descriptions", system)
                self.assertIn("generated Python comments and docstrings", system)
                self.assertIn("Preserve the required JSON keys", system)


if __name__ == "__main__":
    unittest.main()
