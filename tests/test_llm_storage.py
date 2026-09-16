"""服务与存储边界验收：SDK 完全替身，任何测试都不发送真实 API 请求。"""

from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_agent.config import AgentConfig
from factor_agent.errors import ServiceError
from factor_agent.llm import LLMClient, ReplayClient
from factor_agent.models import BacktestResult, Hypothesis, RunState
from factor_agent.storage import RunStore


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(
            id=f"fake-{len(self.calls)}", model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=value), finish_reason="stop")],
            usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5}),
        )


class LLMAndReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.secret = "unit-test-secret-12345-do-not-record"
        environment = patch.dict(os.environ, {"OPENAI_API_KEY": self.secret})
        environment.start()
        self.addCleanup(environment.stop)
        # If a test accidentally forgets its fake client, fail before network I/O.
        forbidden = patch("openai.OpenAI", side_effect=AssertionError("real API client construction is forbidden"))
        forbidden.start()
        self.addCleanup(forbidden.stop)
        self.config = SimpleNamespace(chat_model="openai/requested-model", embedding_model="openai/requested-embedding",
                                      api_base="https://unit-test.invalid/v1", api_timeout=5, api_retries=3)

    def client(self, responses):
        completions = FakeCompletions(responses)
        client = LLMClient(self.config, self.root / "llm")
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        return client, completions

    def test_identical_completed_call_is_cached_across_client_restart(self):
        client, sdk = self.client(['{"hypothesis": "test"}'])
        expected = client.complete("system", "user", purpose="hypothesis/round_000")
        self.assertEqual(client.complete("system", "user", purpose="hypothesis/round_000"), expected)
        restarted, never_called = self.client([])
        self.assertEqual(restarted.complete("system", "user", purpose="hypothesis/round_000"), expected)
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(never_called.calls, [])
        self.assertEqual(sdk.calls[0]["model"], "requested-model")

    def test_invalid_json_resume_preserves_both_raw_responses(self):
        client, sdk = self.client(["malformed JSON", '{"hypothesis": "repaired response"}'])
        with self.assertRaises(ServiceError):
            client.complete("system", "user", purpose="hypothesis/round_000")
        self.assertEqual(client.complete("system", "user", purpose="hypothesis/round_000"),
                         {"hypothesis": "repaired response"})
        responses = [json.loads(path.read_text()) for path in (self.root / "llm").rglob("response*.json")]
        self.assertEqual(len(sdk.calls), 2)
        self.assertEqual(sorted(value["text"] for value in responses),
                         sorted(["malformed JSON", '{"hypothesis": "repaired response"}']))

    def test_saved_raw_response_recovers_after_interruption_without_another_api_call(self):
        from factor_agent.storage import write_json

        client, sdk = self.client(['{"hypothesis": "already received"}'])

        def interrupt_result_save(path, value):
            if Path(path).name == "result.json":
                raise KeyboardInterrupt()
            write_json(path, value)

        with patch("factor_agent.llm.write_json", side_effect=interrupt_result_save):
            with self.assertRaises(KeyboardInterrupt):
                client.complete("system", "user", purpose="hypothesis/round_000")
        restarted, never_called = self.client([])
        self.assertEqual(restarted.complete("system", "user", purpose="hypothesis/round_000"),
                         {"hypothesis": "already received"})
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(never_called.calls, [])
        self.assertEqual(len(list((self.root / "llm").rglob("response_*.json"))), 1)

    def test_quota_failure_pauses_without_retry_or_fallback_and_redacts_error(self):
        import httpx
        from openai import RateLimitError

        response = httpx.Response(429, request=httpx.Request("POST", "https://unit-test.invalid/v1/chat/completions"))
        error = RateLimitError("quota exhausted: " + self.secret, response=response,
                               body={"error": {"type": "insufficient_quota"}})
        client, sdk = self.client([error])
        with self.assertRaises(ServiceError) as caught:
            client.complete("system", "user", purpose="hypothesis/round_000")
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(sdk.calls[0]["model"], "requested-model")
        self.assertNotIn(self.secret, str(caught.exception))
        self.assert_no_secret_artifacts()

    def test_api_key_is_not_serialized_and_echoed_secret_is_redacted(self):
        client, sdk = self.client([json.dumps({"explanation": self.secret})])
        result = client.complete("ordinary system message", "error log accidentally includes " + self.secret,
                                 purpose="review/test")
        self.assertEqual(result["explanation"], "[REDACTED_SECRET]")
        self.assertNotIn(self.secret, sdk.calls[0]["messages"][1]["content"])
        self.assert_no_secret_artifacts()

    def assert_no_secret_artifacts(self):
        paths = list(self.root.rglob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            self.assertNotIn(self.secret, path.read_text(), str(path))

    def make_replay(self):
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps({"responses": [
            {"purpose": "hypothesis", "response": {"hypothesis": "offline"}},
            {"purpose": "design", "response": {"factor": {"description": "offline"}}},
        ]}))
        return fixture, ReplayClient(fixture, self.root / "replay")

    def test_replay_mismatch_is_non_consuming_and_exhaustion_never_calls_api(self):
        fixture, client = self.make_replay()
        with self.assertRaises(ServiceError):
            client.complete("system", "user", purpose="design/round_000")
        first = client.complete("system", "user", purpose="hypothesis/round_000")
        self.assertEqual(first, {"hypothesis": "offline"})
        restarted = ReplayClient(fixture, self.root / "replay")
        self.assertEqual(restarted.complete("system", "user", purpose="hypothesis/round_000"), first)
        restarted.complete("system", "user", purpose="design/round_000")
        with self.assertRaises(ServiceError):
            restarted.complete("system", "user", purpose="review/round_000")

    def test_replay_fixture_change_cannot_reuse_prior_cursor(self):
        fixture, client = self.make_replay()
        client.complete("system", "user", purpose="hypothesis/round_000")
        fixture.write_text(json.dumps({"responses": []}))
        with self.assertRaises(ValueError):
            ReplayClient(fixture, self.root / "replay")


class InputFingerprintTests(unittest.TestCase):
    def test_parallel_data_fingerprint_matches_sequential_content_digest(self):
        from factor_agent.execution import data_fingerprint, file_hash, fingerprint

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number in reversed(range(17)):
                path = root / f"group_{number % 3}" / f"feature_{number:02d}.bin"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(bytes([number]) * (number + 1))
            (root / "empty_directory").mkdir()
            parallel = data_fingerprint(root)
            sequential = fingerprint({str(path.relative_to(root)): file_hash(path)
                                      for path in sorted(root.rglob("*")) if path.is_file()})
            self.assertEqual(parallel, sequential)

    def test_relative_interpreters_resolve_against_yaml_directory_and_bare_commands_use_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / "settings"
            settings.mkdir()
            path = settings / "config.yaml"
            path.write_text("python_executable: ./venv/bin/python\nqlib_python: ../qlib_env/bin/python\n")
            config = AgentConfig.from_file(path)
            self.assertEqual(Path(config.python_executable).resolve(), settings / "venv/bin/python")
            self.assertEqual(Path(config.qlib_python).resolve(), root / "qlib_env/bin/python")
            path.write_text("python_executable: python3\nqlib_python: python3\n")
            config = AgentConfig.from_file(path)
            self.assertEqual(Path(config.python_executable).resolve(), Path(shutil.which("python3")).resolve())
            self.assertEqual(Path(config.qlib_python).resolve(), Path(shutil.which("python3")).resolve())

    def test_changed_market_inputs_reject_resume_even_with_same_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "full/daily_pv.h5", root / "debug/daily_pv.h5", root / "qlib/calendars/day.txt"]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_bytes(b"original")
            options = dict(data_dir=root / "full", debug_data_dir=root / "debug", qlib_provider=root / "qlib",
                           python_executable=sys.executable, qlib_python=sys.executable,
                           chat_model="fake-model", embedding_model="fake-embedding", api_base="https://unit-test.invalid")
            # Package identity is constant here; this test isolates data-content identity.
            with patch("factor_agent.execution.runtime_fingerprint", return_value={"python": "fixture-runtime"}):
                for path in paths:
                    with self.subTest(path=path.relative_to(root)):
                        store = RunStore(root / "state" / path.parent.name)
                        original_fingerprint = AgentConfig(**options).fingerprint()
                        store.save(RunState(config_fingerprint=original_fingerprint))
                        stat = path.stat()
                        path.write_bytes(b"modified")
                        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                        updated_fingerprint = AgentConfig(**options).fingerprint()
                        self.assertNotEqual(original_fingerprint, updated_fingerprint)
                        with self.assertRaises(ValueError):
                            store.load(updated_fingerprint)
                        path.write_bytes(b"original")


class ResearchMetricBoundaryTests(unittest.TestCase):
    def test_research_feedback_receives_only_original_three_metrics_without_mutating_results(self):
        from factor_agent.research import Researcher

        required = {"IC", "1day.excess_return_with_cost.annualized_return",
                    "1day.excess_return_with_cost.max_drawdown"}
        metrics = {"IC": 0.03, "1day.excess_return_with_cost.annualized_return": 0.08,
                   "1day.excess_return_with_cost.max_drawdown": -0.12,
                   "Rank IC": 0.04, "1day.excess_return_with_cost.information_ratio": 0.9}
        captured = []

        def complete(system, user, **kwargs):
            captured.append(json.loads(user))
            return {"Observations": "observed", "Feedback for Hypothesis": "evaluated",
                    "New Hypothesis": "next", "Reasoning": "reason", "Replace Best Result": False}

        with tempfile.TemporaryDirectory() as temporary:
            researcher = Researcher(SimpleNamespace(complete=complete), SimpleNamespace(debug_data_dir=Path(temporary)))
            current = BacktestResult(dict(metrics), "current")
            best = BacktestResult(dict(metrics), "best")
            researcher.review(Hypothesis("hypothesis", "reason"), [], current, best, [])
        self.assertEqual(set(captured[0]["Current combined result"]), required)
        self.assertEqual(set(captured[0]["SOTA result"]), required)
        self.assertEqual(current.metrics, metrics)
        self.assertEqual(best.metrics, metrics)


class CLIPersistenceTests(unittest.TestCase):
    def test_run_directory_lock_excludes_second_writer_and_releases_on_error(self):
        from factor_agent.cli import run_lock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "interrupted work"):
                with run_lock(root):
                    with self.assertRaises(ValueError):
                        with run_lock(root):
                            self.fail("second independently opened writer obtained the same run lock")
                    raise RuntimeError("interrupted work")
            with run_lock(root):
                self.assertTrue((root / ".run.lock").is_file())

    def test_replay_can_resume_after_baseline_before_first_scripted_response(self):
        from factor_agent.cli import _run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = RunState(config_fingerprint="fixture")
            RunStore(root).save(state)
            (root / "run_mode.json").write_text(json.dumps({"mode": "offline_replay", "rounds_target": 1}))
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps({"responses": []}))
            args = SimpleNamespace(command="replay", run_dir=root, fixture=fixture, rounds=1)
            config = SimpleNamespace(public_dict=lambda: {}, fingerprint=lambda: "fixture")
            with patch("factor_agent.pipeline.Pipeline.run", return_value=state), redirect_stdout(io.StringIO()):
                result = _run(args, config)
            self.assertEqual(result, 0)

    def test_rejected_resume_does_not_overwrite_original_config_record(self):
        from factor_agent.cli import _run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            RunStore(root).save(RunState(config_fingerprint="original"))
            original_config = {"setting": "original"}
            (root / "config.json").write_text(json.dumps(original_config))
            (root / "run_mode.json").write_text(json.dumps({"mode": "live_api", "rounds_target": 1}))
            args = SimpleNamespace(command="resume", run_dir=root, rounds=1)
            config = SimpleNamespace(public_dict=lambda: {"setting": "changed"}, fingerprint=lambda: "changed")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-never-used"}):
                with self.assertRaises(ValueError):
                    _run(args, config)
            self.assertEqual(json.loads((root / "config.json").read_text()), original_config)


if __name__ == "__main__":
    unittest.main()
