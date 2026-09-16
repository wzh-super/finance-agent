"""CoSTEER batch barriers, retry budgets and recovery; no API or training."""

from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from factor_agent.coding import FactorCoder
from factor_agent.errors import ServiceError
from factor_agent.memory import CodingMemory
from factor_agent.models import ExecutionResult, FactorSpec
from factor_agent.storage import read_json


class BatchEvolutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.specs = [FactorSpec(name, "Test factor", "close", {"close": "price"}) for name in ("a", "b", "c")]
        self.names = {spec.identity: spec.name for spec in self.specs}
        self.cfg = SimpleNamespace(debug_data_dir=self.root, max_coding_attempts=3, max_task_failures=20,
                                   memory_top_k=1, memory_recent_failures=3, embedding_model="offline-test")
        self.trace, self.contexts, self.events = [], {}, []
        self.success_at = {"a": 3, "b": 1, "c": 2}
        self.pause = None
        self.llm = SimpleNamespace(complete=self.complete, embed=lambda texts: [[1.0, 0.0] for _ in texts])
        self.executor = SimpleNamespace(run=self.execute)

    def developer(self):
        memory = CodingMemory(self.root / "memory.json", self.llm, self.cfg)
        coder = FactorCoder(self.llm, self.executor, memory, self.cfg,
                            event=lambda stage, message, **details: self.events.append(details))
        return coder, memory

    def complete(self, system, user, *, purpose, response_json=True):
        stage = purpose.split("/")[0]
        directory = Path(purpose.split("/", 1)[1])
        name = self.names[directory.parent.name]
        number = int(directory.name.split("_")[1])
        key = (stage, name, number)
        self.trace.append(key)
        if self.pause == key:
            self.pause = None
            raise ServiceError("Offline scripted interruption")
        if stage == "code":
            self.contexts[name, number] = json.loads(user.split("\nCoding memory:\n")[1])
        return {
            "code": {"code": f"print('{name}-{number}')"},
            "format_review": {"output_format_decision": True, "output_format_feedback": "Valid output"},
            "code_review": "Formula checked",
            "final_review": {"final_decision": number >= self.success_at[name], "final_feedback": "Review complete"},
        }[stage]

    def execute(self, path, spec, directory):
        number = int(path.parent.name.split("_")[1])
        self.trace.append(("execute", spec.name, number))
        return ExecutionResult(True, stats={"rows": 1})

    def run_batch(self):
        coder, memory = self.developer()
        return coder.develop(self.specs, self.root / "coding"), memory

    def test_generation_and_evaluation_follow_batch_order_and_skip_approved_tasks(self):
        results, memory = self.run_batch()
        calls = [(stage, name, number) for stage, name, number in self.trace if stage in {"code", "execute"}]
        self.assertEqual(calls, [
            ("code", "a", 1), ("code", "b", 1), ("code", "c", 1),
            ("execute", "a", 1), ("execute", "b", 1), ("execute", "c", 1),
            ("code", "a", 2), ("code", "c", 2), ("execute", "a", 2), ("execute", "c", 2),
            ("code", "a", 3), ("execute", "a", 3),
        ])
        self.assertEqual([len(item.attempts) for item in results], [3, 1, 2])
        self.assertTrue(all(item.implemented for item in results))
        self.assertEqual([(item["spec"]["name"], item["attempt"]["number"]) for item in memory.data["attempts"]],
                         [("a", 1), ("b", 1), ("c", 1), ("a", 2), ("c", 2), ("a", 3)])

    def test_all_contexts_are_frozen_until_the_next_evolution_round(self):
        self.run_batch()
        for name in ("a", "b", "c"):
            context = self.contexts[name, 1]
            self.assertEqual(context["previous_failed_attempts"], [])
            self.assertEqual(context["similar_successful_implementations"], [])
        for name in ("a", "c"):
            context = self.contexts[name, 2]
            self.assertEqual(len(context["previous_failed_attempts"]), 1)
            self.assertEqual([item["factor"]["name"] for item in context["similar_successful_implementations"]], ["b"])
        self.assertEqual(len(self.contexts["a", 3]["previous_failed_attempts"]), 2)

    def test_round_budget_is_shared_but_not_a_total_number_of_factor_attempts(self):
        self.cfg.max_coding_attempts = 2
        self.success_at = {name: 99 for name in self.success_at}
        results, memory = self.run_batch()
        self.assertEqual([len(item.attempts) for item in results], [2, 2, 2])
        self.assertFalse(any(item.implemented for item in results))
        self.assertEqual(len(memory.data["attempts"]), 6)
        self.assertEqual(len(list((self.root / "coding").glob("evolution_*"))), 2)

    def test_generation_pause_never_evaluates_a_partial_batch(self):
        self.pause = ("code", "b", 1)
        coder, memory = self.developer()
        with self.assertRaises(ServiceError):
            coder.develop(self.specs, self.root / "coding")
        self.assertEqual(memory.data["attempts"], [])
        self.assertFalse(any(stage == "execute" for stage, _, _ in self.trace))
        self.assertTrue(all((self.root / "coding" / item.identity / "attempt_001/retrieved_memory.json").exists()
                            for item in self.specs))
        results, _ = self.run_batch()
        self.assertTrue(all(item.implemented for item in results))
        counts = Counter(self.trace)
        self.assertEqual(counts["code", "a", 1], 1)
        self.assertEqual(counts["code", "b", 1], 2)
        self.assertEqual(counts["execute", "a", 1], 1)

    def test_review_pause_keeps_finished_attempt_private_and_resumes_once(self):
        self.pause = ("final_review", "b", 1)
        coder, memory = self.developer()
        with self.assertRaises(ServiceError):
            coder.develop(self.specs, self.root / "coding")
        self.assertEqual(memory.data["attempts"], [])
        self.assertTrue((self.root / "coding" / self.specs[0].identity / "attempt_001/attempt.json").exists())
        results, memory = self.run_batch()
        self.assertTrue(all(item.implemented for item in results))
        counts = Counter(self.trace)
        self.assertEqual(counts["final_review", "a", 1], 1)
        self.assertEqual(counts["final_review", "b", 1], 2)
        self.assertEqual(counts["execute", "a", 1], 1)
        self.assertEqual(counts["execute", "b", 1], 1)
        self.assertEqual(self.contexts["c", 1]["similar_successful_implementations"], [])
        self.assertEqual(len(memory.data["attempts"]), 6)

    def test_crash_after_memory_commit_reuses_contexts_without_duplicate_attempts(self):
        coder, memory = self.developer()
        record = memory.record_batch

        def commit_then_pause(attempts):
            record(attempts)
            raise KeyboardInterrupt()

        with patch.object(memory, "record_batch", side_effect=commit_then_pause):
            with self.assertRaises(KeyboardInterrupt):
                coder.develop(self.specs, self.root / "coding")
        self.assertEqual(len(read_json(self.root / "memory.json")["attempts"]), 3)
        self.assertFalse((self.root / "coding/evolution_001/completed.json").exists())
        results, memory = self.run_batch()
        self.assertTrue(all(item.implemented for item in results))
        self.assertEqual(len(memory.data["attempts"]), 6)
        self.assertTrue(all(Counter(self.trace)["code", name, 1] == 1 for name in ("a", "b", "c")))

    def test_memory_commit_failure_is_atomic_in_memory_and_on_disk(self):
        coder, memory = self.developer()
        with patch.object(memory, "_save", side_effect=OSError("Offline disk error")):
            with self.assertRaises(OSError):
                coder.develop(self.specs, self.root / "coding")
        self.assertEqual(memory.data["attempts"], [])
        self.assertFalse((self.root / "memory.json").exists())
        results = coder.develop(self.specs, self.root / "coding")
        self.assertTrue(all(item.implemented for item in results))
        self.assertEqual(len(memory.data["attempts"]), 6)

    def test_cumulative_task_failures_stop_only_that_task_and_exact_success_reuses(self):
        self.cfg.max_task_failures = 2
        self.success_at = {"a": 99, "b": 1, "c": 1}
        first, _ = self.run_batch()
        self.assertEqual([len(item.attempts) for item in first], [2, 1, 1])
        self.assertIn("cumulative", first[0].error)
        previous_calls = list(self.trace)
        coder, _ = self.developer()
        second = coder.develop(self.specs, self.root / "next_research_round")
        self.assertEqual(self.trace, previous_calls)
        self.assertEqual([item.implemented for item in second], [False, True, True])
        self.assertTrue(all(not item.attempts for item in second))
        self.assertEqual(second[1].error, "")

    def test_completed_batch_is_read_only_and_reordered_specs_are_rejected(self):
        self.run_batch()
        previous_calls = list(self.trace)
        self.run_batch()
        self.assertEqual(self.trace, previous_calls)
        coder, _ = self.developer()
        with self.assertRaisesRegex(ValueError, "schedule changed"):
            coder.develop(list(reversed(self.specs)), self.root / "coding")

    def test_missing_frozen_context_cannot_be_reconstructed_from_newer_memory(self):
        self.pause = ("final_review", "b", 1)
        coder, _ = self.developer()
        with self.assertRaises(ServiceError):
            coder.develop(self.specs, self.root / "coding")
        (self.root / "coding" / self.specs[0].identity / "attempt_001/retrieved_memory.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing frozen coding context"):
            self.run_batch()

    def test_legacy_per_factor_checkpoint_is_preserved_and_refused(self):
        legacy = self.root / "coding" / self.specs[0].identity / "attempt_001"
        legacy.mkdir(parents=True)
        code = legacy / "factor.py"
        code.write_text("print('historical code')")
        with self.assertRaisesRegex(ValueError, "Legacy coding checkpoint"):
            self.run_batch()
        self.assertEqual(code.read_text(), "print('historical code')")
        self.assertFalse((self.root / "coding/schedule.json").exists())


if __name__ == "__main__":
    unittest.main()
