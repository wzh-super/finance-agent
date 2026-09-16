"""Store coding attempts and retrieve successful implementations and repairs.

JSON records express CoSTEER task, success and error relationships directly.
Embeddings retrieve similar implementations; error signatures retrieve repairs.
Correct implementations remain useful even when they do not improve returns.
"""

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import re

from .errors import ServiceError
from .models import CodingAttempt, FactorSpec
from .progress import null_event
from .storage import read_json, write_json


def error_signature(attempt: dict) -> str:
    """Match Python error classes first; otherwise use the value/final rejection."""
    execution = attempt["execution"]
    text = execution.get("error", "") + "\n" + execution.get("stdout", "")
    matches = re.findall(r"(?:^|\n)([\w.]+(?:Error|Exception))\s*:", text)
    if matches:
        return matches[-1]
    detail = execution.get("error") or attempt.get("value_feedback") or attempt.get("final_feedback", "")
    detail = re.sub(r"\b\d+(?:\.\d+)?\b", "#", detail)
    return detail.strip()[:300] or "Unknown failure"


class CodingMemory:
    """Persistent attempt records and retrieval, independent from research acceptance."""

    def __init__(self, path: Path, llm, config, *, event=None):
        self.path = Path(path)
        self.llm = llm
        self.config = config
        self.event = event or null_event
        self.data = read_json(self.path) if self.path.exists() else {"version": 1, "attempts": [], "embeddings": {}}
        if self.data.get("version") != 1:
            raise ValueError("Unsupported coding-memory version")
        self.event("memory", "Coding memory loaded" if self.path.exists() else "Initialized empty coding memory",
                   kind="memory_loaded", attempts=len(self.data["attempts"]),
                   successful_tasks=len({item["identity"] for item in self.data["attempts"]
                                         if item["attempt"]["accepted"]}), artifact=self.path)

    def _save(self):
        write_json(self.path, self.data)

    def attempts(self, spec: FactorSpec) -> list[dict]:
        return [item for item in self.data["attempts"] if item["identity"] == spec.identity]

    def success(self, spec: FactorSpec) -> dict | None:
        return next((item for item in self.attempts(spec) if item["attempt"]["accepted"]), None)

    def failure_count(self, spec: FactorSpec) -> int:
        return sum(not item["attempt"]["accepted"] for item in self.attempts(spec))

    def record(self, spec: FactorSpec, attempt: CodingAttempt) -> None:
        """Snapshot code rather than referring to a mutable execution workspace."""
        self.record_batch([(spec, attempt)])

    def record_batch(self, attempts: list[tuple[FactorSpec, CodingAttempt]]) -> None:
        """Publish an evaluated evolution round atomically, with idempotent recovery.

        The coder calls this only after every scheduled task has been evaluated.
        Until then, the saved attempt files are evidence, not retrieval memory.
        """
        existing = {item["key"]: item for item in self.data["attempts"]}
        additions, events = [], []
        for spec, attempt in attempts:
            key = f"{spec.identity}:{Path(attempt.code_path).resolve()}:{attempt.number}"
            code = Path(attempt.code_path).read_text()
            digest = hashlib.sha256(Path(attempt.code_path).read_bytes()).hexdigest()
            expected = attempt.execution.stats.get("code_sha256")
            if expected and expected != digest:
                raise ValueError(f"Code changed after execution at {attempt.code_path}; use a new run directory")
            item = {
                "key": key, "identity": spec.identity, "spec": asdict(spec),
                "code": code, "code_sha256": digest, "attempt": asdict(attempt),
                "error_signature": "" if attempt.accepted else error_signature(asdict(attempt)),
            }
            cached = key in existing
            if cached:
                if existing[key] != item:
                    raise ValueError(f"Recorded code or feedback changed at {attempt.code_path}; use a new run directory")
            else:
                additions.append(item)
                existing[key] = item
            events.append((spec, attempt, cached))
        if additions:
            previous = self.data["attempts"]
            self.data["attempts"] = previous + additions
            try:
                self._save()
            except BaseException:
                self.data["attempts"] = previous
                raise
        for spec, attempt, cached in events:
            self.event("memory", "Coding attempt already recorded; reusing the existing memory entry" if cached
                       else "Coding attempt saved to memory", kind="attempt_cached" if cached else "attempt_recorded",
                       factor=spec.name, identity=spec.identity, attempt=attempt.number,
                       accepted=attempt.accepted, cached=cached, failures=self.failure_count(spec),
                       code_path=attempt.code_path, artifact=self.path)

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        cache = self.data["embeddings"]
        model = getattr(self.config, "embedding_model", "")
        keys = [hashlib.sha256((model + "\n" + text).encode()).hexdigest() for text in texts]
        pending = dict((key, text) for key, text in zip(keys, texts) if key not in cache)
        self.event("memory", "Checking the embedding cache for similarity retrieval", kind="embedding_lookup",
                   cached_count=sum(key in cache for key in keys), missing_count=len(pending), model=model,
                   artifact=self.path)
        if pending:
            vectors = self.llm.embed(list(pending.values()))
            if len(vectors) != len(pending):
                raise ServiceError("Embedding service returned the wrong number of vectors")
            for key, vector in zip(pending, vectors):
                if not vector or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in vector):
                    raise ServiceError("Embedding service returned an empty or invalid vector")
                cache[key] = vector
            self._save()
        return [cache[key] for key in keys]

    def _similar_successes(self, spec: FactorSpec) -> list[dict]:
        successes = {}
        for item in self.data["attempts"]:
            if item["attempt"]["accepted"] and item["identity"] != spec.identity:
                successes.setdefault(item["identity"], item)
        items = list(successes.values())
        limit = self.config.memory_top_k
        if not items or limit <= 0:
            return []
        texts = [spec.describe()] + [FactorSpec(**item["spec"]).describe() for item in items]
        vectors = self._vectors(texts)
        query = vectors[0]

        def cosine(vector):
            if len(vector) != len(query):
                raise ServiceError("Embedding vectors have inconsistent dimensions")
            denominator = math.sqrt(sum(v * v for v in query) * sum(v * v for v in vector))
            return sum(a * b for a, b in zip(query, vector)) / denominator if denominator else 0.0

        ranked = sorted(zip(items, vectors[1:]), key=lambda pair: cosine(pair[1]), reverse=True)
        return [self._render(item) for item, _ in ranked[:limit]]

    @staticmethod
    def _render(item: dict) -> dict:
        attempt = item["attempt"]
        return {
            "factor": item["spec"],
            "code": item["code"],
            "feedback": {
                "execution_success": attempt["execution"]["success"],
                "execution_error": attempt["execution"]["error"],
                "execution_stdout": attempt["execution"]["stdout"][-4000:],
                "value_feedback": attempt["value_feedback"],
                "code_feedback": attempt["code_feedback"],
                "final_feedback": attempt["final_feedback"],
            },
        }

    def context(self, spec: FactorSpec) -> dict:
        """Retrieve recent failures, a runnable fallback, successful examples and repairs."""
        previous = [item for item in self.attempts(spec) if not item["attempt"]["accepted"]]
        runnable = next((item for item in reversed(previous) if item["attempt"]["execution"]["success"]), None)
        recent_limit = self.config.memory_recent_failures
        recent = previous[-recent_limit:] if recent_limit > 0 else []
        pairs = []
        if previous and self.config.memory_top_k > 0:
            signature = previous[-1]["error_signature"]
            for item in reversed(self.data["attempts"]):
                if item["attempt"]["accepted"] or item["error_signature"] != signature:
                    continue
                success = self.success(FactorSpec(**item["spec"]))
                if success is not None:
                    pairs.append({"error": signature, "failed": self._render(item), "corrected": self._render(success)})
                if len(pairs) >= self.config.memory_top_k:
                    break
        context = {
            "previous_failed_attempts": [self._render(item) for item in recent],
            "latest_runnable_attempt": self._render(runnable) if runnable else None,
            "similar_successful_implementations": self._similar_successes(spec),
            "similar_error_repairs": pairs,
        }
        self.event("memory", "Coding reference retrieval completed", kind="memory_retrieved", factor=spec.name,
                   identity=spec.identity, previous_failures=len(recent), has_runnable_attempt=runnable is not None,
                   similar_successes=len(context["similar_successful_implementations"]),
                   similar_error_repairs=len(pairs),
                   reference_factors=[item["factor"]["name"] for item in context["similar_successful_implementations"]],
                   artifact=self.path)
        return context
