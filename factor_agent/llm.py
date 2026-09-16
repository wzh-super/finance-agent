"""Interchangeable clients for live model APIs and explicit offline replay.

Requests are cached by their complete content and reused on resume. Raw text,
model identity and token usage are preserved. Service errors remain separate
from research feedback, and credentials are excluded from request artifacts.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .errors import ServiceError
from .progress import null_event
from .storage import read_json, write_json


def request_key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _model_name(name: str) -> str:
    # Existing RD-Agent .env files use LiteLLM's OpenAI provider prefix.
    return name.removeprefix("openai/")


def redact(text: str) -> str:
    for name, value in os.environ.items():
        if len(value) >= 8 and any(marker in name.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            text = text.replace(value, "[REDACTED_SECRET]")
    return text


def parse_response(raw: str, response_json: bool):
    if not raw.strip():
        raise ValueError("Empty response")
    value = json.loads(raw) if response_json else raw
    if response_json and not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


class LLMClient:
    """OpenAI-compatible Chat/Embedding API; uses exactly the configured models."""

    def __init__(self, config, directory: Path, *, event=None):
        self.config = config
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._client = None
        self.event = event or null_event

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ServiceError("OPENAI_API_KEY is missing; load your .env or use the offline replay command")
            self._client = OpenAI(api_key=key, base_url=self.config.api_base,
                                  timeout=self.config.api_timeout, max_retries=0)
        return self._client

    def _call(self, call, directory: Path, *, purpose="", model=""):
        """Retry transient service/network errors only; quota/auth failures pause."""
        from openai import APIConnectionError, APIStatusError, APITimeoutError
        for attempt in range(self.config.api_retries):
            try:
                return call()
            except (APIConnectionError, APIStatusError, APITimeoutError) as exc:
                status = getattr(exc, "status_code", None)
                message = redact(str(exc))
                error_path = directory / f"error_{time.time_ns()}.json"
                write_json(error_path, {"status": status, "message": message})
                retryable = status is None or status in (408, 409, 500, 502, 503, 504)
                if not retryable or attempt + 1 == self.config.api_retries:
                    self.event("llm", "Model service request failed; checkpoint preserved", kind="request_error",
                               purpose=purpose, model=model, status=status, attempt=attempt + 1,
                               error_type=type(exc).__name__, artifact=error_path)
                    raise ServiceError(f"API request failed; checkpoint preserved: {message}") from exc
                self.event("llm", "Transient model service failure; retrying the request", kind="request_retry",
                           purpose=purpose, model=model, status=status, attempt=attempt + 1,
                           max_attempts=self.config.api_retries, retry_after_seconds=min(2 ** attempt, 4),
                           error_type=type(exc).__name__, artifact=error_path)
                time.sleep(min(2 ** attempt, 4))

    def complete(self, system: str, user: str, *, purpose: str, response_json: bool = True):
        request = {"purpose": purpose, "model": _model_name(self.config.chat_model),
                   "api_base": self.config.api_base, "temperature": 0.5,
                   "messages": [{"role": "system", "content": redact(system)}, {"role": "user", "content": redact(user)}],
                   "response_json": response_json}
        directory = self.directory / "requests" / request_key(request)
        result_path = directory / "result.json"
        if result_path.is_file():
            self.event("llm", "Reusing a saved model response", kind="request_cache", purpose=purpose,
                       model=request["model"], cached=True, artifact=result_path,
                       request_path=directory / "request.json")
            return read_json(result_path)["value"]
        # A response may have arrived just before interruption. Recover it before
        # making another paid request; malformed earlier replies remain evidence.
        for saved in sorted(directory.glob("response_*.json"), reverse=True):
            try:
                payload = read_json(saved)
                value = parse_response(payload["text"], response_json)
            except (ValueError, KeyError, TypeError):
                continue
            write_json(result_path, {"value": value, "source_response": saved.name})
            self.event("llm", "Recovering a result from the raw response saved before interruption", kind="response_recovered", purpose=purpose,
                       model=request["model"], cached=True, response_path=saved, artifact=result_path,
                       usage=payload.get("usage"), request_path=directory / "request.json")
            return value
        write_json(directory / "request.json", request)
        kwargs = {"model": request["model"], "messages": request["messages"], "temperature": 0.5}
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        self.event("llm", "Starting model request", kind="request_start", purpose=purpose,
                   model=request["model"], request_path=directory / "request.json", mode="live_api")
        started = time.monotonic()
        response = self._call(lambda: self.client.chat.completions.create(**kwargs), directory,
                              purpose=purpose, model=request["model"])
        raw = response.choices[0].message.content if response.choices else None
        raw = raw or ""
        raw = redact(raw)
        response_path = directory / f"response_{time.time_ns()}.json"
        usage = response.usage.model_dump() if response.usage else None
        write_json(response_path, {
            "id": response.id, "model": response.model, "text": raw,
            "finish_reason": response.choices[0].finish_reason if response.choices else None,
            "usage": usage,
        })
        try:
            value = parse_response(raw, response_json)
        except (ValueError, TypeError) as exc:
            self.event("llm", "Invalid response format; raw content preserved", kind="response_invalid", purpose=purpose,
                       model=response.model, duration_seconds=time.monotonic() - started,
                       usage=usage, response_path=response_path, error_type=type(exc).__name__)
            raise ServiceError(f"Invalid JSON or empty response saved at {directory}; resume to retry") from exc
        write_json(result_path, {"value": value})
        self.event("llm", "Model response completed", kind="request_completed", purpose=purpose,
                   model=response.model, duration_seconds=time.monotonic() - started, usage=usage,
                   response_path=response_path, artifact=result_path,
                   finish_reason=response.choices[0].finish_reason if response.choices else None)
        return value

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        model = _model_name(self.config.embedding_model)
        for index, text in enumerate(texts, 1):
            request = {"model": model, "api_base": self.config.api_base, "text": text}
            directory = self.directory / "embeddings" / request_key(request)
            result_path = directory / "result.json"
            if result_path.is_file():
                vector = read_json(result_path)["vector"]
                self.event("llm", "Reusing a saved embedding", kind="embedding_cache", purpose="embedding",
                           model=model, cached=True, index=index, count=len(texts), artifact=result_path)
            else:
                write_json(directory / "request.json", request)
                self.event("llm", "Starting embedding request", kind="embedding_start", purpose="embedding",
                           model=model, index=index, count=len(texts), request_path=directory / "request.json")
                started = time.monotonic()
                response = self._call(lambda: self.client.embeddings.create(model=model, input=[text]), directory,
                                      purpose="embedding", model=model)
                vector = response.data[0].embedding
                if not vector or any(not math.isfinite(v) for v in vector):
                    self.event("llm", "Embedding response is empty or contains invalid values", kind="embedding_error",
                               purpose="embedding", model=model, duration_seconds=time.monotonic() - started,
                               request_path=directory / "request.json")
                    raise ServiceError("Embedding API returned an empty or non-finite vector")
                write_json(result_path, {"vector": vector, "model": response.model,
                                         "usage": response.usage.model_dump() if response.usage else None})
                self.event("llm", "Embedding request completed", kind="embedding_completed", purpose="embedding",
                           model=response.model, duration_seconds=time.monotonic() - started,
                           usage=response.usage.model_dump() if response.usage else None,
                           index=index, count=len(texts), dimensions=len(vector), artifact=result_path)
            vectors.append(vector)
        return vectors


class ReplayClient:
    """Consume a labelled fixture, never an API. Useful for learning and debugging.

    Fixture responses have `purpose` (the prefix before '/') and `response`.
    Synthetic embeddings are only for exercising offline memory plumbing.
    They are not a replacement for the live embedding model.
    """

    def __init__(self, fixture: Path, directory: Path, *, event=None):
        self.event = event or null_event
        self.fixture = read_json(fixture)
        self.path = Path(directory) / "replay_state.json"
        self.fixture_hash = request_key(self.fixture)
        if self.path.is_file():
            self.state = read_json(self.path)
            if self.state["fixture_hash"] != self.fixture_hash:
                raise ValueError("Replay fixture changed; use a new run directory")
        else:
            self.state = {"fixture_hash": self.fixture_hash, "next_index": 0, "cache": {}}
        self.event("llm", "Offline replay ready; no API calls will be made", kind="replay_ready", mode="offline_replay",
                   fixture=fixture, next_index=self.state["next_index"], artifact=self.path)

    def complete(self, system: str, user: str, *, purpose: str, response_json: bool = True):
        request = {"system": system, "user": user, "purpose": purpose, "response_json": response_json}
        key = request_key(request)
        if key in self.state["cache"]:
            self.event("llm", "Offline replay: reusing a previously consumed response", kind="request_cache", mode="offline_replay",
                       purpose=purpose, cached=True, fixture_index=self.state["cache"][key]["fixture_index"],
                       artifact=self.path)
            return self.state["cache"][key]["response"]
        index = self.state["next_index"]
        self.event("llm", "Offline replay: reading the scripted response", kind="request_start", mode="offline_replay",
                   purpose=purpose, fixture_index=index, artifact=self.path)
        responses = self.fixture["responses"]
        if index >= len(responses):
            self.event("llm", "Offline replay responses exhausted", kind="request_error", mode="offline_replay",
                       purpose=purpose, fixture_index=index, artifact=self.path)
            raise ServiceError(f"Offline fixture exhausted at {purpose}; no API fallback is available")
        item = responses[index]
        if item["purpose"] != purpose.split("/")[0]:
            self.event("llm", "Offline replay stage mismatch; cursor preserved", kind="request_error",
                       mode="offline_replay", purpose=purpose, expected_purpose=item["purpose"],
                       fixture_index=index, artifact=self.path)
            raise ServiceError(f"Offline fixture expected {item['purpose']}, received {purpose}")
        value = item["response"]
        if response_json and not isinstance(value, dict):
            self.event("llm", "Invalid offline replay response format", kind="response_invalid", mode="offline_replay",
                       purpose=purpose, fixture_index=index, artifact=self.path)
            raise ServiceError("Offline fixture response must be a JSON object")
        self.state["cache"][key] = {"request": request, "response": value, "fixture_index": index}
        self.state["next_index"] += 1
        write_json(self.path, self.state)
        self.event("llm", "Offline replay response saved", kind="request_completed", mode="offline_replay",
                   purpose=purpose, fixture_index=index, artifact=self.path)
        return value

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.event("llm", "Offline replay: generating test vectors without API calls", kind="embedding_replay",
                   mode="offline_replay", count=len(texts), dimensions=64)
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for token in re.findall(r"\w+", text.lower()):
                vector[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 64] += 1.0
            length = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / length for v in vector])
        return vectors
