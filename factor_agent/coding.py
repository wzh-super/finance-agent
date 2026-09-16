"""Develop a batch of factors with the upstream CoSTEER evolution boundaries.

Each evolution round freezes retrieval, generates all unfinished tasks, evaluates
the batch, then publishes its memory. Tasks execute serially within each phase.
Per-request and per-batch checkpoints preserve these boundaries after a pause.
"""

import hashlib
from pathlib import Path
import time

from .errors import ExperimentFailed, ServiceError
from .models import CodingAttempt, ExecutionResult, FactorArtifact, FactorSpec, factor_from_dict
from .progress import null_event
from .research import json_text, prompt, require_bool, require_object, require_text, scenario
from .storage import read_json, write_json


def attempt_from_dict(value: dict) -> CodingAttempt:
    return CodingAttempt(**{**value, "execution": ExecutionResult(**value["execution"])})


class FactorCoder:
    """Evolve unfinished factors together; retain every attempt and review."""

    def __init__(self, llm, executor, memory, config, *, event=None):
        self.llm = llm
        self.executor = executor
        self.memory = memory
        self.config = config
        self.event = event or null_event

    def _request(self, name: str, directory: Path, system: str, user: str, *, response_json=True):
        """A completed request survives interruption even before the attempt finishes."""
        path = directory / f"{name}.json"
        provenance = directory / f"{name}_request.json"
        request_hash = hashlib.sha256(json_text([system, user, response_json]).encode()).hexdigest()
        if provenance.exists() and read_json(provenance).get("sha256") != request_hash:
            raise ValueError(f"Request inputs changed at {directory}/{name}; use a new run directory")
        if path.exists():
            if not provenance.exists():
                raise ValueError(f"Missing request provenance at {path}; use a new run directory")
            self.event(name, "Restoring a saved coding-stage response", kind="stage_cache", cached=True,
                       purpose=f"{name}/{directory}", artifact=path, provenance=provenance)
            return read_json(path)
        write_json(provenance, {"sha256": request_hash})
        response = self.llm.complete(system, user, purpose=f"{name}/{directory}", response_json=response_json)
        write_json(path, response)
        return response

    @staticmethod
    def _verify_code(spec: FactorSpec, code_path: Path, provenance: Path) -> None:
        if not code_path.is_file() or not provenance.is_file():
            raise ValueError(f"Missing factor code/provenance at {code_path}; use a new run directory")
        saved = read_json(provenance)
        actual = hashlib.sha256(code_path.read_bytes()).hexdigest()
        if saved.get("spec_identity") != spec.identity or saved.get("code_sha256") != actual:
            raise ValueError(f"Factor code or specification changed at {code_path}; use a new run directory")

    def _verify_artifact(self, spec: FactorSpec, artifact: FactorArtifact, directory: Path) -> None:
        if artifact.spec.identity != spec.identity:
            raise ValueError(f"Factor specification changed at {directory}; use a new run directory")
        for attempt in artifact.attempts:
            code_path = Path(attempt.code_path)
            self._verify_code(spec, code_path, code_path.parent / "provenance.json")
        if artifact.code_path and artifact.code_path not in {attempt.code_path for attempt in artifact.attempts}:
            self._verify_code(spec, Path(artifact.code_path), directory / "reuse.json")

    def develop(self, specs: list[FactorSpec], directory: Path) -> list[FactorArtifact]:
        """Mirror RAGEvoAgent: query -> evolve batch -> evaluate -> update memory.

        ``max_coding_attempts`` retains its public configuration name, but bounds
        the number of batch evolution rounds, not a separate inner task loop.
        """
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        identities = [spec.identity for spec in specs]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate factor specifications in a coding batch")
        manifest = {"version": 1, "schedule": "batch_evolution", "identities": identities,
                    "max_evolution_rounds": self.config.max_coding_attempts}
        manifest_path = directory / "schedule.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise ValueError(f"Coding schedule changed at {directory}; use a new run directory")
        else:
            if any((directory / identity).exists() for identity in identities):
                raise ValueError("Legacy coding checkpoint has no batch schedule; use its original source or a new run directory")
            write_json(manifest_path, manifest)

        artifacts = [FactorArtifact(spec) for spec in specs]
        for index, spec in enumerate(specs, 1):
            factor_dir = directory / spec.identity
            factor_dir.mkdir(parents=True, exist_ok=True)
            spec_path = factor_dir / "spec.json"
            if spec_path.exists():
                if FactorSpec(**read_json(spec_path)).identity != spec.identity:
                    raise ValueError(f"Factor specification changed at {factor_dir}; use a new run directory")
            else:
                write_json(spec_path, spec)
            self.event("code", f"Factor {index}/{len(specs)} in the coding batch: {spec.name}", kind="factor_start",
                       factor=spec.name, identity=spec.identity, index=index, count=len(specs),
                       formulation=spec.formulation, artifact=factor_dir)

        if (directory / "completed.json").exists():
            for index, spec in enumerate(specs):
                factor_dir = directory / spec.identity
                artifact = factor_from_dict(read_json(factor_dir / "factor.json"))
                self._verify_artifact(spec, artifact, factor_dir)
                artifacts[index] = artifact
                self.event("code", "Restoring a completed factor development result", kind="factor_cache", factor=spec.name,
                           identity=spec.identity, cached=True, implemented=artifact.implemented,
                           attempts=len(artifact.attempts), artifact=factor_dir / "factor.json")
            return artifacts

        started = time.monotonic()
        by_identity = {artifact.spec.identity: artifact for artifact in artifacts}
        for number in range(1, self.config.max_coding_attempts + 1):
            batch_dir = directory / f"evolution_{number:03d}"
            plan_path = batch_dir / "plan.json"
            if plan_path.exists():
                plan = read_json(plan_path)
            else:
                plan = self._plan_round(artifacts, number)
                write_json(plan_path, plan)
            planned = plan["scheduled"] + plan["blocked"] + list(plan["reused"])
            expected = {item.spec.identity for item in artifacts if not item.implemented and not item.error}
            if plan["number"] != number or set(planned) != expected or len(planned) != len(set(planned)):
                raise ValueError(f"Invalid coding evolution plan at {plan_path}")
            self.event("code", f"Evolution round {number}/{self.config.max_coding_attempts}", kind="evolution_start",
                       evolution_round=number, max_evolution_rounds=self.config.max_coding_attempts,
                       scheduled_factors=[by_identity[key].spec.name for key in plan["scheduled"]],
                       already_approved=sum(item.implemented for item in artifacts), artifact=plan_path)

            for identity, success in plan["reused"].items():
                self._reuse(by_identity[identity], success, directory / identity)
            for identity in plan["blocked"]:
                artifact = by_identity[identity]
                artifact.error = "This factor has reached its cumulative failed-attempt limit"
                self.event("code", "Cumulative failure limit reached for this factor", kind="attempts_exhausted",
                           factor=artifact.spec.name, identity=identity, max_failures=self.config.max_task_failures,
                           evolution_round=number, artifact=directory / identity)

            scheduled = [by_identity[key] for key in plan["scheduled"]]
            # Freeze every task's references before any new code or feedback.
            # The persisted contexts also prevent new information leaking in on resume.
            for artifact in scheduled:
                self._context(artifact.spec, directory / artifact.spec.identity / f"attempt_{number:03d}", number)
            self.event("code", "Generating code for all unfinished factors", kind="evolution_generation",
                       evolution_round=number, scheduled_count=len(scheduled), artifact=plan_path)
            for artifact in scheduled:
                self._generate(artifact.spec, directory / artifact.spec.identity / f"attempt_{number:03d}", number)

            self.event("code", "Evaluating the generated factor batch", kind="evolution_evaluation",
                       evolution_round=number, scheduled_count=len(scheduled), artifact=plan_path)
            attempts = []
            for artifact in scheduled:
                attempt_dir = directory / artifact.spec.identity / f"attempt_{number:03d}"
                attempt = self._attempt(artifact.spec, attempt_dir, number)
                artifact.attempts.append(attempt)
                artifact.implemented = attempt.accepted
                attempts.append((artifact.spec, attempt))

            # No task in this batch may query these records before this barrier.
            # One atomic write plus idempotent replay closes the commit crash window.
            self.memory.record_batch(attempts)
            write_json(batch_dir / "completed.json", {"number": number, "scheduled": plan["scheduled"],
                       "approved": [item.spec.identity for item in artifacts if item.implemented]})
            self.event("code", "Evolution round evaluated; batch feedback committed to coding memory",
                       kind="evolution_completed", evolution_round=number,
                       approved_count=sum(item.implemented for item in artifacts), total_count=len(artifacts),
                       artifact=batch_dir / "completed.json")

            pending = [item for item in artifacts if not item.implemented and not item.error]
            if not pending:
                break
            for artifact in pending:
                if number < self.config.max_coding_attempts and self.memory.failure_count(artifact.spec) < self.config.max_task_failures:
                    self.event("code", "Implementation not approved; scheduling repair in the next evolution round",
                               kind="repair_planned", factor=artifact.spec.name, identity=artifact.spec.identity,
                               attempt=number, next_attempt=number + 1, evolution_round=number,
                               feedback=artifact.attempts[-1].final_feedback, artifact=directory / artifact.spec.identity)

        for index, artifact in enumerate(artifacts, 1):
            self._finish(artifact, directory / artifact.spec.identity / "factor.json")
            if not artifact.implemented:
                self.event("code", "Factor development finished without an approved implementation", kind="factor_failed",
                           factor=artifact.spec.name, identity=artifact.spec.identity, attempts=len(artifact.attempts),
                           max_evolution_rounds=self.config.max_coding_attempts, artifact=directory / artifact.spec.identity)
            self.event("code", f"Factor implementation {'approved' if artifact.implemented else 'not approved'}: {artifact.spec.name}",
                       kind="factor_completed", factor=artifact.spec.name, identity=artifact.spec.identity,
                       index=index, count=len(specs), implemented=artifact.implemented,
                       attempts=len(artifact.attempts), error=artifact.error,
                       code_path=artifact.code_path, artifact=directory / artifact.spec.identity / "factor.json")
        write_json(directory / "completed.json", {"identities": identities})
        self.event("code", "Factor batch development completed", kind="coding_completed",
                   duration_seconds=time.monotonic() - started,
                   approved_count=sum(item.implemented for item in artifacts), total_count=len(artifacts),
                   artifact=directory / "completed.json")
        return artifacts

    def _plan_round(self, artifacts: list[FactorArtifact], number: int) -> dict:
        plan = {"number": number, "scheduled": [], "blocked": [], "reused": {}}
        for artifact in artifacts:
            if artifact.implemented or artifact.error:
                continue
            spec = artifact.spec
            success = self.memory.success(spec)
            if success is not None:
                plan["reused"][spec.identity] = {"key": success["key"], "code": success["code"]}
            elif self.memory.failure_count(spec) >= self.config.max_task_failures:
                plan["blocked"].append(spec.identity)
            else:
                plan["scheduled"].append(spec.identity)
        return plan

    def _reuse(self, artifact: FactorArtifact, success: dict, directory: Path) -> None:
        spec = artifact.spec
        code_path, provenance = directory / "reused_factor.py", directory / "reuse.json"
        expected = {"source_attempt": success["key"], "spec_identity": spec.identity,
                    "code_sha256": hashlib.sha256(success["code"].encode()).hexdigest(),
                    "reason": "Exact factor specification already has an accepted implementation"}
        if code_path.exists() and code_path.read_text() != success["code"]:
            raise ValueError(f"Reused factor code changed at {code_path}; use a new run directory")
        if provenance.exists() and read_json(provenance) != expected:
            raise ValueError(f"Reused factor provenance changed at {provenance}; use a new run directory")
        if not code_path.exists():
            code_path.write_text(success["code"])
        if not provenance.exists():
            write_json(provenance, expected)
        artifact.code_path, artifact.implemented = str(code_path), True
        self.event("code", "Reusing a successful implementation for an exact task match", kind="implementation_reused",
                   factor=spec.name, identity=spec.identity, cached=True, source_attempt=success["key"],
                   code_path=code_path, code=success["code"], artifact=provenance)

    @staticmethod
    def _finish(artifact: FactorArtifact, path: Path) -> FactorArtifact:
        successful = next((attempt for attempt in artifact.attempts if attempt.accepted), None)
        if successful:
            artifact.code_path = successful.code_path
            artifact.values_path = successful.execution.data_path
            artifact.implemented = True
        elif artifact.implemented:
            pass  # An exact task match reuses a snapshot without a new attempt.
        elif artifact.attempts:
            artifact.code_path = artifact.attempts[-1].code_path
            artifact.error = artifact.error or artifact.attempts[-1].final_feedback or "Coding attempts exhausted"
        else:
            artifact.error = artifact.error or "No coding attempts were available"
        write_json(path, artifact)
        return artifact

    def _context(self, spec: FactorSpec, directory: Path, number: int) -> dict:
        path = directory / "retrieved_memory.json"
        cached = path.exists()
        if cached:
            context = read_json(path)
        else:
            if (directory / "code_request.json").exists() or (directory / "factor.py").exists():
                raise ValueError(f"Missing frozen coding context at {path}; use a new run directory")
            context = self.memory.context(spec)
            write_json(path, context)
        self.event("memory", "Coding references frozen for this evolution round", kind="attempt_context", factor=spec.name,
                   identity=spec.identity, attempt=number, evolution_round=number, artifact=path,
                   previous_failures=len(context.get("previous_failed_attempts", [])),
                   similar_successes=len(context.get("similar_successful_implementations", [])),
                   similar_error_repairs=len(context.get("similar_error_repairs", [])),
                   has_runnable_attempt=context.get("latest_runnable_attempt") is not None, cached=cached)
        return context

    def _generate(self, spec: FactorSpec, directory: Path, number: int) -> None:
        """Generate one task's code without running it or publishing any feedback."""
        directory.mkdir(parents=True, exist_ok=True)
        code_path = directory / "factor.py"
        resumed = code_path.exists() or (directory / "code_request.json").exists()
        self.event("code", f"{'Resuming' if resumed else 'Starting'} factor attempt in evolution round {number}/{self.config.max_coding_attempts}",
                   kind="attempt_start", factor=spec.name, identity=spec.identity, attempt=number,
                   evolution_round=number, max_attempts=self.config.max_coding_attempts, resumed=resumed, artifact=directory)
        if (directory / "attempt.json").exists() or (directory / "generation_error.json").exists():
            self._verify_code(spec, code_path, directory / "provenance.json")
            return
        try:
            if not code_path.exists():
                context = read_json(directory / "retrieved_memory.json")
                code_response_cached = (directory / "code.json").exists()
                response = self._request(
                    "code", directory, prompt("coding", scenario(self.config)),
                    "Target factor:\n" + spec.describe() + "\nCoding memory:\n" + json_text(context),
                )
                response = require_object(response, "code")
                code_path.write_text(require_text(response, "code", "code"))
                self.event("code", "Restoring Python code from a saved response" if code_response_cached else "Generated Python code saved",
                           kind="code_restored" if code_response_cached else "code_generated", factor=spec.name,
                           identity=spec.identity, attempt=number, code_path=code_path,
                           code=code_path.read_text(), cached=code_response_cached, artifact=directory / "code.json")
            else:
                self.event("code", "Restoring Python code already saved for this attempt", kind="code_resumed", factor=spec.name,
                           identity=spec.identity, attempt=number, cached=True,
                           code_path=code_path, code=code_path.read_text())

            provenance = directory / "provenance.json"
            if not provenance.exists():
                response = read_json(directory / "code.json")
                expected_code = require_text(require_object(response, "code"), "code", "code")
                if code_path.read_text() != expected_code:
                    raise ValueError(f"Factor code changed at {code_path}; use a new run directory")
                write_json(provenance, {"spec_identity": spec.identity,
                                       "code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest()})
            self._verify_code(spec, code_path, provenance)
        except ExperimentFailed as error:
            # A malformed response is failed work, preserved until batch evaluation.
            if not code_path.exists():
                code_path.write_text("")
            write_json(directory / "provenance.json", {"spec_identity": spec.identity,
                       "code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest()})
            write_json(directory / "generation_error.json", {"error": str(error)})
        except ServiceError:
            self._paused(spec, directory, number)
            raise

    def _attempt(self, spec: FactorSpec, directory: Path, number: int) -> CodingAttempt:
        """Evaluate generated code; memory changes only after the entire batch."""
        code_path = directory / "factor.py"
        self._verify_code(spec, code_path, directory / "provenance.json")
        attempt_file = directory / "attempt.json"
        if attempt_file.exists():
            attempt = attempt_from_dict(read_json(attempt_file))
            if attempt.number != number or Path(attempt.code_path) != code_path:
                raise ValueError(f"Coding attempt changed at {attempt_file}; use a new run directory")
            self.event("code", "Restoring a completed coding attempt", kind="attempt_resumed", factor=spec.name,
                       identity=spec.identity, attempt=number, accepted=attempt.accepted,
                       execution_success=attempt.execution.success, feedback=attempt.final_feedback,
                       cached=True, artifact=attempt_file)
            return attempt
        attempt = CodingAttempt(number, str(code_path), ExecutionResult(False))
        started = time.monotonic()
        try:
            if (directory / "generation_error.json").exists():
                raise ExperimentFailed(read_json(directory / "generation_error.json")["error"])
            # The executor owns execution.json. Keep its result separately for an
            # interrupted evaluator, without overwriting the executor's cache manifest.
            execution_path = directory / "coding_execution.json"
            execution_cached = execution_path.exists()
            if execution_path.exists():
                attempt.execution = ExecutionResult(**read_json(execution_path))
            else:
                attempt.execution = self.executor.run(code_path, spec, directory / "debug")
                write_json(execution_path, attempt.execution)
            expected_hash = attempt.execution.stats.get("code_sha256")
            if expected_hash and expected_hash != hashlib.sha256(code_path.read_bytes()).hexdigest():
                raise ValueError(f"Execution code hash mismatch at {code_path}; use a new run directory")
            self.event("code", "Restoring the Debug execution result" if execution_cached else "Debug execution result saved",
                       kind="debug_result", factor=spec.name, identity=spec.identity, attempt=number,
                       cached=execution_cached, success=attempt.execution.success,
                       error=attempt.execution.error, stats=attempt.execution.stats,
                       returncode=attempt.execution.returncode, data_path=attempt.execution.data_path,
                       execution_log=directory / "debug" / "execution.log", artifact=execution_path)
            self._evaluate(spec, attempt, directory)
        except ExperimentFailed as error:
            # Malformed model content is failed work, while ServiceError deliberately
            # escapes this handler so an unavailable API never rejects a research idea.
            attempt.final_feedback = str(error)
            self.event("code", "Invalid coding or review response; recording a failed attempt", kind="attempt_error",
                       factor=spec.name, identity=spec.identity, attempt=number, error=str(error), artifact=directory)
        except ServiceError:
            self._paused(spec, directory, number)
            raise
        write_json(directory / "attempt.json", attempt)
        self.event("code", "Coding attempt approved" if attempt.accepted else "Coding attempt not approved", kind="attempt_completed",
                   factor=spec.name, identity=spec.identity, attempt=number, accepted=attempt.accepted,
                   duration_seconds=time.monotonic() - started, feedback=attempt.final_feedback,
                   artifact=directory / "attempt.json")
        return attempt

    def _paused(self, spec: FactorSpec, directory: Path, number: int) -> None:
        self.event("code", "Model service unavailable; coding paused with completed stages preserved", kind="attempt_paused",
                   factor=spec.name, identity=spec.identity, attempt=number, evolution_round=number, artifact=directory)

    def _evaluate(self, spec: FactorSpec, attempt: CodingAttempt, directory: Path) -> None:
        """Keep output review, code semantics and final implementation judgement visible."""
        execution = attempt.execution
        value_summary = json_text({"success": execution.success, "error": execution.error, "stats": execution.stats})
        format_ok = False
        if execution.success:
            response = require_object(self._request(
                "format_review", directory, prompt("format_review", scenario(self.config)),
                "Target factor:\n" + spec.describe() + "\nOutput dataframe checks:\n" + value_summary,
            ), "format_review")
            format_ok = require_bool(response.get("output_format_decision"), "format_review")
            attempt.value_feedback = require_text(response, "output_format_feedback", "format_review")
        else:
            attempt.value_feedback = "Execution or deterministic output validation failed: " + execution.error
        self.event("format_review", "Factor value and output format review", kind="format_review", factor=spec.name,
                   identity=spec.identity, attempt=attempt.number, accepted=format_ok,
                   skipped=not execution.success, feedback=attempt.value_feedback,
                   artifact=directory / "format_review.json" if execution.success else directory / "coding_execution.json")

        critic = self._request(
            "code_review", directory, prompt("code_review", scenario(self.config)),
            "Factor information:\n" + spec.describe()
            + "\nPython code:\n" + Path(attempt.code_path).read_text()
            + "\nExecution feedback:\n" + execution.stdout[-8000:] + "\n" + execution.error
            + "\nFactor value feedback:\n" + attempt.value_feedback + "\n" + value_summary,
            response_json=False,
        )
        if not isinstance(critic, str) or not critic.strip():
            raise ExperimentFailed("code_review: expected a nonempty textual critique")
        attempt.code_feedback = critic
        self.event("code_review", "Factor code semantic review", kind="code_review", factor=spec.name,
                   identity=spec.identity, attempt=attempt.number, feedback=critic,
                   artifact=directory / "code_review.json")

        # The LLM may explain execution errors, but cannot override a failed run or
        # malformed output. A profitable-looking result never bypasses this gate.
        response = require_object(self._request(
            "final_review", directory, prompt("final_review", scenario(self.config)),
            "Factor information:\n" + spec.describe() + "\nExecution feedback:\n"
            + execution.stdout[-8000:] + "\n" + execution.error
            + "\nValue feedback:\n" + attempt.value_feedback
            + "\nCode feedback:\n" + attempt.code_feedback,
        ), "final_review")
        decision = require_bool(response.get("final_decision"), "final_review")
        attempt.final_feedback = require_text(response, "final_feedback", "final_review")
        attempt.accepted = execution.success and format_ok and decision
        if decision and not attempt.accepted:
            attempt.final_feedback += "\nRejected by deterministic execution/output-format gate."
        self.event("final_review", "Final factor implementation review", kind="final_review", factor=spec.name,
                   identity=spec.identity, attempt=attempt.number, llm_decision=decision,
                   execution_success=execution.success, format_ok=format_ok, accepted=attempt.accepted,
                   feedback=attempt.final_feedback, artifact=directory / "final_review.json")
