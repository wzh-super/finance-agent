"""Explicit stage inputs and outputs as ordinary JSON-serializable dataclasses.

Implementation success and research acceptance are distinct decisions, recorded
in FactorArtifact and Feedback. String paths make artifacts easy to locate.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


@dataclass
class Hypothesis:
    hypothesis: str
    reason: str


@dataclass
class FactorSpec:
    name: str
    description: str
    formulation: str
    variables: dict[str, str] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        content = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:20]

    def describe(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    data_path: str | None = None
    error: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    returncode: int | None = None


@dataclass
class CodingAttempt:
    number: int
    code_path: str
    execution: ExecutionResult
    value_feedback: str = ""
    code_feedback: str = ""
    final_feedback: str = ""
    accepted: bool = False


@dataclass
class FactorArtifact:
    spec: FactorSpec
    code_path: str = ""
    values_path: str | None = None
    attempts: list[CodingAttempt] = field(default_factory=list)
    implemented: bool = False
    error: str = ""


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    workspace: str
    factor_names: list[str] = field(default_factory=list)
    added_factors: list[FactorArtifact] = field(default_factory=list)
    dropped_factors: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    observations: str
    hypothesis_evaluation: str
    new_hypothesis: str
    reason: str
    decision: bool


@dataclass
class Experiment:
    round_id: int
    stage: str = "propose"
    hypothesis: Hypothesis | None = None
    specs: list[FactorSpec] = field(default_factory=list)
    factors: list[FactorArtifact] = field(default_factory=list)
    result: BacktestResult | None = None
    feedback: Feedback | None = None
    error: str = ""


@dataclass
class RunState:
    config_fingerprint: str
    version: int = 1
    history: list[Experiment] = field(default_factory=list)
    current: Experiment | None = None
    accepted_factors: list[FactorArtifact] = field(default_factory=list)
    best_result: BacktestResult | None = None


def factor_from_dict(value: dict) -> FactorArtifact:
    attempts = [CodingAttempt(**{**a, "execution": ExecutionResult(**a["execution"])})
                for a in value.get("attempts", [])]
    return FactorArtifact(**{**value, "spec": FactorSpec(**value["spec"]), "attempts": attempts})


def result_from_dict(value: dict) -> BacktestResult:
    return BacktestResult(**{**value, "added_factors": [factor_from_dict(f) for f in value.get("added_factors", [])]})


def experiment_from_dict(value: dict) -> Experiment:
    return Experiment(**{
        **value,
        "hypothesis": Hypothesis(**value["hypothesis"]) if value.get("hypothesis") else None,
        "specs": [FactorSpec(**s) for s in value.get("specs", [])],
        "factors": [factor_from_dict(f) for f in value.get("factors", [])],
        "result": result_from_dict(value["result"]) if value.get("result") else None,
        "feedback": Feedback(**value["feedback"]) if value.get("feedback") else None,
    })


def state_from_dict(value: dict) -> RunState:
    if value.get("version") != 1:
        raise ValueError("Unsupported state version; expected version 1")
    return RunState(**{
        **value,
        "history": [experiment_from_dict(e) for e in value["history"]],
        "current": experiment_from_dict(value["current"]) if value.get("current") else None,
        "accepted_factors": [factor_from_dict(f) for f in value["accepted_factors"]],
        "best_result": result_from_dict(value["best_result"]) if value.get("best_result") else None,
    })
