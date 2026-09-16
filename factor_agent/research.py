"""Propose hypotheses, define factors and interpret backtest results.

Each research action makes a separate LLM request and passes ordinary data
objects. This module does not execute code, train models or change architectures.
Finance prompts are adapted from RD-Agent fin_factor; see the project README for provenance.
"""

import json
from pathlib import Path

from .errors import ExperimentFailed
from .models import BacktestResult, Experiment, FactorArtifact, FactorSpec, Feedback, Hypothesis


def prompt(name: str, scenario_text: str = "") -> str:
    """Read a small finance prompt; the only dynamic placeholder is the scenario."""
    text = (Path(__file__).parent / "prompts" / (name + ".txt")).read_text()
    return text.replace("{{ scenario }}", scenario_text)


def scenario(config) -> str:
    """Describe the actual data interface and the fixed evaluation model."""
    data_dir = Path(config.debug_data_dir)
    files = sorted(p.name for p in data_dir.iterdir() if p.is_file()) if data_dir.exists() else []
    readme = data_dir / "README.md"
    data_description = readme.read_text() if readme.exists() else "Data README is not available."
    return prompt("scenario") + (
        "\n\nThis run uses a fixed CPU LightGBM evaluation model. Only implement factors; "
        "do not change training, model architecture or backtest configuration.\n"
        "Language requirement: Write all natural-language responses in English, including "
        "factor names and descriptions, explanations, reviews, and feedback. Use English "
        "for generated Python comments and docstrings. Preserve the required JSON keys, "
        "mathematical formulas, data column names, and execution interfaces exactly.\n"
        "Input files in the execution directory: " + ", ".join(files) + ".\n"
        "daily_pv.h5 is a pandas HDF DataFrame with MultiIndex (datetime, instrument), "
        "containing daily $open, $high, $low, $close, $volume and $factor where available. "
        "Read data from the current execution directory. The same code runs first on the "
        "debug data and then on the full data; do not hard-code dates or instrument names. "
        "Use only information available at or before each factor timestamp.\n"
        "Input data documentation:\n" + data_description
    )


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def require_object(value, stage: str) -> dict:
    if not isinstance(value, dict):
        raise ExperimentFailed(f"{stage}: expected a JSON object, got {type(value).__name__}")
    return value


def require_text(value: dict, key: str, stage: str) -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ExperimentFailed(f"{stage}: missing or empty string field {key!r}")
    return text


def require_bool(value, stage: str) -> bool:
    """Accept the explicit boolean spellings used in the upstream prompts."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower().strip() in {"yes", "true", "1", "no", "false", "0"}:
        return value.lower().strip() in {"yes", "true", "1"}
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ExperimentFailed(f"{stage}: expected an explicit boolean/yes/no decision")


def history_context(history: list[Experiment]) -> str:
    """Research history is separate from coding memory, including rejected trials."""
    records = []
    for exp in history:
        records.append({
            "round": exp.round_id,
            "hypothesis": exp.hypothesis.__dict__ if exp.hypothesis else None,
            "factors": [spec.__dict__ for spec in exp.specs],
            "implemented": [f.spec.name for f in exp.factors if f.implemented],
            "backtest_result": exp.result.metrics if exp.result else None,
            "feedback": exp.feedback.__dict__ if exp.feedback else None,
            "error": exp.error,
        })
    return json_text(records) if records else "No previous hypothesis and feedback: this is the first round."


class Researcher:
    """The three research actions surrounding the code-and-backtest experiment."""

    def __init__(self, llm, config):
        self.llm = llm
        self.config = config

    def propose(self, history: list[Experiment]) -> Hypothesis:
        guidance = (
            "Try the easiest and fastest factors to experiment with from various perspectives first."
            if len(history) < 15 else
            "Now, you need to try factors that can achieve high IC (e.g., machine learning-based factors)."
        )
        system = (
            "Generate new hypotheses for factors in a data-driven research and development process.\n"
            + scenario(self.config) + "\nAnalyze successful and failed trials. Refine a direction or explore "
            "a new one. The latest feedback's suggested hypothesis is a reference, not an instruction.\n"
            + prompt("hypothesis_rules") + "\n" + prompt("hypothesis_schema")
        )
        user = "Experimental history:\n" + history_context(history)
        if history:
            user += "\nLatest trial (pay particular attention):\n" + history_context(history[-1:])
        user += "\nAdditional guidance:\n" + guidance
        reply = require_object(self.llm.complete(system, user, purpose=f"hypothesis/round_{len(history):03d}"), "propose")
        return Hypothesis(require_text(reply, "hypothesis", "propose"), require_text(reply, "reason", "propose"))

    def design(self, hypothesis: Hypothesis, history: list[Experiment], accepted: list[FactorArtifact]) -> list[FactorSpec]:
        system = (
            "Generate 1–5 concrete new factors to test the supplied hypothesis.\n" + scenario(self.config)
            + "\nSpecify all window sizes and other parameters. Avoid reimplementing factors already "
            "in the accepted library: these are automatically included in every new experiment.\n"
            + prompt("factor_schema")
        )
        user = (
            "Target hypothesis:\n" + json_text(hypothesis.__dict__)
            + "\nExperimental history:\n" + history_context(history)
            + "\nAccepted factors:\n" + json_text([factor.spec.__dict__ for factor in accepted])
        )
        reply = require_object(self.llm.complete(system, user, purpose=f"design/round_{len(history):03d}"), "design")
        if not 1 <= len(reply) <= 5:
            raise ExperimentFailed("design: expected 1–5 factor definitions")
        accepted_names = {factor.spec.name for factor in accepted}
        specs = []
        for name, value in reply.items():
            if not isinstance(name, str) or not name.strip():
                raise ExperimentFailed("design: factor name must be a nonempty string")
            value = require_object(value, f"design/{name}")
            variables = require_object(value.get("variables"), f"design/{name}/variables")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in variables.items()):
                raise ExperimentFailed(f"design/{name}: variables must map strings to strings")
            spec = FactorSpec(name, require_text(value, "description", "design"),
                              require_text(value, "formulation", "design"), variables)
            if name not in accepted_names:
                specs.append(spec)
        if not specs:
            raise ExperimentFailed("design: every proposed name already exists in the accepted factor library")
        return specs

    def review(self, hypothesis: Hypothesis, factors: list[FactorArtifact], result: BacktestResult,
               best_result: BacktestResult, history: list[Experiment]) -> Feedback:
        system = prompt("research_review", scenario(self.config))
        used = set(result.factor_names)
        details = [{
            **factor.spec.__dict__,
            "Factor Implementation": factor.implemented and factor.spec.name in used,
            "failure_or_drop_reason": result.dropped_factors.get(factor.spec.name, factor.error),
        } for factor in factors]
        metrics = ("IC", "1day.excess_return_with_cost.annualized_return",
                   "1day.excess_return_with_cost.max_drawdown")
        user = json_text({
            "Target hypothesis": hypothesis.__dict__,
            "Tasks and Factors": details,
            "Current combined result": {name: result.metrics[name] for name in metrics},
            "SOTA result": {name: best_result.metrics[name] for name in metrics},
            "Note": "Only factors with Factor Implementation=true were tested in this result; "
                    "others cannot support or refute their hypotheses.",
        })
        reply = require_object(self.llm.complete(system, user, purpose=f"feedback/round_{len(history):03d}"), "review")
        return Feedback(
            observations=require_text(reply, "Observations", "review"),
            hypothesis_evaluation=require_text(reply, "Feedback for Hypothesis", "review"),
            new_hypothesis=require_text(reply, "New Hypothesis", "review"),
            reason=require_text(reply, "Reasoning", "review"),
            decision=require_bool(reply.get("Replace Best Result"), "review"),
        )
