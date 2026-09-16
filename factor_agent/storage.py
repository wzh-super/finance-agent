"""Transparent artifact storage: JSON checkpoints are authoritative; Markdown is a reading view."""

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .models import RunState, state_from_dict
from .progress import ProgressLogger


def _encode(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def write_json(path: Path, value: Any):
    """Write completely, fsync, then replace: interruption cannot leave half a JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, default=_encode, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


class RunStore:
    """One run has one authoritative state.json and one directory per experiment."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.progress = ProgressLogger(self.root)
        self.round_id = None

    def load(self, config_fingerprint: str) -> RunState:
        if not self.state_path.exists():
            return RunState(config_fingerprint=config_fingerprint)
        try:
            state = state_from_dict(read_json(self.state_path))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid checkpoint {self.state_path}: {exc}") from exc
        if state.config_fingerprint != config_fingerprint:
            raise ValueError("Configuration fingerprint changed; use the original configuration or a new run directory")
        expected = list(range(len(state.history)))
        if [e.round_id for e in state.history] != expected or any(e.stage != "complete" for e in state.history):
            raise ValueError("Invalid checkpoint: completed experiment history is inconsistent")
        if state.current and (state.current.round_id != len(state.history) or
                              state.current.stage not in {"propose", "design", "code", "backtest", "review", "commit"}):
            raise ValueError("Invalid checkpoint: current experiment stage/index is inconsistent")
        return state

    def save(self, state: RunState):
        self.round_id = state.current.round_id if state.current else (state.history[-1].round_id if state.history else None)
        write_json(self.state_path, state)
        if state.current:
            write_json(self.round_dir(state.current.round_id) / "experiment.json", state.current)
        for experiment in state.history[-1:]:
            write_json(self.round_dir(experiment.round_id) / "experiment.json", experiment)
        self.write_report(state)
        try:
            self.write_trajectory()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.event("warning", "Trajectory export failed; the JSON checkpoint has been saved", error=str(error),
                       checkpoint=self.state_path)

    def round_dir(self, round_id: int) -> Path:
        path = self.root / "rounds" / f"{round_id:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def event(self, stage: str, message: str, **details):
        if self.round_id is not None:
            details.setdefault("round", self.round_id)
        self.progress.event(stage, message, **details)

    def write_trajectory(self):
        from .trajectory import write_trajectory
        if self.state_path.exists():
            return write_trajectory(self.root)

    def write_report(self, state: RunState):
        lines = ["# Factor research run", "", f"Completed rounds: {len(state.history)}", "",
                 "Accepted factors: " + (", ".join(f.spec.name for f in state.accepted_factors) or "none"), ""]
        for e in state.history:
            lines.extend([f"## Round {e.round_id}", "", e.hypothesis.hypothesis if e.hypothesis else "Proposal failed.", ""])
            if e.result:
                lines.extend(["| Metric | Value |", "| --- | ---: |"])
                lines.extend(f"| {k} | {v:.8g} |" for k, v in sorted(e.result.metrics.items()))
                lines.append("")
            if e.feedback:
                lines.extend([f"Accepted: {e.feedback.decision}", "", e.feedback.observations, "", e.feedback.reason, ""])
            if e.error:
                lines.extend([f"Failure: {e.error}", ""])
            lines.append(f"Artifacts: [rounds/{e.round_id:03d}](rounds/{e.round_id:03d}/)")
            lines.append("")
        (self.root / "report.md").write_text("\n".join(lines), encoding="utf-8")
