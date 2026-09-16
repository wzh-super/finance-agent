"""One explicit event sink for terminal progress and the durable research timeline.

Components receive ``store.event`` as an optional callback. This module has no
agent, model or storage dependencies, and never changes research decisions.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading


def null_event(stage: str, message: str, **details):
    """Keep programmatic components quiet unless a caller supplies an event sink."""


def redact(text: str) -> str:
    for name, value in os.environ.items():
        if len(value) >= 8 and any(marker in name.upper() for marker in
                                   ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            text = text.replace(value, "[REDACTED_SECRET]")
    return text


def safe_value(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): ("[REDACTED_SECRET]" if any(marker in str(k).upper() for marker in
                    ("API_KEY", "PASSWORD", "SECRET", "AUTHORIZATION")) else safe_value(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value]
    if isinstance(value, (str, Path)):
        return redact(str(value))
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return redact(str(value))


class ProgressLogger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def event(self, stage: str, message: str, **details):
        now = datetime.now(timezone.utc)
        item = safe_value({"time": now.isoformat(), "stage": stage,
                           "message": message, "details": details})
        message_text = item["message"]
        if len(message_text) > 6000:
            message_text = message_text[:6000] + "\n... Full content is available in events.jsonl and the original artifact."
        title = f"[{now.astimezone():%H:%M:%S}] [{item['stage']}] {message_text}"
        lines = [title]
        for key, value in item["details"].items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
            text = str(value)
            # Full content remains in events.jsonl and the referenced artifact.
            # Console excerpts stop large subprocess/prompt payloads hiding progress.
            if len(text) > 6000:
                text = text[:6000] + "\n... Truncated; see events.jsonl or the original artifact for full content."
            lines.extend(f"  {key}: {line}" if number == 0 else f"    {line}"
                         for number, line in enumerate(text.splitlines() or [""]))
        rendered = "\n".join(lines) + "\n"
        rendered = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", rendered)
        with self._lock:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            with (self.root / "progress.log").open("a", encoding="utf-8") as handle:
                handle.write(rendered)
            print(rendered, end="", flush=True)


def metric_comparison(current: dict, previous: dict | None = None) -> str:
    """Use the same labels in live progress and exported trajectories."""
    labels = {"IC": "IC", "Rank IC": "Rank IC",
              "1day.excess_return_with_cost.annualized_return": "Annualized excess return after costs",
              "1day.excess_return_with_cost.information_ratio": "Excess-return information ratio after costs",
              "1day.excess_return_with_cost.max_drawdown": "Excess-return maximum drawdown after costs"}
    rows = ["Metric | Reference | Current | Change", "--- | ---: | ---: | ---:"]
    for key, label in labels.items():
        if key not in current:
            continue
        value, old = current[key], (previous or {}).get(key)
        percent = key.endswith(("annualized_return", "max_drawdown"))
        fmt = lambda number: f"{number:.4%}" if percent else f"{number:.6f}"
        delta = "—" if old is None else (f"{(value-old)*100:+.4f} pp" if percent else f"{value-old:+.6f}")
        rows.append(f"{label} | {fmt(old) if old is not None else '—'} | {fmt(value)} | {delta}")
    return "\n".join(rows)
