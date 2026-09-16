"""Read saved artifacts into a complete research narrative, without running agents.

The JSON/code/log files remain the evidence. Markdown is a regenerable reading
view, including unfinished coding attempts that have not reached state.json.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from .progress import metric_comparison, redact, safe_value


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _block(text, language=""):
    text = str(text)
    longest = max((len(match) for match in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def render_trajectory(root: Path, *, relative_to: Path | None = None) -> str:
    """Only read JSON, code and small review files; never load data or call an API."""
    root = Path(root).resolve()
    state = _read(root / "state.json")
    if state is None:
        raise FileNotFoundError(f"No saved state: {root / 'state.json'}")
    base = Path(relative_to).resolve() if relative_to is not None else root

    def link(path, label=None):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        name = label or os.path.relpath(path, root)
        target = os.path.relpath(path, base)
        return f"[{name}](<{target}>)" + (" (file missing)" if not path.exists() else "")

    mode = (_read(root / "run_mode.json") or {}).get("mode", "unknown")
    current = state.get("current")
    accepted = [factor["spec"]["name"] for factor in state.get("accepted_factors", [])]
    lines = ["# Agent Research Trajectory", "", f"Run directory: `{root}`", f"Mode: `{mode}`",
             f"Completed rounds: {len(state.get('history', []))}",
             f"Current stage: {current['stage'] if current else 'no unfinished stage'}", "",
             "Generated from saved artifacts without calling a model or executing code. This document records requests, responses, code, reviews and experiment results.", "",
             "Accepted factor library: " + (", ".join(accepted) or "empty"), "",
             "## Timeline and Source", "", link("events.jsonl"), link("config.json", "Run configuration (credentials excluded)"), ""]
    if (root / "progress.log").exists():
        lines.extend([link("progress.log", "Timestamped terminal progress"), ""])
    if (root / "source_snapshot/manifest.json").exists():
        lines.extend([link("source_snapshot/manifest.json", "Source snapshot and file hashes"), ""])
    else:
        lines.extend(["No source snapshot was saved for this run. Generated factor code remains in the attempt directories.", ""])
    if mode == "offline_replay":
        lines.extend(["This replay uses scripted responses, including reviews and research decisions. Actual execution and backtest results are preserved in the corresponding artifacts.", ""])

    baseline = _read(root / "baseline/worker_result.json")
    comparator = baseline.get("metrics", {}) if baseline else {}
    if baseline:
        lines.extend(["## Fixed-Model Baseline", "", metric_comparison(comparator), "",
                      link("baseline/worker_result.json"), link("baseline/backtest.log"), ""])
    else:
        lines.extend(["## Fixed-Model Baseline", "", "No completed baseline worker record is available.", ""])

    experiments = [*state.get("history", []), *([current] if current else [])]
    for experiment in experiments:
        number = experiment["round_id"]
        directory = root / "rounds" / f"{number:03d}"
        lines.extend([f"## Round {number} — {experiment['stage']}", "", link(directory / "experiment.json"), ""])
        hypothesis = experiment.get("hypothesis")
        if hypothesis:
            lines.extend(["### Hypothesis and Rationale", "", hypothesis["hypothesis"], "", hypothesis["reason"], ""])
        else:
            lines.extend(["No hypothesis has been saved yet.", ""])
        schedule = _read(directory / "coding/schedule.json")
        if schedule:
            names = {identity: spec["name"] for identity, spec in zip(schedule["identities"], experiment.get("specs", []))}
            lines.extend(["### Coding Evolution Rounds", "",
                          "Each round freezes references, generates the unfinished factor batch, evaluates it, then commits its feedback to memory.",
                          f"Maximum evolution rounds: {schedule['max_evolution_rounds']}", "",
                          "| Evolution round | Scheduled factors | Batch completed | Plan |",
                          "| --- | --- | --- | --- |"])
            for plan_path in sorted((directory / "coding").glob("evolution_*/plan.json")):
                plan = _read(plan_path)
                committed = (plan_path.parent / "completed.json").exists()
                factors = ", ".join(names.get(identity, identity) for identity in plan["scheduled"]) or "none"
                lines.append(f"| {plan['number']} | {factors} | {json.dumps(committed)} | {link(plan_path)} |")
            lines.append("")
        artifacts = {factor["spec"]["name"]: factor for factor in experiment.get("factors", [])}
        folders = {}
        for folder in sorted((directory / "coding").glob("*")):
            spec = _read(folder / "spec.json")
            if spec:
                folders[spec["name"]] = folder
                saved = _read(folder / "factor.json")
                if saved:
                    artifacts[spec["name"]] = saved
        for index, spec in enumerate(experiment.get("specs", []), 1):
            lines.extend([f"### Factor {index}: {spec['name']}", "", spec.get("description", ""), "",
                          _block(spec.get("formulation", "")), "",
                          _block(json.dumps(spec.get("variables", {}), ensure_ascii=False, indent=2), "json"), ""])
            factor = artifacts.get(spec["name"])
            implemented = json.dumps(factor["implemented"]) if factor else "pending"
            lines.extend([f"Implementation review — implemented: {implemented} (separate from research acceptance)", ""])
            if factor and factor.get("error"):
                lines.extend(["Implementation error:", _block(factor["error"]), ""])
            folder = folders.get(spec["name"])
            attempts = sorted(folder.glob("attempt_*")) if folder else []
            if folder and (folder / "reuse.json").exists():
                lines.extend(["Reused implementation: " + link(folder / "reuse.json"),
                              link(folder / "reused_factor.py"), ""])
            if not attempts and not factor:
                lines.extend(["No coding attempt has been saved yet.", ""])
            for attempt_dir in attempts:
                attempt = _read(attempt_dir / "attempt.json")
                lines.extend([f"#### {attempt_dir.name}" + (" (incomplete)" if attempt is None else ""), ""])
                code = attempt_dir / "factor.py"
                if code.exists():
                    lines.extend([link(code), f"SHA-256: `{hashlib.sha256(code.read_bytes()).hexdigest()}`", "",
                                  _block(code.read_text(encoding="utf-8"), "python"), ""])
                else:
                    lines.extend(["No generated code file is available yet.", ""])
                execution = (_read(attempt_dir / "coding_execution.json") or
                             (attempt or {}).get("execution") or
                             (_read(attempt_dir / "debug/execution.json") or {}).get("result"))
                if execution:
                    stats = execution.get("stats", {})
                    summary = {key: execution.get(key) for key in ("success", "returncode", "error")}
                    summary["stats"] = {key: stats[key] for key in ("rows", "instruments", "start", "end", "nan_count") if key in stats}
                    lines.extend(["Debug execution:", _block(json.dumps(summary, ensure_ascii=False, indent=2), "json"),
                                  link(attempt_dir / "debug/execution.log"), ""])
                    if execution.get("stdout"):
                        lines.extend(["Execution output:", _block(execution["stdout"]), ""])
                else:
                    lines.extend(["No Debug execution result is available yet.", ""])
                for filename, label, field in (("format_review", "Output Format Review", "value_feedback"),
                                               ("code_review", "Code Semantic Review", "code_feedback"),
                                               ("final_review", "Final Implementation Review", "final_feedback")):
                    review = _read(attempt_dir / f"{filename}.json")
                    if review is None:
                        review = (attempt or {}).get(field)
                    if review:
                        text = review if isinstance(review, str) else json.dumps(review, ensure_ascii=False, indent=2)
                        lines.extend([label + ":", _block(text), ""])
                    else:
                        lines.extend([label + ": no record available yet.", ""])
                if attempt is not None:
                    lines.extend([f"Attempt implementation accepted: {json.dumps(attempt['accepted'])}", ""])
                if (attempt_dir / "retrieved_memory.json").exists():
                    lines.extend(["Coding memory: " + link(attempt_dir / "retrieved_memory.json"), ""])
        lines.extend(["### Full-Data Execution and Backtest", ""])
        full_results = sorted((directory / "backtest/full").glob("*/execution.json"))
        for path in full_results:
            execution = _read(path)["result"]
            stats = execution.get("stats", {})
            name = stats.get("spec", {}).get("name", path.parent.name)
            lines.extend([f"- {name}: success={execution['success']}, rows={stats.get('rows', 'unknown')}; " + link(path)])
            if execution.get("error"):
                lines.extend([_block(execution["error"])])
        if not full_results:
            lines.append("No full-data factor execution result is available yet.")
        lines.append("")
        dropped = _read(directory / "backtest/dropped_factors.json")
        if dropped:
            lines.extend(["Factors excluded from the backtest and reasons:", _block(json.dumps(dropped, ensure_ascii=False, indent=2)), ""])
        result = experiment.get("result")
        worker = _read(directory / "backtest/worker_result.json")
        if result or worker:
            metrics = result["metrics"] if result else worker["metrics"]
            lines.extend([metric_comparison(metrics, comparator), "",
                          link(directory / "backtest/config.yaml", "Actual training and backtest configuration"),
                          link(directory / "backtest/worker_result.json"),
                          link(directory / "backtest/backtest.log"),
                          link(directory / "backtest/equity_curve.svg", "Cumulative excess return after costs"), ""])
        else:
            lines.extend(["No completed backtest result is available yet.", ""])
        feedback = experiment.get("feedback")
        if feedback:
            lines.extend(["### Research Feedback and Acceptance Decision", "", f"research decision: {json.dumps(feedback['decision'])}", ""])
            for key, label in (("observations", "Observations"), ("hypothesis_evaluation", "Hypothesis evaluation"),
                               ("reason", "Acceptance or rejection rationale"), ("new_hypothesis", "Suggested next hypothesis")):
                lines.extend([label + ":", feedback.get(key, ""), ""])
            if feedback["decision"] and result:
                comparator = result["metrics"]
        else:
            lines.extend(["No research acceptance decision has been made yet.", ""])
        if experiment.get("error"):
            lines.extend(["Experiment error:", _block(experiment["error"]), ""])

    lines.extend(["## Original Model Interactions", "", "These files retain complete requests, original responses, parsed results and token usage. Rendering this section makes no requests.", ""])
    requests = [directory for directory in (root / "llm/requests").glob("*") if (directory / "request.json").is_file()]
    for directory in sorted(requests, key=lambda p: ((p / "request.json").stat().st_mtime_ns, p.name)):
        request = _read(directory / "request.json")
        if request is None:
            continue
        lines.extend([f"### {request.get('purpose', directory.name)}", "",
                      f"model: {request.get('model', 'unknown')}", link(directory / "request.json"), ""])
        for path in sorted(directory.glob("response_*.json")):
            response = _read(path)
            lines.extend([link(path), "usage: " + json.dumps(response.get("usage"), ensure_ascii=False),
                          _block(response.get("text", "")), ""])
        for path in sorted(directory.glob("error_*.json")):
            lines.extend([link(path), _block(json.dumps(_read(path), ensure_ascii=False, indent=2)), ""])
        if not list(directory.glob("response_*.json")):
            lines.extend(["No original model response has been saved yet.", ""])
    embedding_requests = sorted((root / "llm/embeddings").glob("*/request.json"))
    if embedding_requests:
        lines.extend(["### Embedding Retrieval", ""])
        for request in embedding_requests:
            lines.extend(["- " + link(request) + "; " + link(request.parent / "result.json")])
        lines.append("")
    if (root / "llm/replay_state.json").exists():
        lines.extend(["Offline scripted responses and request context: " + link("llm/replay_state.json"), ""])
    return redact("\n".join(str(line) for line in safe_value(lines))) + "\n"


def write_trajectory(root: Path, output: Path | None = None) -> Path:
    """Export atomically; readers never see half of a Markdown trajectory."""
    import tempfile
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / "trajectory.md"
    if output.suffix.lower() != ".md":
        raise ValueError("Trajectory output must be a .md file; JSON, code and raw logs cannot be overwritten")
    if output.exists() and not output.read_text(encoding="utf-8").startswith(("# Agent Research Trajectory\n", "# Agent \u8fd0\u884c\u8f68\u8ff9\n")):
        raise ValueError(f"Refusing to overwrite an existing non-trajectory file: {output}")
    text = render_trajectory(root, relative_to=output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".trajectory-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def snapshot_source(root: Path):
    """Keep the exact small source tree so strict checkpoints remain reproducible."""
    root = Path(root)
    snapshot = root / "source_snapshot"
    if snapshot.exists():
        return
    import tempfile
    temporary = Path(tempfile.mkdtemp(prefix=".source_snapshot-", dir=root))
    package = Path(__file__).parent
    files = {}
    try:
        for path in sorted(package.rglob("*")):
            if path.suffix not in {".py", ".txt", ".yaml"}:
                continue
            relative = Path("factor_agent") / path.relative_to(package)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = "from factor_agent.cli import main\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n"
        (temporary / "run.py").write_text(entry, encoding="utf-8")
        files["run.py"] = hashlib.sha256(entry.encode()).hexdigest()
        (temporary / "manifest.json").write_text(json.dumps({"created": datetime.now(timezone.utc).isoformat(),
                                                            "files": files}, indent=2) + "\n")
        os.replace(temporary, snapshot)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
