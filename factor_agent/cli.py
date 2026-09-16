"""Assemble components at the CLI boundary; pipeline.py owns the research loop.

check validates the environment without calling an LLM; run/resume use the
configured API. replay uses labelled responses with real factor/Qlib execution.
"""

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from .config import AgentConfig
from .errors import ExperimentFailed, ServiceError
from .progress import redact
from .storage import RunStore, read_json, write_json


@contextmanager
def run_lock(directory: Path):
    """One writer per run: prevents duplicate API calls/training and lost updates."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process is using this run directory: {directory}") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parser():
    result = argparse.ArgumentParser(description="Readable finance factor research: hypothesis → code → backtest → feedback")
    commands = result.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="Read/export a saved trajectory; no API, execution or data scan")
    show.add_argument("--run-dir", type=Path, required=True)
    show.add_argument("--output", type=Path, help="Write a Markdown trajectory instead of displaying it")
    for name, description in (("check", "Validate data and interpreter paths without API calls"),
                              ("run", "Start a new live API research run"),
                              ("resume", "Resume a saved live API run"),
                              ("replay", "Replay labelled responses with real factor execution and Qlib backtests")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", type=Path, help="YAML settings; relative data paths use this file's directory")
        command.add_argument("--env-file", type=Path, default=Path(".env"), help="API configuration (default: cwd/.env)")
        if name != "check":
            command.add_argument("--run-dir", type=Path, required=True, help="Dedicated artifact/checkpoint directory")
            command.add_argument("--rounds", type=int, default=1, help="Target TOTAL completed rounds, including earlier rounds")
        if name == "replay":
            command.add_argument("--fixture", type=Path, required=True, help="JSON with explicitly scripted LLM responses")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "show":
            from .trajectory import render_trajectory, write_trajectory
            if args.output is None:
                print(render_trajectory(args.run_dir), end="")
            else:
                path = write_trajectory(args.run_dir, args.output)
                print(f"Trajectory exported: {path} (saved artifacts only; no research executed)")
            return 0
        load_dotenv(args.env_file, override=False)
        config = AgentConfig.from_file(args.config)
        config.validate_paths()
        if args.command == "check":
            from .execution import runtime_fingerprint
            print(f"Full data: {config.data_dir / 'daily_pv.h5'}")
            print(f"Debug data: {config.debug_data_dir / 'daily_pv.h5'}")
            print(f"Qlib data: {config.qlib_provider}")
            for name in ("python_executable", "qlib_python"):
                runtime = runtime_fingerprint(getattr(config, name))
                print(f"{name}: {runtime['executable']}; packages={runtime['packages']}")
            print("Configuration paths and interpreters are available. No API request or training was started.")
            return 0
        if args.rounds <= 0:
            raise ValueError("--rounds must be positive")
        return _run(args, config)
    except (ExperimentFailed, ServiceError, ValueError, FileNotFoundError) as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed stages are saved; use resume (or replay) with the same run directory.", file=sys.stderr)
        return 130


def _run(args, config):
    # Delayed imports keep --help small and prevent optional runtime work at import.
    from .backtest import QlibBacktester
    from .coding import FactorCoder
    from .execution import FactorExecutor
    from .llm import LLMClient, ReplayClient
    from .memory import CodingMemory
    from .pipeline import Pipeline
    from .research import Researcher
    from .trajectory import snapshot_source

    with run_lock(args.run_dir):
        store = RunStore(args.run_dir)
        store.event("run", "Starting research session", mode="offline_replay" if args.command == "replay" else "live_api",
                    command=args.command, target_rounds=args.rounds, run_dir=store.root,
                    agent_python=getattr(config, "python_executable", None),
                    qlib_python=getattr(config, "qlib_python", None),
                    chat_model=getattr(config, "chat_model", None),
                    embedding_model=getattr(config, "embedding_model", None),
                    progress_log=store.root / "progress.log")
        if args.command == "run" and store.state_path.exists():
            raise ValueError("This directory already has a checkpoint; use resume or choose a new run directory")
        if args.command == "resume" and not store.state_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {store.state_path}")
        mode_path = store.root / "run_mode.json"
        previous_mode = read_json(mode_path).get("mode") if mode_path.is_file() else None
        if store.state_path.exists() and previous_mode not in {"offline_replay", "live_api"}:
            raise ValueError("Checkpoint run mode is missing or invalid; restore run_mode.json before resuming")
        if args.command == "resume" and previous_mode == "offline_replay":
            raise ValueError("This is an offline replay; resume with replay and the original --fixture")
        if args.command == "replay" and previous_mode == "live_api":
            raise ValueError("Cannot switch a live run to an offline replay")
        if args.command == "run" and previous_mode == "offline_replay":
            raise ValueError("Cannot switch an offline replay to a live run; choose a new directory")
        if args.command != "replay" and not os.environ.get("OPENAI_API_KEY"):
            raise ServiceError("OPENAI_API_KEY is missing; check --env-file before starting a live run")
        # Validate before writing descriptive artifacts; a rejected resume must
        # not overwrite the configuration that produced the saved experiment.
        store.event("check", "Validating source, configuration, data and dependency fingerprints; the first data scan may take time")
        if store.state_path.exists():
            store.load(config.fingerprint())
        snapshot_source(store.root)
        llm = (ReplayClient(args.fixture, store.root / "llm", event=store.event) if args.command == "replay"
               else LLMClient(config, store.root / "llm", event=store.event))
        executor = FactorExecutor(config, event=store.event)
        memory = CodingMemory(store.root / "coding_memory.json", llm, config, event=store.event)
        write_json(store.root / "config.json", config.public_dict())
        write_json(store.root / "run_mode.json", {"mode": "offline_replay" if args.command == "replay" else "live_api",
                                                  "rounds_target": args.rounds})
        pipeline = Pipeline(config, store, Researcher(llm, config),
                            FactorCoder(llm, executor, memory, config, event=store.event),
                            QlibBacktester(config, executor, event=store.event))
        try:
            state = pipeline.run(args.rounds)
        except BaseException as error:
            store.event("paused", "Run interrupted; completed requests and stages are preserved", error_type=type(error).__name__,
                        error=str(error), checkpoint=store.state_path)
            raise
        finally:
            # Even an unfinished coding attempt can contain valuable code/reviews.
            # Never mask the primary error if exporting a reading view fails.
            try:
                store.write_trajectory()
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(redact(f"Trajectory export failed; original artifacts are preserved: {error}"), file=sys.stderr)
        store.event("run", "Research session complete", completed_rounds=len(state.history),
                    accepted_factors=[f.spec.name for f in state.accepted_factors],
                    report=store.root / "report.md", trajectory=store.root / "trajectory.md")
    return 0
