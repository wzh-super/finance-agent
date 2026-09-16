"""Execute factor.py against Debug/Full inputs and validate its file contract.

Each attempt has its own working directory. The child receives no API credentials;
this is local execution, not a security sandbox for untrusted Python. Successful
outputs are reused only when code, inputs, interpreter and output hashes match.
"""

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import codecs
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import pandas as pd

from .models import ExecutionResult, FactorSpec
from .progress import null_event
from .storage import read_json, write_json


_HASH_CACHE: dict[tuple, str] = {}


def file_hash(path: Path) -> str:
    path = Path(path)
    stat = path.stat()
    signature = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature in _HASH_CACHE:
        return _HASH_CACHE[signature]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _HASH_CACHE[signature] = value
    return value


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def data_fingerprint(directory: Path) -> str:
    """Hash every input file; filenames alone cannot identify market data."""
    paths = [path for path in sorted(directory.rglob("*")) if path.is_file()]
    # Qlib stores thousands of small feature files. Concurrent reads avoid a
    # long serial filesystem round trip per file; the resulting digest is identical.
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = pool.map(file_hash, paths)
        return fingerprint({str(path.relative_to(directory)): value for path, value in zip(paths, hashes)})


def runtime_fingerprint(executable: str) -> dict:
    """Read package versions in the actual child interpreter, without importing Qlib."""
    resolved = shutil.which(executable) or executable
    script = (
        "import importlib.metadata as m,json,sys; "
        "wanted={'pandas','numpy','tables','pyarrow','pyqlib','lightgbm','mlflow','scipy',"
        "'matplotlib','pyyaml','ruamel.yaml','ruamel-yaml','scikit-learn'}; "
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'packages':{d.metadata['Name'].lower():d.version for d in m.distributions() "
        "if d.metadata['Name'].lower() in wanted}}))"
    )
    result = subprocess.run([resolved, "-c", script], capture_output=True, text=True,
                            env=child_environment(), timeout=60, check=True)
    return json.loads(result.stdout)


def child_environment() -> dict[str, str]:
    """Preserve execution settings, remove credentials and network proxy settings."""
    sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "PROXY")
    env = {key: value for key, value in os.environ.items()
           if not any(word in key.upper() for word in sensitive)}
    # Generated factors do not need parent Python import hooks or site customizations.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)
    env.update(MLFLOW_ALLOW_FILE_STORE="true", MLFLOW_DISABLE_TELEMETRY="true",
               PYTHONUNBUFFERED="1")
    return env


class _LogFollower:
    """Read the child's regular log without a pipe or partial UTF-8 corruption."""

    def __init__(self, path: Path, event, stage: str):
        self.reader = path.open("rb")
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending = ""
        self.event, self.stage, self.path = event, stage, path

    def emit(self, *, final=False) -> bool:
        text = self.pending + self.decoder.decode(self.reader.read(), final=final)
        lines = text.splitlines(keepends=True)
        self.pending = ""
        if lines and not final and not lines[-1].endswith(("\n", "\r")):
            self.pending = lines.pop()
        # Carriage-return progress updates become separate readable lines. The
        # log itself retains every byte, including carriage returns and blanks.
        message = "\n".join(line.rstrip("\r\n") for line in lines).strip("\n")
        if message:
            self.event(self.stage, message, action="stdout", stream="stdout+stderr", log_path=str(self.path))
            return True
        return False

    def close(self):
        self.reader.close()


def run_process(argv: list[str], directory: Path, log_path: Path, timeout: int, *,
                event=None, stage="process") -> tuple[int, str]:
    """Save the entire child log and optionally stream it while work is running.

    A regular file is the output sink, so a slow observer cannot deadlock a full
    stdout pipe. Polling limits event batches to twice per second. Silent children
    produce a heartbeat every 30 seconds. Without an observer the original single
    wait call remains unchanged. Timeout/interruption kills the whole child group.
    """
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(argv, cwd=directory, env=child_environment(),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        follower = None
        try:
            if event is None:
                code = process.wait(timeout=timeout)
            else:
                follower = _LogFollower(log_path, event, stage)
                event(stage, "Child process started", action="start", argv=argv, pid=process.pid,
                      directory=str(directory), log_path=str(log_path), timeout_seconds=timeout)
                last_visible = started
                while True:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    try:
                        code = process.wait(timeout=min(0.5, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if follower.emit():
                            last_visible = time.monotonic()
                        if time.monotonic() - last_visible >= 30:
                            event(stage, "Child process is still running; waiting for new output", action="heartbeat", pid=process.pid,
                                  elapsed_seconds=round(time.monotonic() - started, 3), log_path=str(log_path))
                            last_visible = time.monotonic()
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            log.write(f"\nExecution exceeded {timeout} seconds; child process group terminated.\n".encode())
            log.flush()
            code = -signal.SIGKILL
            if event is not None:
                event(stage, "Child process timed out; the entire process group was terminated", action="timeout", timeout_seconds=timeout,
                      log_path=str(log_path))
        finally:
            if follower is not None:
                try:
                    follower.emit(final=True)
                finally:
                    follower.close()
        if event is not None:
            event(stage, "Child process finished", action="complete", returncode=code,
                  duration_seconds=round(time.monotonic() - started, 3), log_path=str(log_path))
    return code, log_path.read_text(errors="replace")


def validate_factor(frame: pd.DataFrame, spec: FactorSpec) -> dict:
    """Check the documented daily single-factor interface deterministically.

    Warm-up NaNs are allowed. Empty, all-NaN, infinite, duplicate, nonnumeric or
    intraday outputs fail before entering Qlib. These checks make the original
    interface explicit; they are engineering validation, not proof of no leakage.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Factor output must be a nonempty pandas DataFrame")
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != ["datetime", "instrument"]:
        raise ValueError("Factor index must be MultiIndex(datetime, instrument), in that order")
    if frame.index.has_duplicates:
        raise ValueError("Factor contains duplicate datetime/instrument rows")
    if list(frame.columns) != [spec.name]:
        raise ValueError(f"Factor must contain exactly one column named {spec.name!r}")
    dates = frame.index.get_level_values("datetime")
    if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or dates.tz is not None:
        raise ValueError("datetime must contain timezone-naive timestamps without NaT")
    if not (dates == dates.normalize()).all():
        raise ValueError("Factor output must use daily midnight timestamps")
    instruments = frame.index.get_level_values("instrument")
    if not all(isinstance(value, str) and value for value in instruments.unique()):
        raise ValueError("instrument must contain nonempty strings")
    values = frame.iloc[:, 0]
    if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError("Factor values must be numeric")
    if np.isinf(values.to_numpy(dtype=float)).any() or values.notna().sum() == 0:
        raise ValueError("Factor values contain infinity or no finite observations")
    return {"rows": len(frame), "columns": list(frame.columns), "index": frame.index.names,
            "start": str(dates.min()), "end": str(dates.max()),
            "instruments": len(instruments.unique()), "nan_count": int(values.isna().sum())}


class FactorExecutor:
    def __init__(self, config, event=None):
        self.config = config
        self._runtime = None
        self.event = event or null_event
        self._process_event = event

    def _finished(self, stage, spec, result, directory, started, *, cached=False):
        stats = result.stats
        self.event(stage, f"{spec.name}: " + ("reusing verified execution results" if cached else "factor execution finished"),
                   action="cache_hit" if cached else "complete", factor=spec.name, success=result.success,
                   rows=stats.get("rows"), dates=[stats.get("start"), stats.get("end")],
                   instruments=stats.get("instruments"), nan_count=stats.get("nan_count"),
                   duration_seconds=round(time.monotonic() - started, 3), error=result.error,
                   output_path=result.data_path, log_path=str(directory / "execution.log"),
                   manifest_path=str(directory / "execution.json"))

    def run(self, code_path: Path, spec: FactorSpec, directory: Path, *, full=False) -> ExecutionResult:
        started = time.monotonic()
        stage = "factor.full" if full else "factor.debug"
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "execution.json"
        try:
            source = Path(self.config.data_dir if full else self.config.debug_data_dir).resolve()
            self.event(stage, f"{spec.name}: starting {'full-data' if full else 'debug-sample'} execution", action="start",
                       factor=spec.name, code_path=str(code_path), data_dir=str(source), directory=str(directory),
                       timeout_seconds=self.config.factor_timeout)
            if not (source / "daily_pv.h5").is_file():
                raise FileNotFoundError(f"Missing market data: {source / 'daily_pv.h5'}")
            if self._runtime is None:
                self._runtime = runtime_fingerprint(self.config.python_executable)
            dependencies = {"code_sha256": file_hash(code_path), "input_fingerprint": data_fingerprint(source),
                            "spec": asdict(spec), "runtime": self._runtime, "full": full,
                            "executor_sha256": file_hash(Path(__file__))}
            cache_key = fingerprint(dependencies)
            output = directory / "result.h5"
            if manifest.exists():
                saved = read_json(manifest)
                result = ExecutionResult(**saved["result"])
                if (saved.get("cache_key") == cache_key and result.success and output.is_file()
                        and result.stats.get("output_sha256") == file_hash(output)):
                    self._finished(stage, spec, result, directory, started, cached=True)
                    return result
            # The subprocess can never consume an output left by a prior attempt.
            output.unlink(missing_ok=True)
            target = directory / "factor.py"
            if code_path.resolve() != target:
                shutil.copyfile(code_path, target)
            for path in source.iterdir():
                if path.is_file():
                    if path.name in {"factor.py", "result.h5", "execution.json", "execution.log"}:
                        raise ValueError(f"Reserved filename in source data: {path.name}")
                    link = directory / path.name
                    link.unlink(missing_ok=True)
                    link.symlink_to(path)
            options = {"event": self._process_event, "stage": stage} if self._process_event is not None else {}
            returncode, stdout = run_process([self.config.python_executable, "factor.py"], directory,
                                             directory / "execution.log", self.config.factor_timeout, **options)
            if returncode != 0:
                result = ExecutionResult(False, stdout=stdout, returncode=returncode,
                                         error=f"factor.py exited with status {returncode}")
            elif not output.is_file():
                result = ExecutionResult(False, stdout=stdout, returncode=returncode,
                                         error="factor.py did not create result.h5")
            else:
                try:
                    self.event(stage, f"{spec.name}: reading and validating output", action="validation", factor=spec.name,
                               output_path=str(output))
                    frame = pd.read_hdf(output, key="data")
                    stats = validate_factor(frame, spec)
                    stats.update(dependencies, output_sha256=file_hash(output))
                    result = ExecutionResult(True, stdout=stdout, data_path=str(output), stats=stats,
                                             returncode=returncode)
                except Exception as error:
                    result = ExecutionResult(False, stdout=stdout, error=str(error), returncode=returncode)
                    self.event(stage, f"{spec.name}: output validation failed", action="validation_failed", factor=spec.name,
                               error=str(error), output_path=str(output))
            write_json(manifest, {"cache_key": cache_key, "result": asdict(result)})
            self._finished(stage, spec, result, directory, started)
            return result
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            result = ExecutionResult(False, error=str(error))
            self._finished(stage, spec, result, directory, started)
            return result
