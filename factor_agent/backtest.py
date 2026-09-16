"""Combine approved factor columns and evaluate the unchanged CPU LightGBM task.

The baseline and combined YAML files intentionally retain RD-Agent's different
preprocessors. This module preserves that research configuration and makes file
provenance, accepted-factor reuse and completion checks explicit.
"""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import pandas as pd
import yaml

from .errors import ExperimentFailed
from .execution import (data_fingerprint, file_hash, fingerprint, run_process,
                        runtime_fingerprint, validate_factor)
from .models import BacktestResult, FactorArtifact, result_from_dict
from .progress import null_event
from .qlib_worker import REQUIRED_METRICS
from .storage import read_json, write_json


def deduplicate_new_factors(accepted: pd.DataFrame, candidate: pd.DataFrame,
                            threshold: float = 0.99) -> tuple[pd.DataFrame, dict[str, str]]:
    """Original rule: mean daily signed Pearson correlation, max over old columns.

    A large negative correlation is intentionally not removed. NaN correlation
    also fails the original `< threshold` comparison. Candidates are compared
    with accepted factors only, not with each other or with Alpha158 columns.
    """
    joined = pd.concat([accepted, candidate], axis=1)
    keep, dropped = [], {}
    for new_name in candidate.columns:
        correlations = []
        for old_name in accepted.columns:
            daily = [day[old_name].corr(day[new_name])
                     for _, day in joined.groupby(level="datetime")]
            correlations.append(pd.Series(daily, dtype=float).mean())
        maximum = pd.Series(correlations, dtype=float).max()
        if maximum < threshold:
            keep.append(new_name)
        else:
            dropped[new_name] = f"Mean daily correlation with accepted factors is {maximum}; required < {threshold}"
    return candidate.loc[:, keep], dropped


def combine_factors(accepted: pd.DataFrame | None, candidate: pd.DataFrame) -> pd.DataFrame:
    """Preserve the original first-round NaNs and later-round complete-case join."""
    combined = pd.concat([accepted, candidate], axis=1).dropna() if accepted is not None else candidate
    if combined.empty:
        raise ExperimentFailed("No complete factor observations remain after combining accepted and new factors")
    combined = combined.sort_index()
    combined = combined.loc[:, ~combined.columns.duplicated(keep="last")].copy()
    combined.columns = pd.MultiIndex.from_product([["feature"], combined.columns])
    return combined


class QlibBacktester:
    def __init__(self, config, executor, event=None):
        self.config = config
        self.executor = executor
        self.event = event or null_event
        self._process_event = event

    @staticmethod
    def _verify_candidate(factor: FactorArtifact):
        """A resumed backtest must still execute the exact code approved by coding."""
        path = Path(factor.code_path)
        matching = [attempt for attempt in factor.attempts if attempt.code_path == factor.code_path]
        manifest = path.parent / ("provenance.json" if matching else "reuse.json")
        try:
            saved = read_json(manifest)
            code_hash = file_hash(path)
            if saved["spec_identity"] != factor.spec.identity or saved["code_sha256"] != code_hash:
                raise ValueError("factor code or specification changed after review")
            if matching and not any(attempt.accepted and attempt.execution.success and
                                    attempt.execution.stats.get("code_sha256", code_hash) == code_hash
                                    for attempt in matching):
                raise ValueError("factor code does not match a successful coding review")
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(f"Candidate {factor.spec.name} provenance is invalid: {error}; use a new run directory") from error

    def _accepted_frame(self, artifact: FactorArtifact, input_fingerprint: str) -> pd.DataFrame:
        """An accepted factor cannot silently vanish or change on a later round."""
        if not artifact.values_path:
            raise ValueError(f"Accepted factor {artifact.spec.name} has no full-data artifact")
        output = Path(artifact.values_path)
        manifest = output.parent / "execution.json"
        try:
            saved = read_json(manifest)["result"]
            stats = saved["stats"]
            if (not saved["success"] or not stats["full"]
                    or stats["spec"] != asdict(artifact.spec)
                    or stats["input_fingerprint"] != input_fingerprint
                    or stats["code_sha256"] != file_hash(Path(artifact.code_path))
                    or stats["output_sha256"] != file_hash(output)):
                raise ValueError("code, market input or saved full-data values changed")
            frame = pd.read_hdf(output, key="data")
            stats = validate_factor(frame, artifact.spec)
            self.event("factor.combine", f"Reusing accepted factor {artifact.spec.name}", action="accepted_reuse",
                       factor=artifact.spec.name, rows=stats["rows"], dates=[stats["start"], stats["end"]],
                       nan_count=stats["nan_count"], code_path=artifact.code_path, values_path=str(output))
            return frame
        except (OSError, KeyError, ValueError) as error:
            self.event("factor.combine", f"Accepted factor {artifact.spec.name} failed integrity checks", action="integrity_failed",
                       factor=artifact.spec.name, error=str(error), values_path=str(output))
            raise ValueError(f"Accepted factor {artifact.spec.name} cannot be reused: {error}") from error

    def _prepare_factors(self, accepted, candidates, directory):
        self.event("factor.combine", "Preparing full-data factors and verifying data fingerprints", action="start",
                   accepted_factors=[factor.spec.name for factor in accepted],
                   candidates=[factor.spec.name for factor in candidates], data_dir=str(self.config.data_dir))
        input_id = data_fingerprint(Path(self.config.data_dir))
        old_names = [factor.spec.name for factor in accepted]
        if len(set(old_names)) != len(old_names):
            raise ValueError("Accepted factor library contains duplicate names")
        old_frames = [self._accepted_frame(factor, input_id) for factor in accepted]
        old = pd.concat(old_frames, axis=1) if old_frames else None
        survivors, frames, dropped, names = [], [], {}, set(old_names)
        for index, factor in enumerate(candidates, start=1):
            name = factor.spec.name
            self.event("factor.combine", f"Processing candidate {index}/{len(candidates)}: {name}", action="candidate",
                       factor=name, candidate_index=index, candidate_total=len(candidates))
            if name in names:
                dropped[name] = "Duplicate factor name already present in the accepted library or this experiment"
                self.event("factor.combine", f"Dropping factor {name}: duplicate name", action="drop", factor=name, reason=dropped[name])
                continue
            names.add(name)
            if not factor.implemented:
                dropped[name] = factor.error or "Factor implementation was not approved"
                self.event("factor.combine", f"Dropping factor {name}: implementation was not approved", action="drop", factor=name, reason=dropped[name])
                continue
            self._verify_candidate(factor)
            result = self.executor.run(Path(factor.code_path), factor.spec,
                                       directory / "full" / factor.spec.identity, full=True)
            if not result.success:
                dropped[name] = result.error or "Full-data execution failed"
                self.event("factor.combine", f"Dropping factor {name}: full-data execution failed", action="drop", factor=name, reason=dropped[name])
                continue
            survivors.append(replace(factor, values_path=result.data_path))
            frames.append(pd.read_hdf(result.data_path, key="data"))
        if not frames:
            write_json(directory / "dropped_factors.json", dropped)
            raise ExperimentFailed("No candidate factor survived full-data execution")
        new = pd.concat(frames, axis=1)
        if old is not None:
            self.event("factor.combine", "Checking daily cross-sectional correlations with accepted factors", action="deduplicate",
                       accepted_count=old.shape[1], candidate_count=new.shape[1], threshold=0.99,
                       correlation="mean daily signed Pearson correlation")
            new, redundant = deduplicate_new_factors(old, new)
            dropped.update(redundant)
            for name, reason in redundant.items():
                self.event("factor.combine", f"Dropping factor {name}: correlation filter rejected the candidate", action="drop", factor=name, reason=reason)
        write_json(directory / "dropped_factors.json", dropped)
        if new.empty:
            raise ExperimentFailed("All new factors duplicate accepted factors by the original correlation rule")
        survivors = [factor for factor in survivors if factor.spec.name in new.columns]
        combined = combine_factors(old, new)
        self.event("factor.combine", "Factor combination finished", action="complete", rows=len(combined),
                   factor_columns=combined.shape[1], factor_names=list(combined.columns.get_level_values(1)),
                   kept_candidates=[factor.spec.name for factor in survivors], dropped_factors=dropped,
                   drop_report_path=str(directory / "dropped_factors.json"))
        return combined, survivors, dropped

    def _finished(self, result, started, *, cached=False, recovered=False):
        self.event("qlib", "Reusing completed backtest" if cached else "Fixed LightGBM training and backtest finished",
                   action="cache_hit" if cached else "complete", recovered_worker=recovered,
                   duration_seconds=round(time.monotonic() - started, 3), status=result.metadata.get("status"),
                   recorder_id=result.metadata.get("recorder_id"), feature_count=result.metadata.get("total_feature_count"),
                   report_rows=result.metadata.get("report_rows"), metrics=result.metrics,
                   factor_names=result.factor_names, workspace=result.workspace,
                   artifacts={name: str(Path(result.workspace) / name) for name in (
                       "backtest.json", "worker_result.json", "config.yaml", "backtest.log", "qlib_res.csv",
                       "ret.pkl", "returns.csv", "equity_curve.svg", "mlruns")})

    def run(self, accepted: list[FactorArtifact], candidates: list[FactorArtifact],
            directory: Path) -> BacktestResult:
        started = time.monotonic()
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        baseline = not accepted and not candidates
        self.event("qlib", "Starting baseline backtest" if baseline else "Starting backtest with candidate factors", action="start", baseline=baseline,
                   accepted_count=len(accepted), candidate_count=len(candidates), workspace=str(directory))
        if accepted and not candidates:
            raise ExperimentFailed("A factor experiment needs at least one new candidate")
        combined, added, dropped = None, [], {}
        if not baseline:
            combined, added, dropped = self._prepare_factors(accepted, candidates, directory)
            combined.to_parquet(directory / "combined_factors_df.parquet", engine="pyarrow")
            self.event("factor.combine", "Qlib factor input table saved", action="saved", rows=len(combined),
                       factor_columns=combined.shape[1], path=str(directory / "combined_factors_df.parquet"))
        template = Path(__file__).parent / "templates" / ("baseline.yaml" if baseline else "combined.yaml")
        config = yaml.safe_load(template.read_text())
        config["qlib_init"]["provider_uri"] = str(Path(self.config.qlib_provider).expanduser().resolve())
        config_path = directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        self.event("qlib", "Fixed model and train/validation/test periods configured", action="configuration",
                   model=config["task"]["model"],
                   segments={name: [str(date) for date in dates]
                             for name, dates in config["task"]["dataset"]["kwargs"]["segments"].items()},
                   portfolio=json.loads(json.dumps(config["port_analysis_config"], default=str)),
                   expected_features=20 + (combined.shape[1] if combined is not None else 0),
                   provider=config["qlib_init"]["provider_uri"], config_path=str(config_path),
                   interpreter=self.config.qlib_python, timeout_seconds=self.config.backtest_timeout)
        worker = Path(__file__).with_name("qlib_worker.py")
        runtime = runtime_fingerprint(self.config.qlib_python)
        provider = Path(self.config.qlib_provider).expanduser().resolve()
        if not (provider / "calendars" / "day.txt").is_file():
            raise ExperimentFailed(f"Missing Qlib daily provider: {provider}")
        dependencies = {"config_sha256": file_hash(config_path), "template_sha256": file_hash(template),
                        "worker_sha256": file_hash(worker), "backtester_sha256": file_hash(Path(__file__)),
                        "qlib_provider_fingerprint": data_fingerprint(provider), "runtime": runtime,
                        "factor_fingerprint": file_hash(directory / "combined_factors_df.parquet") if not baseline else None}
        cache_key = fingerprint(dependencies)
        manifest = directory / "backtest.json"
        outputs = ["worker_result.json", "qlib_res.csv", "ret.pkl", "returns.csv", "equity_curve.svg"]
        if manifest.exists():
            cached = read_json(manifest)
            if cached.get("cache_key") == cache_key and all(
                    (directory / name).is_file() and file_hash(directory / name) == digest
                    for name, digest in cached.get("outputs", {}).items()) and set(cached.get("outputs", {})) == set(outputs):
                result = result_from_dict(cached["result"])
                if result.metadata.get("status") == "FINISHED":
                    self._finished(result, started, cached=True)
                    return result
        # The worker writes its own completion manifest. A crash between child
        # completion and the parent checkpoint must not cause another training run.
        recovered = False
        try:
            previous = read_json(directory / "worker_result.json")
            recovered = (previous.get("cache_key") == cache_key and previous.get("status") == "FINISHED"
                         and set(previous.get("outputs", {})) == set(outputs) - {"worker_result.json"}
                         and all((directory / name).is_file() and file_hash(directory / name) == digest
                                 for name, digest in previous["outputs"].items()))
        except (OSError, ValueError, KeyError):
            pass
        if recovered:
            self.event("qlib", "Recovered completed Qlib results without retraining", action="worker_recovered",
                       recorder_id=previous.get("recorder_id"), worker_result_path=str(directory / "worker_result.json"))
        if not recovered:
            for name in outputs:
                (directory / name).unlink(missing_ok=True)
            options = {"event": self._process_event, "stage": "qlib.process"} if self._process_event is not None else {}
            returncode, stdout = run_process([self.config.qlib_python, str(worker), str(config_path), cache_key], directory,
                                            directory / "backtest.log", self.config.backtest_timeout, **options)
            if returncode != 0:
                self.event("qlib", "Qlib child process failed", action="failed", returncode=returncode,
                           duration_seconds=round(time.monotonic() - started, 3), log_path=str(directory / "backtest.log"))
                raise ExperimentFailed(f"Qlib exited with status {returncode}; see {directory / 'backtest.log'}\n{stdout[-3000:]}")
        if not all((directory / name).is_file() for name in outputs):
            raise ExperimentFailed(f"Qlib did not produce all result artifacts in {directory}")
        payload = read_json(directory / "worker_result.json")
        metrics = payload.get("metrics", {})
        if payload.get("status") != "FINISHED" or not payload.get("recorder_id") or any(
                key not in metrics or not math.isfinite(metrics[key]) for key in REQUIRED_METRICS):
            raise ExperimentFailed("Qlib recorder status or required metrics failed validation")
        names = list(combined.columns.get_level_values(1)) if combined is not None else []
        if payload.get("feature_count") != 20 + len(names):
            raise ExperimentFailed(f"Qlib loaded {payload.get('feature_count')} features; expected {20 + len(names)}")
        result = BacktestResult(metrics, str(directory), names, added, dropped,
                                {**dependencies, "cache_key": cache_key, "status": payload["status"],
                                 "recorder_id": payload["recorder_id"], "base_feature_count": 20,
                                 "total_feature_count": payload.get("feature_count", 20 + len(names)),
                                 "report_rows": payload.get("report_rows"),
                                 "dropped_nonfinite_metrics": payload.get("dropped_nonfinite_metrics", [])})
        write_json(manifest, {"cache_key": cache_key, "outputs": {name: file_hash(directory / name) for name in outputs},
                              "result": asdict(result)})
        self._finished(result, started, recovered=recovered)
        return result
