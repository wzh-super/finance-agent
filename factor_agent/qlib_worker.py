"""A standalone child process: fixed YAML → Qlib training → exact recorder export.

No factor-agent or RD-Agent imports are needed in the Qlib environment. The local
MLflow logging compatibility fix is preserved from the current RD-Agent checkout.
"""

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import json
import hashlib
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.parse import urlparse


@contextmanager
def synchronous_file_logging():
    """Avoid reading partially written local MLflow metric files."""
    from qlib.workflow.recorder import MLflowRecorder

    original = MLflowRecorder.start_run

    @wraps(original)
    def start_run(recorder, *args, **kwargs):
        result = original(recorder, *args, **kwargs)
        if urlparse(str(recorder.uri)).scheme in ("", "file") and recorder.async_log is not None:
            recorder.async_log.wait()
            recorder.async_log = None
        return result

    with patch.object(MLflowRecorder, "start_run", start_run):
        yield


REQUIRED_METRICS = ("IC", "Rank IC", "1day.excess_return_with_cost.annualized_return",
                    "1day.excess_return_with_cost.information_ratio",
                    "1day.excess_return_with_cost.max_drawdown")


def run(config_path: Path, cache_key: str):
    import pandas as pd
    import qlib
    from qlib.config import C
    from qlib.model.trainer import task_train
    from ruamel.yaml import YAML

    with config_path.open() as handle:
        config = YAML(typ="safe", pure=True).load(handle)
    manager = deepcopy(C["exp_manager"])
    manager["kwargs"]["uri"] = (Path.cwd() / "mlruns").as_uri()
    qlib.init(**config["qlib_init"], exp_manager=manager)
    with synchronous_file_logging():
        recorder = task_train(config["task"], experiment_name="factor_agent")
        recorder.save_objects(config=config)
    if recorder.info["status"] != "FINISHED":
        raise RuntimeError(f"Qlib recorder did not finish: {recorder.info}")
    all_metrics = {key: float(value) for key, value in recorder.list_metrics().items()}
    metrics = {key: value for key, value in all_metrics.items() if math.isfinite(value)}
    for key in REQUIRED_METRICS:
        if key not in metrics or not math.isfinite(metrics[key]):
            raise RuntimeError(f"Missing or nonfinite Qlib metric: {key}")
    report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    if not isinstance(report, pd.DataFrame) or report.empty:
        raise RuntimeError("Qlib portfolio report is empty or invalid")
    pd.Series(metrics, name="value").to_csv("qlib_res.csv")
    report.to_pickle("ret.pkl")
    report.to_csv("returns.csv")
    # This is Qlib's daily portfolio report, not an independently reconstructed backtest.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 4))
    (report["return"] - report["cost"] - report["bench"]).cumsum().plot(ax=axis)
    axis.set(title="Cumulative excess return after costs (Qlib)", ylabel="Cumulative daily excess return")
    fig.tight_layout()
    fig.savefig("equity_curve.svg")
    plt.close(fig)
    trained = recorder.load_object("params.pkl")
    result = {"status": recorder.info["status"], "recorder_id": recorder.info["id"],
              "experiment_id": recorder.info.get("experiment_id"), "metrics": metrics,
              "report_rows": len(report), "feature_count": trained.model.num_feature(),
              "dropped_nonfinite_metrics": sorted(set(all_metrics) - set(metrics)), "cache_key": cache_key,
              "outputs": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("qlib_res.csv", "ret.pkl", "returns.csv", "equity_curve.svg")}}
    temporary = Path("worker_result.json.tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, "worker_result.json")


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), sys.argv[2])
