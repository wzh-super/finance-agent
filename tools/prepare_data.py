"""Export RD-Agent-compatible Full/Debug inputs from an existing Qlib provider.

Adapted from microsoft/RD-Agent v0.8.0:
rdagent/scenarios/qlib/experiment/factor_data_template/generate.py (MIT).
No API calls, downloads, model training or changes to the provider are performed.
"""

import argparse
from pathlib import Path


FIELDS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
DATA_README = """# Daily factor input

Read `daily_pv.h5` with `pandas.read_hdf('daily_pv.h5', key='data')`.
The index is `(datetime, instrument)`, sorted by date and instrument.
Columns: $open, $close, $high, $low (adjusted daily prices), $volume
(daily volume), and $factor (price adjustment factor, not a research factor).
Full data starts on 2008-12-29. Debug data covers 2018-2019 for up to
the first 100 instruments in the Full data, following the upstream recipe.
Missing market values are retained. Use only information available at each date.
"""


def select_frames(raw):
    """Select the upstream date ranges and instrument order without filling NaNs."""
    full = raw.reorder_levels(["datetime", "instrument"]).sort_index()
    full = full.loc["2008-12-29":, FIELDS]
    instruments = full.index.get_level_values("instrument").unique()[:100]
    debug = full.loc["2018-01-01":"2019-12-31"]
    debug = debug.loc[debug.index.get_level_values("instrument").isin(instruments)]
    if full.empty or debug.empty:
        raise ValueError("Provider must contain Full data and observations in the 2018-2019 Debug period")
    return full, debug


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True, help="Existing Qlib China daily provider")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Parent of factor_full and factor_debug")
    args = parser.parse_args(argv)
    provider = args.provider.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (provider / "calendars/day.txt").is_file():
        parser.error(f"Qlib daily calendar not found: {provider}")
    targets = [output / name for name in ("factor_full", "factor_debug")]
    for target in targets:
        if target.exists():
            parser.error(f"Output already exists: {target}; choose a new --output directory")

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region="cn")
    print("Reading the full Qlib instrument universe; this may take several minutes.", flush=True)
    frames = select_frames(D.features(D.instruments(), FIELDS, freq="day"))
    for target, frame in zip(targets, frames):
        target.mkdir(parents=True)
        frame.to_hdf(target / "daily_pv.h5", key="data", mode="w")
        (target / "README.md").write_text(DATA_README, encoding="utf-8")
        print(f"Saved {target}: {len(frame):,} rows, {frame.index.get_level_values('instrument').nunique()} instruments")


if __name__ == "__main__":
    main()
