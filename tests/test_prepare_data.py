"""Small synthetic inputs verify the data-export recipe without loading Qlib."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location("prepare_data", Path(__file__).resolve().parents[1] / "tools/prepare_data.py")
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


class PrepareDataTests(unittest.TestCase):
    def test_export_selection_preserves_schema_missing_values_and_upstream_windows(self):
        dates = pd.to_datetime(["2008-12-28", "2008-12-29", "2018-01-02", "2019-12-31", "2020-01-02"])
        index = pd.MultiIndex.from_product([[f"S{i:03}" for i in range(102)], dates], names=["instrument", "datetime"])
        raw = pd.DataFrame(1.0, index=index, columns=prepare_data.FIELDS)
        raw.loc[("S000", pd.Timestamp("2018-01-02")), "$factor"] = np.nan
        full, debug = prepare_data.select_frames(raw.iloc[::-1])
        self.assertEqual(full.index.names, ["datetime", "instrument"])
        self.assertTrue(full.index.is_monotonic_increasing)
        self.assertEqual(full.shape, (408, 6))
        self.assertEqual(debug.shape, (200, 6))
        self.assertEqual(list(debug.columns), prepare_data.FIELDS)
        self.assertEqual(debug.index.get_level_values("instrument").nunique(), 100)
        self.assertEqual(debug.index.get_level_values("datetime").min(), pd.Timestamp("2018-01-02"))
        self.assertTrue(pd.isna(debug.loc[(pd.Timestamp("2018-01-02"), "S000"), "$factor"]))

    def test_missing_debug_period_is_rejected(self):
        index = pd.MultiIndex.from_tuples([("A", pd.Timestamp("2021-01-01"))], names=["instrument", "datetime"])
        with self.assertRaisesRegex(ValueError, "2018-2019"):
            prepare_data.select_frames(pd.DataFrame(1.0, index=index, columns=prepare_data.FIELDS))
