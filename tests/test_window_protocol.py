"""Tests for the rolling-window train/validation target protocol.

Manuscript Section 3.3: within a 60-day optimization window the first 53 target
days are optimized and the latest 7 target days validated. These tests verify
the resolved target-date ranges are strictly disjoint and that the sample counts
are 51 (train) and 5 (val) for a 3-day horizon, not 53/7.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._heavy_stubs import install_stubs  # noqa: E402

install_stubs()

from src import rolling  # noqa: E402


class _FakeCore:
    """Minimal core exposing the two methods the protocol helpers need."""

    def __init__(self, start="2018-01-01", end="2019-12-31"):
        self.pm25_time = pd.date_range(start, end, freq="D").values

    def find_valid_start_indices(self, target_start, target_end, in_len, out_len):
        ts = pd.Timestamp(target_start)
        te = pd.Timestamp(target_end)
        valid = []
        total = len(self.pm25_time)
        max_start = total - (in_len + out_len)
        for s in range(max_start + 1):
            tgt = self.pm25_time[s + in_len : s + in_len + out_len]
            if tgt[0] >= ts and tgt[-1] <= te:
                valid.append(s)
        return np.array(valid, dtype=np.int64)


class WindowProtocolTest(unittest.TestCase):
    def test_first_forecast_uses_latest_60_observed_target_days(self):
        origin = pd.Timestamp("2018-12-27")
        start, end = rolling._build_window_bounds(origin, window_days=60, in_len=5)
        self.assertEqual(start, pd.Timestamp("2018-11-02"))
        self.assertEqual(end, pd.Timestamp("2018-12-31"))
        protocol = rolling.resolve_window_target_protocol(start, end, 5, 3, True)
        self.assertEqual(protocol.train_target_end, pd.Timestamp("2018-12-24"))
        self.assertEqual(protocol.val_target_start, pd.Timestamp("2018-12-25"))
        self.assertEqual(protocol.val_target_end, pd.Timestamp("2018-12-31"))
        actual = rolling.assert_window_protocol_disjoint(_FakeCore(), protocol, 5, 3)
        self.assertEqual(actual["target_day_overlap"], 0)
        self.assertEqual((actual["train_samples_actual"], actual["val_samples_actual"]), (51, 5))

    def test_cross_season_forecast_has_no_unobserved_training_targets(self):
        origin = pd.Timestamp("2019-02-22")
        start, end = rolling._build_window_bounds(origin, window_days=60, in_len=5)
        forecast_days = pd.date_range(origin + pd.Timedelta(days=5), periods=3)
        protocol = rolling.resolve_window_target_protocol(start, end, 5, 3, True)
        self.assertEqual(end, pd.Timestamp("2019-02-26"))
        self.assertEqual(forecast_days[-1], pd.Timestamp("2019-03-01"))
        self.assertLess(protocol.val_target_end, forecast_days[0])
        self.assertEqual((protocol.train_target_days, protocol.val_target_days), (53, 7))

    def test_three_day_origins_cover_every_2019_day_once(self):
        first, last = pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31")
        origins = pd.date_range(first - pd.Timedelta(days=5), last - pd.Timedelta(days=2), freq="3D")
        counts = {day: 0 for day in pd.date_range(first, last)}
        for origin in origins:
            _, end = rolling._build_window_bounds(origin, 60, 5)
            forecast_days = pd.date_range(origin + pd.Timedelta(days=5), periods=3)
            self.assertLess(end, forecast_days[0])
            for day in forecast_days:
                if day in counts:
                    counts[day] += 1
        self.assertEqual(len(origins), 123)
        self.assertTrue(all(count == 1 for count in counts.values()))
        self.assertEqual(counts[last], 1)
        # Keep the manuscript's 123 updates; the final one forecasts 2020 only.
        self.assertGreater(origins[-1] + pd.Timedelta(days=5), last)

    def test_60day_window_splits_53_7(self):
        protocol = rolling.resolve_window_target_protocol(
            pd.Timestamp("2018-11-01"),
            pd.Timestamp("2018-12-30"),
            in_len=5,
            out_len=3,
            enable_val=True,
        )
        self.assertEqual(protocol.train_target_days, 53)
        self.assertEqual(protocol.val_target_days, 7)
        # 53 target days -> 51 three-day samples; 7 target days -> 5 samples.
        self.assertEqual(protocol.train_samples, 51)
        self.assertEqual(protocol.val_samples, 5)
        self.assertEqual(protocol.train_target_start, pd.Timestamp("2018-11-01"))
        self.assertEqual(protocol.train_target_end, pd.Timestamp("2018-12-23"))
        self.assertEqual(protocol.val_target_start, pd.Timestamp("2018-12-24"))
        self.assertEqual(protocol.val_target_end, pd.Timestamp("2018-12-30"))
        # Strictly disjoint target-date ranges.
        self.assertLess(protocol.train_target_end, protocol.val_target_start)

    def test_disabled_uses_full_window(self):
        protocol = rolling.resolve_window_target_protocol(
            pd.Timestamp("2018-11-01"),
            pd.Timestamp("2018-12-30"),
            in_len=5,
            out_len=3,
            enable_val=False,
        )
        self.assertEqual(protocol.train_target_days, 60)
        self.assertEqual(protocol.train_samples, 58)
        self.assertIsNone(protocol.val_target_start)
        self.assertIsNone(protocol.val_target_end)
        self.assertEqual(protocol.val_samples, 0)

    def test_short_window_raises(self):
        with self.assertRaises(ValueError):
            rolling.resolve_window_target_protocol(
                pd.Timestamp("2018-12-01"),
                pd.Timestamp("2018-12-05"),
                in_len=5,
                out_len=3,
                enable_val=True,
            )

    def test_actual_samples_are_disjoint(self):
        core = _FakeCore()
        protocol = rolling.resolve_window_target_protocol(
            pd.Timestamp("2018-11-01"),
            pd.Timestamp("2018-12-30"),
            in_len=5,
            out_len=3,
            enable_val=True,
        )
        record = rolling.assert_window_protocol_disjoint(core, protocol, in_len=5, out_len=3)
        self.assertEqual(record["target_day_overlap"], 0)
        self.assertEqual(record["train_samples_actual"], 51)
        self.assertEqual(record["val_samples_actual"], 5)

    def test_overlap_detection_catches_bad_protocol(self):
        core = _FakeCore()
        # Deliberately overlapping ranges (train end == val start).
        bad = rolling.WindowTargetProtocol(
            train_target_start=pd.Timestamp("2018-11-01"),
            train_target_end=pd.Timestamp("2018-12-24"),
            val_target_start=pd.Timestamp("2018-12-24"),
            val_target_end=pd.Timestamp("2018-12-30"),
            train_target_days=54,
            val_target_days=7,
            train_samples=52,
            val_samples=5,
        )
        with self.assertRaises(RuntimeError):
            rolling.assert_window_protocol_disjoint(core, bad, in_len=5, out_len=3)


if __name__ == "__main__":
    unittest.main()
