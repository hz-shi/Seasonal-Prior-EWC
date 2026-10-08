"""Tests for chunked streaming normalization fitting.

Verifies that the streaming reduction matches the previous whole-array
``np.nanmean``/``np.nanstd``/``np.nanmin``/``np.nanmax`` results, is independent
of chunk size, handles all-NaN channels, and that save/load round-trips the fit
range while remaining compatible with older npz files.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import normalization  # noqa: E402
from src.normalization import PM25Normalizer  # noqa: E402


class _FakeCore:
    def __init__(self, n_days=40, h=4, w=5, seed=0, v2_all_nan=False):
        rng = np.random.default_rng(seed)
        self.pm25_time = pd.date_range("2018-01-01", periods=n_days, freq="D").values
        self._x = rng.normal(size=(n_days, h, w)).astype(np.float32)
        self._y = rng.uniform(0.0, 100.0, size=(n_days, h, w)).astype(np.float32)
        self._v2_all_nan = v2_all_nan
        self.source_specs = [("srcA", None, ["v1", "v2"])]
        self.time_index_map = {"srcA": np.arange(n_days)}

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

    def get_feature_days(self, name, var, idx):
        base = self._x[idx]
        if var == "v2":
            if self._v2_all_nan:
                return np.full_like(base, np.nan)
            return base * 2.0
        return base

    def get_pm25_days(self, idx):
        return self._y[idx]


def _reference_days(core, in_len, out_len, start, end):
    starts = core.find_valid_start_indices(start, end, in_len, out_len)
    in_days = np.unique(np.concatenate([np.arange(s, s + in_len) for s in starts]))
    out_days = np.unique(np.concatenate([np.arange(s + in_len, s + in_len + out_len) for s in starts]))
    return in_days, out_days


class StreamingReductionTest(unittest.TestCase):
    def test_chunk_size_independence(self):
        core = _FakeCore()
        idx = np.arange(len(core.pm25_time))
        get = lambda chunk: core.get_feature_days("srcA", "v1", chunk)
        for chunk in (1, 3, 7, 32, 1000):
            mean, std = normalization._streaming_mean_std(get, idx, chunk_size=chunk)
            mn, mx = normalization._streaming_minmax(get, idx, chunk_size=chunk)
            ref = core.get_feature_days("srcA", "v1", idx)
            self.assertAlmostEqual(mean, float(np.nanmean(ref)), places=5)
            self.assertAlmostEqual(std, float(np.nanstd(ref)), places=5)
            self.assertAlmostEqual(mn, float(np.nanmin(ref)), places=5)
            self.assertAlmostEqual(mx, float(np.nanmax(ref)), places=5)

    def test_all_nan_returns_neutral(self):
        idx = np.arange(10)
        get = lambda chunk: np.full((len(chunk), 2, 2), np.nan, dtype=np.float32)
        self.assertEqual(normalization._streaming_mean_std(get, idx), (0.0, 1.0))
        self.assertEqual(normalization._streaming_minmax(get, idx), (0.0, 1.0))

    def test_inf_is_ignored(self):
        idx = np.arange(4)
        data = np.array([1.0, 2.0, np.inf, -np.inf], dtype=np.float32).reshape(4, 1, 1)
        get = lambda chunk: data[chunk]
        mean, std = normalization._streaming_mean_std(get, idx)
        self.assertAlmostEqual(mean, 1.5, places=6)
        mn, mx = normalization._streaming_minmax(get, idx)
        self.assertEqual((mn, mx), (1.0, 2.0))


class FitMatchesReferenceTest(unittest.TestCase):
    def _fit(self, core, x_mode, y_mode):
        normalizer = PM25Normalizer(x_mode=x_mode, y_mode=y_mode)
        normalizer.fit(
            core=core,
            in_len=5,
            out_len=3,
            train_target_start="2018-01-01",
            train_target_end="2018-01-20",
        )
        return normalizer

    def test_minmax_matches_reference(self):
        core = _FakeCore()
        normalizer = self._fit(core, "minmax_01", "minmax_01")
        in_days, out_days = _reference_days(core, 5, 3, "2018-01-01", "2018-01-20")
        src_idx = np.unique(core.time_index_map["srcA"][in_days])
        ref_v1 = core.get_feature_days("srcA", "v1", src_idx)
        ref_v2 = core.get_feature_days("srcA", "v2", src_idx)
        self.assertAlmostEqual(float(normalizer.x_p1[0]), float(np.nanmin(ref_v1)), places=5)
        self.assertAlmostEqual(float(normalizer.x_p2[0]), float(np.nanmax(ref_v1) - np.nanmin(ref_v1)), places=5)
        self.assertAlmostEqual(float(normalizer.x_p1[1]), float(np.nanmin(ref_v2)), places=5)
        ref_y = core.get_pm25_days(out_days)
        self.assertAlmostEqual(float(normalizer.y_p1[0]), float(np.nanmin(ref_y)), places=5)
        self.assertAlmostEqual(float(normalizer.y_p2[0]), float(np.nanmax(ref_y) - np.nanmin(ref_y)), places=5)

    def test_zscore_matches_reference(self):
        core = _FakeCore()
        normalizer = self._fit(core, "zscore", "zscore")
        in_days, out_days = _reference_days(core, 5, 3, "2018-01-01", "2018-01-20")
        src_idx = np.unique(core.time_index_map["srcA"][in_days])
        ref_v1 = core.get_feature_days("srcA", "v1", src_idx)
        self.assertAlmostEqual(float(normalizer.x_p1[0]), float(np.nanmean(ref_v1)), places=5)
        self.assertAlmostEqual(float(normalizer.x_p2[0]), float(np.nanstd(ref_v1)), places=5)
        ref_y = core.get_pm25_days(out_days)
        self.assertAlmostEqual(float(normalizer.y_p1[0]), float(np.nanmean(ref_y)), places=5)
        self.assertAlmostEqual(float(normalizer.y_p2[0]), float(np.nanstd(ref_y)), places=5)

    def test_all_nan_channel_neutral(self):
        core = _FakeCore(v2_all_nan=True)
        normalizer = self._fit(core, "minmax_01", "minmax_01")
        self.assertEqual(float(normalizer.x_p1[1]), 0.0)
        self.assertEqual(float(normalizer.x_p2[1]), 1.0)

    def test_fit_range_recorded(self):
        core = _FakeCore()
        normalizer = self._fit(core, "minmax_01", "minmax_01")
        self.assertEqual(normalizer.fit_target_start, "2018-01-01")
        self.assertEqual(normalizer.fit_target_end, "2018-01-20")


class SaveLoadTest(unittest.TestCase):
    def test_roundtrip_preserves_fit_range(self):
        core = _FakeCore()
        normalizer = PM25Normalizer(x_mode="minmax_01", y_mode="minmax_01")
        normalizer.fit(core, 5, 3, "2018-01-01", "2018-01-20")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stats.npz")
            normalizer.save(path)
            loaded = PM25Normalizer.load(path)
        self.assertEqual(loaded.fit_target_start, "2018-01-01")
        self.assertEqual(loaded.fit_target_end, "2018-01-20")
        np.testing.assert_allclose(loaded.x_p1, normalizer.x_p1)
        np.testing.assert_allclose(loaded.y_p2, normalizer.y_p2)
        self.assertEqual(loaded.fingerprint(), normalizer.fingerprint())

    def test_load_legacy_npz_without_fit_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.npz")
            np.savez_compressed(
                path,
                x_mode=np.array(["minmax_01"]),
                y_mode=np.array(["minmax_01"]),
                x_p1=np.array([1.0], dtype=np.float32),
                x_p2=np.array([2.0], dtype=np.float32),
                y_p1=np.array([3.0], dtype=np.float32),
                y_p2=np.array([4.0], dtype=np.float32),
                fingerprint=np.array(["deadbeef"]),
            )
            loaded = PM25Normalizer.load(path)
        self.assertIsNone(loaded.fit_target_start)
        self.assertIsNone(loaded.fit_target_end)
        self.assertEqual(float(loaded.x_p1[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
