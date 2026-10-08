"""Tests for known-future GFS covariates as extra input feature planes.

The paper protocol feeds the model the GFS fields for the forecast days
``t+1 .. t+out_len`` in addition to the historical input window. These tests
verify, on small real xarray/zarr fixtures:

* the channel layout (history channels unchanged, then lead-major GFS planes
  broadcast across the history time steps), with ``in_len`` still 5;
* that only GFS is read for the future dates (no future ERA5/MEIC), and that
  the PM2.5 target never leaks into ``x``;
* that the serial, threaded and preloaded access paths produce identical
  channel order;
* that a horizon mismatch is rejected;
* that normalization statistics cover the new channels, reuse one statistic
  per GFS variable across leads, are fitted only on the 2017-2018 range, and
  that the input-schema fingerprint/schema guard rejects legacy 41-channel
  statistics.

On a machine without torch the heavy imports are stubbed so the pure
numpy/xarray logic can still run; on the L20 server the real torch is used.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _install_torch_stub() -> None:
    """Install a minimal torch/pytorch_lightning stub for CPU-only machines."""
    torch_mod = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")

    class _Dataset:
        pass

    class _DataLoader:
        pass

    torch_utils_data.Dataset = _Dataset
    torch_utils_data.DataLoader = _DataLoader
    torch_utils_data.get_worker_info = lambda: None
    torch_utils.data = torch_utils_data
    torch_mod.utils = torch_utils
    torch_mod.from_numpy = lambda a: a
    torch_mod.Tensor = object
    torch_mod.__getattr__ = lambda name: mock.MagicMock(name="torch." + name)
    sys.modules["torch"] = torch_mod
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

    for name in [
        "pytorch_lightning",
        "pytorch_lightning.strategies",
        "pytorch_lightning.loggers",
        "pytorch_lightning.callbacks",
    ]:
        sys.modules[name] = mock.MagicMock(name=name)
    pl = sys.modules["pytorch_lightning"]
    pl.LightningDataModule = type("LightningDataModule", (), {})
    pl.LightningModule = type("LightningModule", (), {})


def _ensure_real_data_module():
    """Import the real ``src.data`` even if a sibling test stubbed it."""
    real_torch = False
    try:
        import torch  # noqa: F401

        real_torch = isinstance(getattr(torch, "__version__", None), str)
    except ImportError:
        real_torch = False

    if not real_torch:
        _install_torch_stub()
        # Sibling tests may have replaced these with MagicMock; drop the stubs
        # so the real modules (which only need the torch stub above) import.
        for name in [
            "src.data",
            "src.reg",
            "src.reg.conditions",
            "src.reg.condition_aware_ewc",
        ]:
            if isinstance(sys.modules.get(name), mock.MagicMock):
                del sys.modules[name]

    import importlib

    return importlib.import_module("src.data")


_data = _ensure_real_data_module()
MultiSourcePM25Core = _data.MultiSourcePM25Core
DataPaths = _data.DataPaths

from src.normalization import PM25Normalizer  # noqa: E402

YEAR_VARS = ["DEM", "LUC", "NTL"]
ERAL_VARS = ["blh", "d2m", "e", "sp", "t2m", "tp", "u10", "v10"]
ERAP_VARS = [f"{p}{lvl}" for p in ["r", "t", "u", "v"] for lvl in [100, 200, 300, 400, 500]]
GFS_VARS = ["rh", "t2m", "tp", "u10", "v10"]
MEIC_VARS = ["SO2", "NOx", "PM10", "PM25", "NH3"]
HISTORY_CHANNELS = len(YEAR_VARS) + len(ERAL_VARS) + len(ERAP_VARS) + len(GFS_VARS) + len(MEIC_VARS)
GFS_OFFSET = len(YEAR_VARS) + len(ERAL_VARS) + len(ERAP_VARS)


def _source_ds(var_names, time, lat, lon, value_fn):
    data = {}
    for i, v in enumerate(var_names):
        arr = np.empty((len(time), len(lat), len(lon)), dtype=np.float32)
        for t in range(len(time)):
            arr[t] = value_fn(i, t)
        data[v] = (("time", "lat", "lon"), arr)
    return xr.Dataset(data, coords={"time": time, "lat": lat, "lon": lon})


class _Fixture:
    """Small real zarr fixture with yearly / monthly / daily sources."""

    def __init__(self, root, n_days, h, w, gfs_2019_sentinel=False):
        self.root = root
        self.n_days = n_days
        self.h = h
        self.w = w
        self.time = pd.date_range("2017-01-01", periods=n_days, freq="D")
        self.lat = np.arange(h, dtype=np.float32)
        self.lon = np.arange(w, dtype=np.float32)

        years = pd.DatetimeIndex(
            [pd.Timestamp(y, 1, 1) for y in sorted({d.year for d in self.time})]
        )
        months = pd.DatetimeIndex(
            sorted({pd.Timestamp(d.year, d.month, 1) for d in self.time})
        )

        def _write(name, ds):
            path = os.path.join(root, f"{name}.zarr")
            ds.to_zarr(path, mode="w")
            return path

        self.paths = {}
        self.paths["year"] = _write(
            "year",
            _source_ds(YEAR_VARS, years, self.lat, self.lon, lambda i, t: 1000.0 + i),
        )
        self.paths["eral"] = _write(
            "eral",
            _source_ds(ERAL_VARS, self.time, self.lat, self.lon, lambda i, t: 5000.0 + i),
        )
        self.paths["erap"] = _write(
            "erap",
            _source_ds(ERAP_VARS, self.time, self.lat, self.lon, lambda i, t: 7000.0 + i),
        )

        def _gfs_value(i, t):
            if gfs_2019_sentinel and self.time[t].year == 2019:
                return 1.0e6
            return float(i * 100 + t)

        self.paths["gfs"] = _write(
            "gfs", _source_ds(GFS_VARS, self.time, self.lat, self.lon, _gfs_value)
        )
        self.paths["meic"] = _write(
            "meic",
            _source_ds(MEIC_VARS, months, self.lat, self.lon, lambda i, t: 3000.0 + i),
        )

        pm = np.empty((n_days, h, w), dtype=np.float32)
        for t in range(n_days):
            pm[t] = float(t)
        self.paths["pm25"] = _write(
            "pm25",
            xr.Dataset(
                {"pm25": (("time", "lat", "lon"), pm)},
                coords={"time": self.time, "lat": self.lat, "lon": self.lon},
            ),
        )

    def data_paths(self):
        return DataPaths(
            year_path=self.paths["year"],
            eral_path=self.paths["eral"],
            erap_path=self.paths["erap"],
            gfs_path=self.paths["gfs"],
            meic_path=self.paths["meic"],
            pm25_path=self.paths["pm25"],
        )

    def core(self, **kwargs):
        return MultiSourcePM25Core(self.data_paths(), **kwargs)


class FutureGFSChannelLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fx = _Fixture(cls._tmp.name, n_days=20, h=4, w=5)
        cls.core = cls.fx.core(future_gfs_enabled=True, future_gfs_steps=3)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_channel_counts_are_computed(self):
        self.assertEqual(self.core.history_channels, HISTORY_CHANNELS)
        self.assertEqual(self.core.known_future_channels, 3 * len(GFS_VARS))
        self.assertEqual(
            self.core.input_channels, HISTORY_CHANNELS + 3 * len(GFS_VARS)
        )

    def test_feature_names_are_auditable(self):
        names = self.core.input_feature_names
        self.assertEqual(len(names), self.core.input_channels)
        # Historical names follow the core's authoritative source/variable order.
        expected_hist = [
            f"{name}/{v}" for name, _, vars_ in self.core.source_specs for v in vars_
        ]
        self.assertEqual(names[:HISTORY_CHANNELS], expected_hist)
        self.assertEqual(
            names[HISTORY_CHANNELS:],
            [
                f"gfs/lead{lead + 1}/{v}"
                for lead in range(3)
                for v in self.core.gfs_vars
            ],
        )
        self.assertEqual(self.core.channel_names, names)

    def test_in_len_still_five_and_future_lead_major(self):
        x, y = self.core.get_window(2, 5, 3)
        self.assertEqual(x.shape, (5, 4, 5, self.core.input_channels))
        self.assertEqual(y.shape, (3, 4, 5, 1))
        gfs_index = {v: i for i, v in enumerate(GFS_VARS)}
        future = x[..., HISTORY_CHANNELS:]
        for lead in range(3):
            for vi, v in enumerate(self.core.gfs_vars):
                ch = lead * len(self.core.gfs_vars) + vi
                expected = float(gfs_index[v] * 100 + (2 + 5 + lead))
                self.assertAlmostEqual(float(future[0, 0, 0, ch]), expected, places=5)
        # Broadcast across the 5 history time steps.
        for ch in range(future.shape[-1]):
            self.assertTrue(np.allclose(future[:, 0, 0, ch], future[0, 0, 0, ch]))

    def test_future_channels_come_from_gfs_only(self):
        x, _ = self.core.get_window(2, 5, 3)
        # ERA5/ERAL/ERAP history channels carry their distinctive constants.
        eral_start = len(YEAR_VARS)
        self.assertAlmostEqual(float(x[0, 0, 0, eral_start]), 5000.0, places=5)
        erap_start = len(YEAR_VARS) + len(ERAL_VARS)
        self.assertAlmostEqual(float(x[0, 0, 0, erap_start]), 7000.0, places=5)
        # No future channel equals the ERA5/ERAP constants.
        future = x[..., HISTORY_CHANNELS:]
        self.assertFalse(np.any(np.isclose(future, 5000.0)))
        self.assertFalse(np.any(np.isclose(future, 7000.0)))

    def test_horizon_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.core.get_window(2, 5, 2)
        with self.assertRaises(ValueError):
            self.core.get_window(2, 5, 4)

    def test_legacy_disabled_keeps_41_channels(self):
        legacy = self.fx.core(future_gfs_enabled=False)
        self.assertFalse(legacy.future_gfs_enabled)
        self.assertEqual(legacy.known_future_channels, 0)
        self.assertEqual(legacy.input_channels, HISTORY_CHANNELS)
        x, _ = legacy.get_window(2, 5, 3)
        self.assertEqual(x.shape, (5, 4, 5, HISTORY_CHANNELS))
        # Legacy path accepts any out_len (no known-future horizon coupling).
        x2, _ = legacy.get_window(2, 5, 2)
        self.assertEqual(x2.shape, (5, 4, 5, HISTORY_CHANNELS))


class FutureGFSAccessConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fx = _Fixture(cls._tmp.name, n_days=20, h=4, w=5)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_serial_thread_preload_identical(self):
        serial = self.fx.core(future_gfs_enabled=True, future_gfs_steps=3)
        threaded = self.fx.core(
            zarr_parallel_backend="thread",
            zarr_read_workers=2,
            future_gfs_enabled=True,
            future_gfs_steps=3,
        )
        preloaded = self.fx.core(
            preload_to_memory=True, future_gfs_enabled=True, future_gfs_steps=3
        )
        x_serial, y_serial = serial.get_window(2, 5, 3)
        x_thread, y_thread = threaded.get_window(2, 5, 3)
        x_pre, y_pre = preloaded.get_window(2, 5, 3)
        np.testing.assert_array_equal(x_serial, x_thread)
        np.testing.assert_array_equal(x_serial, x_pre)
        np.testing.assert_array_equal(y_serial, y_thread)
        np.testing.assert_array_equal(y_serial, y_pre)

    def test_pm25_target_does_not_leak_into_x(self):
        core_a = self.fx.core(future_gfs_enabled=True, future_gfs_steps=3)
        x_a, _ = core_a.get_window(2, 5, 3)
        # Rewrite the PM2.5 store with different values and rebuild the core.
        pm = np.full((self.fx.n_days, self.fx.h, self.fx.w), 999.0, dtype=np.float32)
        xr.Dataset(
            {"pm25": (("time", "lat", "lon"), pm)},
            coords={"time": self.fx.time, "lat": self.fx.lat, "lon": self.fx.lon},
        ).to_zarr(self.fx.paths["pm25"], mode="w")
        core_b = self.fx.core(future_gfs_enabled=True, future_gfs_steps=3)
        x_b, y_b = core_b.get_window(2, 5, 3)
        np.testing.assert_array_equal(x_a, x_b)
        self.assertTrue(np.allclose(y_b, 999.0))


class FutureGFSNormalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fx = _Fixture(
            cls._tmp.name, n_days=1095, h=2, w=3, gfs_2019_sentinel=True
        )
        cls.future_core = cls.fx.core(future_gfs_enabled=True, future_gfs_steps=3)
        cls.legacy_core = cls.fx.core(future_gfs_enabled=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _fit(self, core):
        normalizer = PM25Normalizer(x_mode="minmax_01", y_mode="minmax_01")
        normalizer.fit(
            core=core,
            in_len=5,
            out_len=3,
            train_target_start="2017-01-01",
            train_target_end="2018-12-31",
        )
        return normalizer

    def test_stats_cover_all_channels(self):
        normalizer = self._fit(self.future_core)
        self.assertEqual(len(normalizer.x_p1), self.future_core.input_channels)
        self.assertEqual(len(normalizer.x_p2), self.future_core.input_channels)
        self.assertEqual(normalizer.input_channels, self.future_core.input_channels)
        self.assertTrue(normalizer.future_gfs_enabled)
        self.assertEqual(normalizer.future_gfs_steps, 3)
        self.assertEqual(
            normalizer.feature_names, self.future_core.input_feature_names
        )

    def test_lead_channels_reuse_gfs_variable_stats(self):
        normalizer = self._fit(self.future_core)
        n_gfs = len(self.future_core.gfs_vars)
        for vi in range(n_gfs):
            base = GFS_OFFSET + vi
            for lead in range(3):
                ch = HISTORY_CHANNELS + lead * n_gfs + vi
                self.assertAlmostEqual(
                    float(normalizer.x_p1[ch]), float(normalizer.x_p1[base]), places=6
                )
                self.assertAlmostEqual(
                    float(normalizer.x_p2[ch]), float(normalizer.x_p2[base]), places=6
                )

    def test_fit_range_excludes_2019(self):
        normalizer = self._fit(self.future_core)
        self.assertEqual(normalizer.fit_target_start, "2017-01-01")
        self.assertEqual(normalizer.fit_target_end, "2018-12-31")
        # The 2019 GFS sentinel (1e6) must not enter the fitted range.
        gfs_stats = normalizer.x_p1[GFS_OFFSET:GFS_OFFSET + len(GFS_VARS)]
        self.assertTrue(np.all(gfs_stats < 1.0e5))
        future_stats = normalizer.x_p1[HISTORY_CHANNELS:]
        self.assertTrue(np.all(future_stats < 1.0e5))

    def test_fingerprint_distinguishes_future_flag(self):
        future_norm = self._fit(self.future_core)
        legacy_norm = self._fit(self.legacy_core)
        self.assertNotEqual(future_norm.fingerprint(), legacy_norm.fingerprint())

    def test_schema_guard_rejects_legacy_stats(self):
        future_norm = self._fit(self.future_core)
        ok, _ = future_norm.validate_against_core(self.future_core)
        self.assertTrue(ok)
        ok, reason = future_norm.validate_against_core(self.legacy_core)
        self.assertFalse(ok)
        self.assertTrue(reason)

        legacy_norm = self._fit(self.legacy_core)
        ok, _ = legacy_norm.validate_against_core(self.legacy_core)
        self.assertTrue(ok)
        ok, reason = legacy_norm.validate_against_core(self.future_core)
        self.assertFalse(ok)
        self.assertTrue(reason)

    def test_save_load_roundtrip_preserves_schema(self):
        normalizer = self._fit(self.future_core)
        path = os.path.join(self._tmp.name, "future_stats.npz")
        normalizer.save(path)
        loaded = PM25Normalizer.load(path)
        self.assertEqual(loaded.input_channels, normalizer.input_channels)
        self.assertTrue(loaded.future_gfs_enabled)
        self.assertEqual(loaded.future_gfs_steps, 3)
        self.assertEqual(loaded.feature_names, normalizer.feature_names)
        self.assertEqual(loaded.fingerprint(), normalizer.fingerprint())
        ok, _ = loaded.validate_against_core(self.future_core)
        self.assertTrue(ok)

    def test_schema_guard_rejects_same_count_with_reordered_features(self):
        normalizer = self._fit(self.future_core)
        normalizer.feature_names[0], normalizer.feature_names[1] = (
            normalizer.feature_names[1], normalizer.feature_names[0]
        )
        ok, reason = normalizer.validate_against_core(self.future_core)
        self.assertFalse(ok)
        self.assertIn("order", reason)

    def test_schema_guard_rejects_truncated_scale_statistics(self):
        normalizer = self._fit(self.future_core)
        normalizer.x_p2 = normalizer.x_p2[:-1]
        ok, reason = normalizer.validate_against_core(self.future_core)
        self.assertFalse(ok)
        self.assertIn("channels", reason)

    def test_legacy_npz_without_schema_is_rejected_for_future_core(self):
        path = os.path.join(self._tmp.name, "legacy_stats.npz")
        np.savez_compressed(
            path,
            x_mode=np.array(["minmax_01"]),
            y_mode=np.array(["minmax_01"]),
            x_p1=np.zeros(HISTORY_CHANNELS, dtype=np.float32),
            x_p2=np.ones(HISTORY_CHANNELS, dtype=np.float32),
            y_p1=np.array([0.0], dtype=np.float32),
            y_p2=np.array([1.0], dtype=np.float32),
            fingerprint=np.array(["deadbeef"]),
        )
        loaded = PM25Normalizer.load(path)
        self.assertIsNone(loaded.input_channels)
        self.assertFalse(loaded.future_gfs_enabled)
        ok, reason = loaded.validate_against_core(self.future_core)
        self.assertFalse(ok)
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
