import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .normalization import PM25Normalizer
from .reg.conditions import condition_ids_for_start_indices


def as_day_index(time_values: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(time_values)).normalize()


def build_time_mapper(src_time: np.ndarray, pm25_time: np.ndarray) -> np.ndarray:
    src_idx = as_day_index(src_time)
    pm_idx = as_day_index(pm25_time)

    src_set = set(src_idx.values)
    if all(t in src_set for t in pm_idx.values):
        mapping = {t: i for i, t in enumerate(src_idx.values)}
        return np.array([mapping[t] for t in pm_idx.values], dtype=np.int64)

    src_years = np.array([d.year for d in src_idx], dtype=np.int64)
    if len(np.unique(src_years)) == len(src_idx):
        mapping = {y: i for i, y in enumerate(src_years)}
        available_years = np.array(sorted(mapping.keys()), dtype=np.int64)

        def nearest_year_index(year: int) -> int:
            if year in mapping:
                return mapping[year]
            nearest_pos = int(np.argmin(np.abs(available_years - year)))
            return mapping[int(available_years[nearest_pos])]

        return np.array([nearest_year_index(d.year) for d in pm_idx], dtype=np.int64)

    src_ym = np.array([d.year * 100 + d.month for d in src_idx], dtype=np.int64)
    if len(np.unique(src_ym)) == len(src_idx):
        mapping = {ym: i for i, ym in enumerate(src_ym)}
        return np.array([mapping[d.year * 100 + d.month] for d in pm_idx], dtype=np.int64)

    raise ValueError("Unsupported source time frequency; expected daily/yearly/monthly.")


@dataclass
class DataPaths:
    year_path: str
    eral_path: str
    erap_path: str
    gfs_path: str
    meic_path: str
    pm25_path: str


class MultiSourcePM25Core:
    def __init__(
        self,
        paths: DataPaths,
        zarr_parallel_backend: str = "serial",
        zarr_read_workers: int = 1,
        preload_to_memory: bool = False,
        preload_workers: int = 0,
        future_gfs_enabled: bool = False,
        future_gfs_steps: int = 0,
    ):
        self.year_ds = xr.open_zarr(paths.year_path)
        self.eral_ds = xr.open_zarr(paths.eral_path)
        self.erap_ds = xr.open_zarr(paths.erap_path)
        self.gfs_ds = xr.open_zarr(paths.gfs_path)
        self.meic_ds = xr.open_zarr(paths.meic_path)
        self.pm25_ds = xr.open_zarr(paths.pm25_path)

        self.zarr_parallel_backend = zarr_parallel_backend
        self.zarr_read_workers = max(1, int(zarr_read_workers))

        self.pm25_var = "pm25"
        self.pm25_time = as_day_index(self.pm25_ds.time.values)
        self.height = int(self.pm25_ds.sizes["lat"])
        self.width = int(self.pm25_ds.sizes["lon"])
        self._pm25_lat = self.pm25_ds["lat"].values.astype(np.float32)
        self._pm25_lon = self.pm25_ds["lon"].values.astype(np.float32)

        self.source_specs = [
            ("year", self.year_ds, list(self.year_ds.data_vars)),
            ("eral", self.eral_ds, list(self.eral_ds.data_vars)),
            ("erap", self.erap_ds, list(self.erap_ds.data_vars)),
            ("gfs", self.gfs_ds, list(self.gfs_ds.data_vars)),
            ("meic", self.meic_ds, list(self.meic_ds.data_vars)),
        ]

        # GFS variable order is captured before any preload closes the dataset so
        # the known-future channel layout stays stable across access backends.
        self.gfs_vars = list(self.gfs_ds.data_vars)

        self.time_index_map: Dict[str, np.ndarray] = {
            name: build_time_mapper(ds.time.values, self.pm25_time.values)
            for name, ds, _ in self.source_specs
        }

        # Known-future GFS covariates: for each forecast lead (t+1 .. t+out_len)
        # the GFS fields are appended as extra feature planes, broadcast across
        # the history time steps. This is explicit conditional-feature
        # concatenation, not additional history time steps.
        self.future_gfs_enabled = bool(future_gfs_enabled)
        self.future_gfs_steps = int(future_gfs_steps) if self.future_gfs_enabled else 0
        if self.future_gfs_enabled and self.future_gfs_steps <= 0:
            raise ValueError(
                "future_gfs_enabled=True requires future_gfs_steps > 0 "
                "(the forecast horizon / out_len)."
            )
        self.history_channels = int(sum(len(vars_) for _, _, vars_ in self.source_specs))
        self.known_future_channels = (
            self.future_gfs_steps * len(self.gfs_vars) if self.future_gfs_enabled else 0
        )
        self.input_channels = self.history_channels + self.known_future_channels

        # ── preload all zarr data into RAM ──────────────────────────────
        self._mem_data: Dict[str, Dict[str, np.ndarray]] | None = None

        if preload_to_memory:
            t0 = time.time()
            # Preserve the declared variable order: the preload futures complete
            # out of order, so rebuilding source_specs from dict insertion order
            # would silently permute feature channels relative to the zarr path.
            source_var_order = {name: list(vars_) for name, _, vars_ in self.source_specs}
            ds_map = {
                "year": self.year_ds,
                "eral": self.eral_ds,
                "erap": self.erap_ds,
                "gfs": self.gfs_ds,
                "meic": self.meic_ds,
                "pm25": self.pm25_ds,
            }
            self._mem_data = {}

            # Build variable-level task list for maximum parallelism
            var_tasks = []
            for name, ds in ds_map.items():
                for v in ds.data_vars:
                    var_tasks.append((name, ds, v))

            # Auto-tune worker count and blosc internal threads
            n_cpus = os.cpu_count() or 4
            if preload_workers <= 0:
                n_workers = min(len(var_tasks), n_cpus)
            else:
                n_workers = max(1, int(preload_workers))

            try:
                from numcodecs import blosc as _blosc
                _prev_nt = _blosc.get_nthreads()
                blosc_nt = max(1, n_cpus // n_workers)
                _blosc.set_nthreads(blosc_nt)
                _has_blosc = True
            except ImportError:
                _has_blosc = False
                blosc_nt = 0

            print(f"[preload] {len(var_tasks)} variables, {n_workers} workers, "
                  f"blosc_nthreads={blosc_nt} (cpus={n_cpus})", flush=True)

            def _preload_variable(name: str, ds, var: str):
                arr = ds[var].values
                nbytes = arr.nbytes
                return name, var, arr, nbytes

            total_bytes = 0
            done_count = 0
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_preload_variable, n, ds, v) for n, ds, v in var_tasks]
                for f in as_completed(futures):
                    name, var, arr, nbytes = f.result()
                    self._mem_data.setdefault(name, {})[var] = arr
                    total_bytes += nbytes
                    done_count += 1
                    if done_count % 10 == 0 or done_count == len(var_tasks):
                        print(f"[preload] {done_count}/{len(var_tasks)} vars done "
                              f"({total_bytes/1e9:.2f} GB)", flush=True)

            if _has_blosc:
                _blosc.set_nthreads(_prev_nt)

            # Close datasets; data now lives in numpy arrays
            for ds in ds_map.values():
                ds.close()
            self.year_ds = None
            self.eral_ds = None
            self.erap_ds = None
            self.gfs_ds = None
            self.meic_ds = None
            self.pm25_ds = None
            self.source_specs = [
                ("year", None, source_var_order["year"]),
                ("eral", None, source_var_order["eral"]),
                ("erap", None, source_var_order["erap"]),
                ("gfs", None, source_var_order["gfs"]),
                ("meic", None, source_var_order["meic"]),
            ]
            elapsed = time.time() - t0
            print(f"[preload] All sources loaded in {elapsed:.1f}s "
                  f"(~{total_bytes / 1e9:.2f} GB, kept original dtype)", flush=True)

    # ── unified PM2.5 access (preload-aware) ───────────────────────────

    @property
    def pm25_lat(self) -> np.ndarray:
        return self._pm25_lat

    @property
    def pm25_lon(self) -> np.ndarray:
        return self._pm25_lon

    @property
    def input_feature_names(self) -> list:
        """Auditable channel names in exact feature-plane order.

        Historical channels are ``"<source>/<var>"`` in ``source_specs`` order;
        when future GFS inputs are enabled the known-future channels follow in
        lead-major order as ``"gfs/lead<k>/<var>"`` (k = 1..future_gfs_steps).
        """
        names = [f"{name}/{v}" for name, _, vars_ in self.source_specs for v in vars_]
        if self.future_gfs_enabled:
            for lead in range(self.future_gfs_steps):
                for v in self.gfs_vars:
                    names.append(f"gfs/lead{lead + 1}/{v}")
        return names

    @property
    def channel_names(self) -> list:
        """Alias of :attr:`input_feature_names` for auditability."""
        return self.input_feature_names

    def _check_future_horizon(self, out_len: int) -> None:
        if self.future_gfs_enabled and int(out_len) != self.future_gfs_steps:
            raise ValueError(
                "future_gfs_enabled=True requires out_len == future_gfs_steps "
                f"({self.future_gfs_steps}); got out_len={out_len}. The known-future "
                "GFS channels are built for the configured forecast horizon."
            )

    def _append_future_gfs_zarr(
        self,
        x_hist: np.ndarray,
        out_idx: np.ndarray,
        lat_slice: slice,
        lon_slice: slice,
    ) -> np.ndarray:
        """Append known-future GFS planes (lead-major) to a history feature stack."""
        if not self.future_gfs_enabled:
            return x_hist
        in_len, h, w, hist_c = x_hist.shape
        x = np.empty((in_len, h, w, self.input_channels), dtype=np.float32)
        x[..., :hist_c] = x_hist
        gfs_map = self.time_index_map["gfs"][out_idx]
        # Read each GFS variable once for all leads, then broadcast per lead.
        gfs_future = {
            v: self._read_feature_arr(self.gfs_ds, v, gfs_map.copy(), lat_slice, lon_slice)
            for v in self.gfs_vars
        }
        ch = hist_c
        for lead in range(self.future_gfs_steps):
            for v in self.gfs_vars:
                x[:, :, :, ch] = gfs_future[v][lead]
                ch += 1
        return x

    def _append_future_gfs_mem(
        self,
        x: np.ndarray,
        out_idx: np.ndarray,
        lat_slice: slice,
        lon_slice: slice,
        ch: int,
    ) -> int:
        """Fill known-future GFS planes (lead-major) from preloaded arrays."""
        if not self.future_gfs_enabled:
            return ch
        gfs_map = self.time_index_map["gfs"][out_idx]
        for lead in range(self.future_gfs_steps):
            for v in self.gfs_vars:
                x[:, :, :, ch] = self._mem_data["gfs"][v][gfs_map[lead], lat_slice, lon_slice]
                ch += 1
        return ch

    def get_pm25_day(self, time_idx: int) -> np.ndarray:
        """Return PM2.5 ground-truth for a single day as float32 (lat, lon)."""
        if self._mem_data is not None and "pm25" in self._mem_data:
            return self._mem_data["pm25"][self.pm25_var][time_idx].astype(np.float32)
        return self.pm25_ds[self.pm25_var].isel(time=time_idx).values.astype(np.float32)

    def get_pm25_days(self, time_indices: np.ndarray) -> np.ndarray:
        """Return PM2.5 ground-truth for multiple days as float32 (time, lat, lon)."""
        if self._mem_data is not None and "pm25" in self._mem_data:
            return self._mem_data["pm25"][self.pm25_var][time_indices].astype(np.float32)
        return self.pm25_ds[self.pm25_var].isel(
            time=xr.DataArray(time_indices, dims="sample")
        ).values.astype(np.float32)

    def get_feature_days(self, name: str, var: str, time_indices: np.ndarray) -> np.ndarray:
        """Return a source feature for multiple days as float32 (time, lat, lon)."""
        if self._mem_data is not None and name in self._mem_data:
            return self._mem_data[name][var][time_indices].astype(np.float32)
        for sn, ds, vars_ in self.source_specs:
            if sn == name and var in vars_:
                return ds[var].isel(
                    time=xr.DataArray(time_indices, dims="sample")
                ).values.astype(np.float32)
        raise KeyError(f"Feature {name}/{var} not found")

    @staticmethod
    def _read_feature_arr(ds, var_name: str, map_idx: np.ndarray, lat_slice: slice, lon_slice: slice) -> np.ndarray:
        for attempt in range(4):
            try:
                return ds[var_name].isel(
                    time=xr.DataArray(map_idx, dims="sample"),
                    lat=lat_slice,
                    lon=lon_slice,
                ).values.astype(np.float32)
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.2 * (2**attempt))
        raise RuntimeError("unreachable zarr read retry state")

    def _effective_zarr_read_workers(self) -> int:
        read_workers = self.zarr_read_workers
        if torch.utils.data.get_worker_info() is not None:
            raw_limit = os.environ.get("PM25_ZARR_READ_WORKERS_IN_DATALOADER", "1")
            try:
                worker_limit = max(1, int(raw_limit))
            except ValueError:
                worker_limit = 1
            read_workers = min(read_workers, worker_limit)
        return max(1, int(read_workers))

    def get_window(
        self,
        start_idx: int,
        in_len: int,
        out_len: int,
        lat_slice: slice | None = None,
        lon_slice: slice | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._check_future_horizon(out_len)
        if self._mem_data is not None:
            return self._get_window_mem(start_idx, in_len, out_len, lat_slice, lon_slice)

        in_idx = np.arange(start_idx, start_idx + in_len, dtype=np.int64)
        out_idx = np.arange(start_idx + in_len, start_idx + in_len + out_len, dtype=np.int64)

        lat_slice = lat_slice if lat_slice is not None else slice(None)
        lon_slice = lon_slice if lon_slice is not None else slice(None)

        tasks = []
        for name, ds, vars_ in self.source_specs:
            map_idx = self.time_index_map[name][in_idx]
            for v in vars_:
                tasks.append((ds, v, map_idx.copy(), lat_slice, lon_slice))

        read_workers = self._effective_zarr_read_workers()
        if self.zarr_parallel_backend == "thread" and read_workers > 1:
            max_workers = min(read_workers, len(tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                feats = list(ex.map(lambda t: self._read_feature_arr(*t), tasks))
        else:
            feats = [self._read_feature_arr(*t) for t in tasks]

        x_hist = np.stack(feats, axis=-1)
        del feats
        x = self._append_future_gfs_zarr(x_hist, out_idx, lat_slice, lon_slice)
        y = self.pm25_ds[self.pm25_var].isel(
            time=xr.DataArray(out_idx, dims="sample"),
            lat=lat_slice,
            lon=lon_slice,
        ).values.astype(np.float32)
        y = y[..., None]
        return x, y

    def _get_window_mem(
        self,
        start_idx: int,
        in_len: int,
        out_len: int,
        lat_slice: slice | None,
        lon_slice: slice | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        in_idx = np.arange(start_idx, start_idx + in_len, dtype=np.int64)
        out_idx = np.arange(start_idx + in_len, start_idx + in_len + out_len, dtype=np.int64)

        lat_slice = lat_slice if lat_slice is not None else slice(None)
        lon_slice = lon_slice if lon_slice is not None else slice(None)

        C = self.input_channels
        h = len(range(*lat_slice.indices(self.height)))
        w = len(range(*lon_slice.indices(self.width)))
        x = np.empty((in_len, h, w, C), dtype=np.float32)
        ch = 0
        for name, _, vars_ in self.source_specs:
            map_idx = self.time_index_map[name][in_idx]
            for v in vars_:
                x[:, :, :, ch] = self._mem_data[name][v][map_idx, lat_slice, lon_slice]
                ch += 1
        ch = self._append_future_gfs_mem(x, out_idx, lat_slice, lon_slice, ch)

        y = np.empty((out_len, h, w, 1), dtype=np.float32)
        y[:, :, :, 0] = self._mem_data["pm25"][self.pm25_var][out_idx, lat_slice, lon_slice]
        return x, y

    def find_valid_start_indices(self, target_start: str, target_end: str, in_len: int, out_len: int) -> np.ndarray:
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





class PM25WindowDataset(Dataset):
    def __init__(
        self,
        core: MultiSourcePM25Core,
        start_indices: np.ndarray,
        in_len: int,
        out_len: int,
        patch_h: int,
        patch_w: int,
        random_patch: bool,
        normalizer: PM25Normalizer,
        shm_cache_enabled: bool = False,
        shm_cache_dir: str = "/dev/shm/pm25_window_cache",
        shm_cache_max_items: int = 0,
        shm_cache_min_free_gb: float = 2.0,
        shm_cache_x_dtype: str = "float32",
        condition_scheme: str = "season4",
        patches_per_sample: int = 1,
    ):
        self.core = core
        self.start_indices = start_indices
        self.in_len = in_len
        self.out_len = out_len
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.random_patch = random_patch
        self.normalizer = normalizer
        self.shm_cache_enabled = shm_cache_enabled
        self.shm_cache_dir = shm_cache_dir
        self.shm_cache_max_items = max(0, int(shm_cache_max_items))
        self.shm_cache_min_free_bytes = int(max(0.0, float(shm_cache_min_free_gb)) * (1024**3))
        self.shm_cache_x_dtype = shm_cache_x_dtype
        self._cache_is_f16 = shm_cache_x_dtype == "float16"
        self.patches_per_sample = max(1, int(patches_per_sample)) if self.random_patch else 1
        self._cache_write_counter = 0
        self.condition_ids = condition_ids_for_start_indices(
            pm25_time=self.core.pm25_time,
            start_indices=self.start_indices,
            in_len=self.in_len,
            condition_scheme=condition_scheme,
        )

        if self.shm_cache_enabled:
            os.makedirs(self.shm_cache_dir, exist_ok=True)

        norm_parts = [self.normalizer.x_mode, self.normalizer.y_mode]
        for arr in [self.normalizer.x_p1, self.normalizer.x_p2, self.normalizer.y_p1, self.normalizer.y_p2]:
            if arr is not None:
                norm_parts.append(str(arr.shape))
                norm_parts.append(f"{float(np.mean(arr)):.6f}")
        self._norm_tag = hashlib.md5("|".join(norm_parts).encode("utf-8")).hexdigest()[:12]

    def _stratified_patch_position(self, base_idx: int, patch_slot: int, h: int, w: int) -> Tuple[int, int]:
        max_top = h - self.patch_h
        max_left = w - self.patch_w
        rng = np.random.default_rng(seed=(base_idx * 2654435761 + patch_slot * 40503) % (2**32))

        if self.patches_per_sample <= 1 or max_top <= 0 or max_left <= 0:
            top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
            left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
            return top, left

        n_lat = int(np.ceil(np.sqrt(self.patches_per_sample)))
        n_lon = int(np.ceil(self.patches_per_sample / n_lat))
        lat_band = patch_slot // n_lon
        lon_band = patch_slot % n_lon

        top_edges = np.linspace(0, max_top + 1, n_lat + 1)
        left_edges = np.linspace(0, max_left + 1, n_lon + 1)
        top_lo = int(np.floor(top_edges[lat_band]))
        top_hi = int(np.floor(top_edges[lat_band + 1])) - 1
        left_lo = int(np.floor(left_edges[lon_band]))
        left_hi = int(np.floor(left_edges[lon_band + 1])) - 1

        top_lo = max(0, min(max_top, top_lo))
        top_hi = max(top_lo, min(max_top, top_hi))
        left_lo = max(0, min(max_left, left_lo))
        left_hi = max(left_lo, min(max_left, left_hi))

        rng = np.random.default_rng(seed=(base_idx * 2654435761 + patch_slot * 40503) % (2**32))
        top = int(rng.integers(top_lo, top_hi + 1)) if top_hi > top_lo else top_lo
        left = int(rng.integers(left_lo, left_hi + 1)) if left_hi > left_lo else left_lo
        return top, left

    def _cache_key(self, start_idx: int, top: int, left: int) -> str:
        dtype_tag = "v3_xf16" if self._cache_is_f16 else "v3"
        return (
            f"s{start_idx}_t{top}_l{left}_ih{self.patch_h}_iw{self.patch_w}_"
            f"in{self.in_len}_out{self.out_len}_{self._norm_tag}_{dtype_tag}"
        )

    def _cache_paths(self, key: str) -> Tuple[str, str, str]:
        return (
            os.path.join(self.shm_cache_dir, f"{key}.x.npy"),
            os.path.join(self.shm_cache_dir, f"{key}.y.npy"),
            os.path.join(self.shm_cache_dir, f"{key}.m.npy"),
        )

    def _load_cache(self, key: str):
        x_path, y_path, m_path = self._cache_paths(key)
        if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(m_path):
            try:
                x = np.load(x_path, mmap_mode="c", allow_pickle=False)
                y = np.load(y_path, mmap_mode="c", allow_pickle=False)
                y_valid = np.load(m_path, mmap_mode="c", allow_pickle=False)
                return x, y, y_valid
            except (FileNotFoundError, OSError, ValueError):
                for path in (x_path, y_path, m_path):
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
        return None

    def _maybe_prune_cache(self, force: bool = False):
        if self.shm_cache_max_items <= 0:
            return
        self._cache_write_counter += 1
        if (not force) and self._cache_write_counter % 64 != 0:
            return
        x_files = [f for f in os.listdir(self.shm_cache_dir) if f.endswith(".x.npy")]
        if len(x_files) <= self.shm_cache_max_items:
            return
        def _safe_mtime(file_name: str) -> float:
            file_path = os.path.join(self.shm_cache_dir, file_name)
            try:
                return os.path.getmtime(file_path)
            except FileNotFoundError:
                return float("inf")

        x_files.sort(key=_safe_mtime)
        to_remove = x_files[: len(x_files) - self.shm_cache_max_items]
        for xf in to_remove:
            yf = xf.replace(".x.npy", ".y.npy")
            mf = xf.replace(".x.npy", ".m.npy")
            for fp in [
                os.path.join(self.shm_cache_dir, xf),
                os.path.join(self.shm_cache_dir, yf),
                os.path.join(self.shm_cache_dir, mf),
            ]:
                try:
                    os.remove(fp)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _cache_free_bytes(self) -> int:
        try:
            return int(shutil.disk_usage(self.shm_cache_dir).free)
        except OSError:
            return 0

    def _prune_cache_for_free_bytes(self, target_free_bytes: int):
        if target_free_bytes <= 0:
            return
        try:
            x_files = [f for f in os.listdir(self.shm_cache_dir) if f.endswith(".x.npy")]
        except OSError:
            return
        def _safe_mtime(file_name: str) -> float:
            file_path = os.path.join(self.shm_cache_dir, file_name)
            try:
                return os.path.getmtime(file_path)
            except FileNotFoundError:
                return float("inf")

        x_files.sort(key=_safe_mtime)
        for xf in x_files:
            if self._cache_free_bytes() >= target_free_bytes:
                break
            yf = xf.replace(".x.npy", ".y.npy")
            mf = xf.replace(".x.npy", ".m.npy")
            for fp in [
                os.path.join(self.shm_cache_dir, xf),
                os.path.join(self.shm_cache_dir, yf),
                os.path.join(self.shm_cache_dir, mf),
            ]:
                try:
                    os.remove(fp)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _write_cache(self, key: str, x: np.ndarray, y: np.ndarray, y_valid: np.ndarray):
        x_path, y_path, m_path = self._cache_paths(key)
        if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(m_path):
            return
        self._maybe_prune_cache(force=True)

        x_save = x
        if self._cache_is_f16:
            np.clip(x, -65504.0, 65504.0, out=x)
            x_save = x.astype(np.float16)

        x_bytes = x_save.nbytes
        expected_bytes = int(x_bytes + y.nbytes + y_valid.nbytes + 8 * 1024 * 1024)
        free_bytes = self._cache_free_bytes()
        required_free = self.shm_cache_min_free_bytes + expected_bytes
        if free_bytes < required_free:
            self._prune_cache_for_free_bytes(required_free)
            free_bytes = self._cache_free_bytes()
            if free_bytes < required_free:
                return

        tmp_x = f"{x_path}.tmp.{os.getpid()}"
        tmp_y = f"{y_path}.tmp.{os.getpid()}"
        tmp_m = f"{m_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_x, "wb") as fx:
                np.save(fx, x_save, allow_pickle=False)
            with open(tmp_y, "wb") as fy:
                np.save(fy, y, allow_pickle=False)
            with open(tmp_m, "wb") as fm:
                np.save(fm, y_valid, allow_pickle=False)
            if not os.path.exists(x_path):
                os.replace(tmp_x, x_path)
            else:
                os.remove(tmp_x)
            if not os.path.exists(y_path):
                os.replace(tmp_y, y_path)
            else:
                os.remove(tmp_y)
            if not os.path.exists(m_path):
                os.replace(tmp_m, m_path)
            else:
                os.remove(tmp_m)
            self._maybe_prune_cache(force=True)
        except OSError:
            for fp in [tmp_x, tmp_y, tmp_m]:
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
            self._maybe_prune_cache(force=True)

    def __len__(self):
        return len(self.start_indices) * self.patches_per_sample

    def __getitem__(self, idx: int):
        base_idx = int(idx // self.patches_per_sample)
        patch_slot = int(idx % self.patches_per_sample)
        s = int(self.start_indices[base_idx])
        condition_id = int(self.condition_ids[base_idx])
        h, w = self.core.height, self.core.width
        top, left = 0, 0
        key = None
        if self.patch_h <= h and self.patch_w <= w:
            if self.random_patch:
                if self.shm_cache_enabled:
                    top, left = self._stratified_patch_position(base_idx, patch_slot, h, w)
                else:
                    if self.patches_per_sample > 1:
                        top, left = self._stratified_patch_position(base_idx, patch_slot, h, w)
                    else:
                        top = np.random.randint(0, h - self.patch_h + 1)
                        left = np.random.randint(0, w - self.patch_w + 1)
            else:
                top = (h - self.patch_h) // 2
                left = (w - self.patch_w) // 2

        if self.shm_cache_enabled:
            key = self._cache_key(s, top, left)
            cached = self._load_cache(key)
            if cached is not None:
                x, y, y_valid = cached
                return {
                    "x": torch.from_numpy(x),
                    "y": torch.from_numpy(y),
                    "y_valid": torch.from_numpy(y_valid),
                    "condition_id": torch.tensor(condition_id, dtype=torch.long),
                }

        lat_slice = slice(top, top + self.patch_h)
        lon_slice = slice(left, left + self.patch_w)

        x, y = self.core.get_window(
            s,
            self.in_len,
            self.out_len,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
        )
        y_valid = np.isfinite(y) & (y > 0)
        x = self.normalizer.transform_x(x)
        y = self.normalizer.transform_y(y)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if self.shm_cache_enabled:
            if key is not None:
                self._write_cache(key, x, y, y_valid.astype(np.bool_))

        x_t = torch.from_numpy(x)
        if self._cache_is_f16:
            x_t = x_t.half()

        return {
            "x": x_t,
            "y": torch.from_numpy(y),
            "y_valid": torch.from_numpy(y_valid),
            "condition_id": torch.tensor(condition_id, dtype=torch.long),
        }


class PM25DataModule(pl.LightningDataModule):
    def __init__(
        self,
        core: MultiSourcePM25Core,
        in_len: int,
        out_len: int,
        train_target_start: str,
        train_target_end: str,
        val_target_start: str,
        val_target_end: str,
        batch_size: int,
        num_workers: int,
        patch_h: int,
        patch_w: int,
        normalizer: PM25Normalizer,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        shm_cache_enabled: bool = False,
        shm_cache_dir: str = "/dev/shm/pm25_window_cache",
        shm_cache_max_items: int = 0,
        shm_cache_min_free_gb: float = 2.0,
        shm_cache_x_dtype: str = "float32",
        condition_scheme: str = "season4",
        train_patches_per_sample: int = 1,
        persistent_workers: bool = False,
        cuda_prefetch: bool = False,
        dataloader_multiprocessing_context: str = "",
    ):
        super().__init__()
        if dataloader_multiprocessing_context not in ("", "fork", "spawn", "forkserver"):
            raise ValueError(
                f"Invalid dataloader_multiprocessing_context: '{dataloader_multiprocessing_context}'. "
                f"Allowed values are '', 'fork', 'spawn', 'forkserver'."
            )
        self.dataloader_multiprocessing_context = dataloader_multiprocessing_context
        if (
            self.dataloader_multiprocessing_context in ("spawn", "forkserver")
            and num_workers > 0
            and getattr(core, "_mem_data", None) is not None
        ):
            raise ValueError(
                f"dataloader_multiprocessing_context='{self.dataloader_multiprocessing_context}' with num_workers > 0 is incompatible with preload_to_memory=True "
                "due to massive memory amplification risk. Please either set preload_to_memory to false, "
                "set num_workers to 0, or change the multiprocessing context."
            )
        self.core = core
        self.in_len = in_len
        self.out_len = out_len
        self.train_target_start = train_target_start
        self.train_target_end = train_target_end
        self.val_target_start = val_target_start
        self.val_target_end = val_target_end
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.normalizer = normalizer
        self.prefetch_factor = max(1, int(prefetch_factor))
        self.pin_memory = pin_memory
        self.shm_cache_enabled = shm_cache_enabled
        self.shm_cache_dir = shm_cache_dir
        self.shm_cache_max_items = max(0, int(shm_cache_max_items))
        self.shm_cache_min_free_gb = max(0.0, float(shm_cache_min_free_gb))
        self.shm_cache_x_dtype = shm_cache_x_dtype
        self.condition_scheme = condition_scheme
        self.train_patches_per_sample = max(1, int(train_patches_per_sample))
        self.persistent_workers = bool(persistent_workers)
        self.cuda_prefetch = bool(cuda_prefetch)
        self._prefetch_active = self.cuda_prefetch and self.pin_memory and torch.cuda.is_available()
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

    def _resolve_dataloader_mp_context(self):
        """Resolve the multiprocessing context used to build DataLoaders.

        - ``num_workers == 0``: return ``None`` so no worker processes are
          created and the previous behavior is preserved exactly.
        - ``num_workers > 0``: honor an explicitly configured context; when
          none is configured, default to ``"spawn"`` so workers do not inherit
          the parent's Zarr/PyTorch thread state through ``fork`` (which can
          futex-deadlock in the rolling pipeline). The single exception is when
          the core has preloaded all data into RAM: ``spawn`` is disallowed
          there (memory amplification), so the platform default is kept.
        """
        if self.num_workers <= 0:
            return None
        import multiprocessing

        context_name = self.dataloader_multiprocessing_context
        if not context_name:
            if getattr(self.core, "_mem_data", None) is not None:
                return None
            context_name = "spawn"
        return multiprocessing.get_context(context_name)

    def setup(self, stage=None):
        tr_idx = self.core.find_valid_start_indices(
            target_start=self.train_target_start,
            target_end=self.train_target_end,
            in_len=self.in_len,
            out_len=self.out_len,
        )
        va_idx = self.core.find_valid_start_indices(
            target_start=self.val_target_start,
            target_end=self.val_target_end,
            in_len=self.in_len,
            out_len=self.out_len,
        )

        self.train_set = PM25WindowDataset(
            core=self.core,
            start_indices=tr_idx,
            in_len=self.in_len,
            out_len=self.out_len,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            random_patch=True,
            normalizer=self.normalizer,
            shm_cache_enabled=self.shm_cache_enabled,
            shm_cache_dir=self.shm_cache_dir,
            shm_cache_max_items=self.shm_cache_max_items,
            shm_cache_min_free_gb=self.shm_cache_min_free_gb,
            shm_cache_x_dtype=self.shm_cache_x_dtype,
            condition_scheme=self.condition_scheme,
            patches_per_sample=self.train_patches_per_sample,
        )
        self.val_set = PM25WindowDataset(
            core=self.core,
            start_indices=va_idx,
            in_len=self.in_len,
            out_len=self.out_len,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            random_patch=False,
            normalizer=self.normalizer,
            shm_cache_enabled=self.shm_cache_enabled,
            shm_cache_dir=self.shm_cache_dir,
            shm_cache_max_items=self.shm_cache_max_items,
            shm_cache_min_free_gb=self.shm_cache_min_free_gb,
            shm_cache_x_dtype=self.shm_cache_x_dtype,
            condition_scheme=self.condition_scheme,
        )

    def train_dataloader(self):
        persistent_workers = self.num_workers > 0 and self.persistent_workers
        mp_context = self._resolve_dataloader_mp_context()
        loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context=mp_context,
        )
        prefetch_device = self._resolve_prefetch_device()
        if prefetch_device is not None:
            self._train_loader = _CudaPrefetcher(loader, prefetch_device)
        else:
            self._train_loader = loader
        return self._train_loader

    def val_dataloader(self):
        persistent_workers = self.num_workers > 0 and self.persistent_workers
        mp_context = self._resolve_dataloader_mp_context()
        loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context=mp_context,
        )
        prefetch_device = self._resolve_prefetch_device()
        if prefetch_device is not None:
            self._val_loader = _CudaPrefetcher(loader, prefetch_device)
        else:
            self._val_loader = loader
        return self._val_loader

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self._prefetch_active and _batch_is_already_on(batch, device):
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _resolve_prefetch_device(self) -> torch.device | None:
        if not self._prefetch_active:
            return None
        trainer = getattr(self, "trainer", None)
        strategy = getattr(trainer, "strategy", None)
        device = getattr(strategy, "root_device", None)
        if device is None:
            device = getattr(trainer, "root_device", None)
        if device is None:
            return None
        device = torch.device(device)
        if device.type != "cuda":
            return None
        return device


    def shutdown_workers(self):
        for loader in (self._train_loader, self._val_loader):
            if loader is None:
                continue
            if isinstance(loader, _CudaPrefetcher):
                loader = loader.loader
            try:
                iterator = getattr(loader, "_iterator", None)
                if iterator is not None:
                    iterator._shutdown_workers()
                    loader._iterator = None
            except Exception:
                pass
        self._train_loader = None
        self._val_loader = None


def _batch_is_already_on(batch, device):
    device = torch.device(device)
    if isinstance(batch, dict):
        return all(_batch_is_already_on(v, device) for v in batch.values())
    if isinstance(batch, (list, tuple)):
        return all(_batch_is_already_on(v, device) for v in batch)
    if isinstance(batch, torch.Tensor):
        return batch.device == device
    return True


class _CudaPrefetcher:
    """Async H2D prefetch wrapper for single-GPU DataLoader.

    Uses a separate CUDA stream to transfer the next batch to GPU
    while the current batch is being processed by the model.
    """

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"_CudaPrefetcher requires a CUDA device, got {self.device}")
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        preloaded = None
        preloaded_event = None
        for batch in self.loader:
            # Transfer next batch on prefetch stream and record completion event.
            with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                next_batch = _move_batch_to_device(batch, self.device)
                next_event = torch.cuda.Event()
                next_event.record(self._stream)

            # Wait for previously preloaded batch, then yield it.
            if preloaded is not None:
                current_stream = torch.cuda.current_stream(self.device)
                current_stream.wait_event(preloaded_event)
                _record_batch_stream(preloaded, current_stream)
                yield preloaded

            # Now enqueue next batch H2D while current batch compute runs.
            preloaded = next_batch
            preloaded_event = next_event

        # Final batch with same lifecycle protection.
        if preloaded is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_event(preloaded_event)
            _record_batch_stream(preloaded, current_stream)
            yield preloaded


def _move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(v, device) for v in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def _record_batch_stream(batch, stream):
    if isinstance(batch, dict):
        for v in batch.values():
            _record_batch_stream(v, stream)
    elif isinstance(batch, (list, tuple)):
        for v in batch:
            _record_batch_stream(v, stream)
    elif isinstance(batch, torch.Tensor) and batch.is_cuda:
        batch.record_stream(stream)
