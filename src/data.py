import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .normalization import PM25Normalizer


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

        self.source_specs = [
            ("year", self.year_ds, list(self.year_ds.data_vars)),
            ("eral", self.eral_ds, list(self.eral_ds.data_vars)),
            ("erap", self.erap_ds, list(self.erap_ds.data_vars)),
            ("gfs", self.gfs_ds, list(self.gfs_ds.data_vars)),
            ("meic", self.meic_ds, list(self.meic_ds.data_vars)),
        ]

        self.time_index_map: Dict[str, np.ndarray] = {
            name: build_time_mapper(ds.time.values, self.pm25_time.values)
            for name, ds, _ in self.source_specs
        }

        self.input_channels = int(sum(len(vars_) for _, _, vars_ in self.source_specs))

    @staticmethod
    def _read_feature_arr(ds, var_name: str, map_idx: np.ndarray, lat_slice: slice, lon_slice: slice) -> np.ndarray:
        return ds[var_name].isel(
            time=xr.DataArray(map_idx, dims="sample"),
            lat=lat_slice,
            lon=lon_slice,
        ).values.astype(np.float32)

    def get_window(
        self,
        start_idx: int,
        in_len: int,
        out_len: int,
        lat_slice: slice | None = None,
        lon_slice: slice | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        in_idx = np.arange(start_idx, start_idx + in_len, dtype=np.int64)
        out_idx = np.arange(start_idx + in_len, start_idx + in_len + out_len, dtype=np.int64)

        lat_slice = lat_slice if lat_slice is not None else slice(None)
        lon_slice = lon_slice if lon_slice is not None else slice(None)

        tasks = []
        for name, ds, vars_ in self.source_specs:
            map_idx = self.time_index_map[name][in_idx]
            for v in vars_:
                tasks.append((ds, v, map_idx.copy(), lat_slice, lon_slice))

        if self.zarr_parallel_backend == "thread" and self.zarr_read_workers > 1:
            max_workers = min(self.zarr_read_workers, len(tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                feats = list(ex.map(lambda t: self._read_feature_arr(*t), tasks))
        else:
            feats = [self._read_feature_arr(*t) for t in tasks]

        x = np.stack(feats, axis=-1)
        y = self.pm25_ds[self.pm25_var].isel(
            time=xr.DataArray(out_idx, dims="sample"),
            lat=lat_slice,
            lon=lon_slice,
        ).values.astype(np.float32)
        y = y[..., None]
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
        self._cache_write_counter = 0

        if self.shm_cache_enabled:
            os.makedirs(self.shm_cache_dir, exist_ok=True)

        norm_parts = [self.normalizer.x_mode, self.normalizer.y_mode]
        for arr in [self.normalizer.x_p1, self.normalizer.x_p2, self.normalizer.y_p1, self.normalizer.y_p2]:
            if arr is not None:
                norm_parts.append(str(arr.shape))
                norm_parts.append(f"{float(np.mean(arr)):.6f}")
        self._norm_tag = hashlib.md5("|".join(norm_parts).encode("utf-8")).hexdigest()[:12]

    def _cache_key(self, start_idx: int, top: int, left: int) -> str:
        return f"s{start_idx}_t{top}_l{left}_ih{self.patch_h}_iw{self.patch_w}_in{self.in_len}_out{self.out_len}_{self._norm_tag}"

    def _cache_paths(self, key: str) -> Tuple[str, str]:
        return (
            os.path.join(self.shm_cache_dir, f"{key}.x.npy"),
            os.path.join(self.shm_cache_dir, f"{key}.y.npy"),
        )

    def _load_cache(self, key: str):
        x_path, y_path = self._cache_paths(key)
        if os.path.exists(x_path) and os.path.exists(y_path):
            x = np.load(x_path, allow_pickle=False)
            y = np.load(y_path, allow_pickle=False)
            return x, y
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
        x_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.shm_cache_dir, f)))
        to_remove = x_files[: len(x_files) - self.shm_cache_max_items]
        for xf in to_remove:
            yf = xf.replace(".x.npy", ".y.npy")
            for fp in [os.path.join(self.shm_cache_dir, xf), os.path.join(self.shm_cache_dir, yf)]:
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
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
        x_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.shm_cache_dir, f)))
        for xf in x_files:
            if self._cache_free_bytes() >= target_free_bytes:
                break
            yf = xf.replace(".x.npy", ".y.npy")
            for fp in [os.path.join(self.shm_cache_dir, xf), os.path.join(self.shm_cache_dir, yf)]:
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass

    def _write_cache(self, key: str, x: np.ndarray, y: np.ndarray):
        x_path, y_path = self._cache_paths(key)
        if os.path.exists(x_path) and os.path.exists(y_path):
            return
        self._maybe_prune_cache(force=True)

        expected_bytes = int(x.nbytes + y.nbytes + 8 * 1024 * 1024)
        free_bytes = self._cache_free_bytes()
        # Keep a free-space safety margin while making room for this write.
        required_free = self.shm_cache_min_free_bytes + expected_bytes
        if free_bytes < required_free:
            self._prune_cache_for_free_bytes(required_free)
            free_bytes = self._cache_free_bytes()
            if free_bytes < required_free:
                return

        tmp_x = f"{x_path}.tmp.{os.getpid()}"
        tmp_y = f"{y_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_x, "wb") as fx:
                np.save(fx, x, allow_pickle=False)
            with open(tmp_y, "wb") as fy:
                np.save(fy, y, allow_pickle=False)
            if not os.path.exists(x_path):
                os.replace(tmp_x, x_path)
            else:
                os.remove(tmp_x)
            if not os.path.exists(y_path):
                os.replace(tmp_y, y_path)
            else:
                os.remove(tmp_y)
            self._maybe_prune_cache(force=True)
        except OSError:
            # Gracefully degrade when shared memory gets tight: skip caching this sample.
            for fp in [tmp_x, tmp_y]:
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
            self._maybe_prune_cache(force=True)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        s = int(self.start_indices[idx])
        h, w = self.core.height, self.core.width
        top, left = 0, 0
        key = None
        if self.patch_h <= h and self.patch_w <= w:
            if self.random_patch:
                if self.shm_cache_enabled:
                    rng = np.random.default_rng(seed=(idx * 2654435761) % (2**32))
                    top = int(rng.integers(0, h - self.patch_h + 1))
                    left = int(rng.integers(0, w - self.patch_w + 1))
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
                x, y = cached
                return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}

        lat_slice = slice(top, top + self.patch_h)
        lon_slice = slice(left, left + self.patch_w)
        x, y = self.core.get_window(
            s,
            self.in_len,
            self.out_len,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
        )

        x = self.normalizer.transform_x(x)
        y = self.normalizer.transform_y(y)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if self.shm_cache_enabled:
            if key is not None:
                self._write_cache(key, x, y)

        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}


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
    ):
        super().__init__()
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
        )

    def train_dataloader(self):
        persistent_workers = self.num_workers > 0
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        persistent_workers = self.num_workers > 0
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
