from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from .data import MultiSourcePM25Core


_EPS = 1e-6


@dataclass
class NormalizationModes:
    x_mode: str
    y_mode: str


def default_modes_for_model(model_name: str) -> NormalizationModes:
    name = model_name.lower()
    if name == "earthformer":
        return NormalizationModes(x_mode="zscore", y_mode="zscore")
    if name in ["iam4vp", "phydnet"]:
        return NormalizationModes(x_mode="minmax_01", y_mode="minmax_01")
    return NormalizationModes(x_mode="none", y_mode="none")


class PM25Normalizer:
    def __init__(self, x_mode: str, y_mode: str):
        self.x_mode = x_mode
        self.y_mode = y_mode
        self.x_p1 = None
        self.x_p2 = None
        self.y_p1 = None
        self.y_p2 = None

    @property
    def fitted(self) -> bool:
        if self.x_mode == "none" and self.y_mode == "none":
            return True
        if self.x_mode != "none" and (self.x_p1 is None or self.x_p2 is None):
            return False
        if self.y_mode != "none" and (self.y_p1 is None or self.y_p2 is None):
            return False
        return True

    def fit(
        self,
        core: "MultiSourcePM25Core",
        in_len: int,
        out_len: int,
        train_target_start: str,
        train_target_end: str,
    ):
        starts = core.find_valid_start_indices(
            target_start=train_target_start,
            target_end=train_target_end,
            in_len=in_len,
            out_len=out_len,
        )
        if len(starts) == 0:
            raise ValueError("No training windows found for normalization fitting.")

        in_days = []
        out_days = []
        for s in starts:
            in_days.append(np.arange(s, s + in_len, dtype=np.int64))
            out_days.append(np.arange(s + in_len, s + in_len + out_len, dtype=np.int64))
        in_days = np.unique(np.concatenate(in_days))
        out_days = np.unique(np.concatenate(out_days))

        if self.x_mode != "none":
            x_stats_1 = []
            x_stats_2 = []
            for name, ds, vars_ in core.source_specs:
                src_idx = np.unique(core.time_index_map[name][in_days])
                src_idx_da = xr.DataArray(src_idx, dims="sample")
                for v in vars_:
                    da = ds[v].isel(time=src_idx_da)
                    if self.x_mode == "zscore":
                        m = float(da.mean().values)
                        s = float(da.std().values)
                        x_stats_1.append(m)
                        x_stats_2.append(max(s, _EPS))
                    elif self.x_mode in ["minmax_01", "minmax_m11"]:
                        mn = float(da.min().values)
                        mx = float(da.max().values)
                        x_stats_1.append(mn)
                        x_stats_2.append(max(mx - mn, _EPS))
                    else:
                        raise ValueError(f"Unsupported x_mode={self.x_mode}")
            self.x_p1 = np.array(x_stats_1, dtype=np.float32)
            self.x_p2 = np.array(x_stats_2, dtype=np.float32)

        if self.y_mode != "none":
            out_days_da = xr.DataArray(out_days, dims="sample")
            y_da = core.pm25_ds[core.pm25_var].isel(time=out_days_da)
            if self.y_mode == "zscore":
                self.y_p1 = np.array([float(y_da.mean().values)], dtype=np.float32)
                self.y_p2 = np.array([max(float(y_da.std().values), _EPS)], dtype=np.float32)
            elif self.y_mode in ["minmax_01", "minmax_m11"]:
                mn = float(y_da.min().values)
                mx = float(y_da.max().values)
                self.y_p1 = np.array([mn], dtype=np.float32)
                self.y_p2 = np.array([max(mx - mn, _EPS)], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported y_mode={self.y_mode}")

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        if self.x_mode == "none":
            return x
        p1 = self.x_p1.reshape(1, 1, 1, -1)
        p2 = self.x_p2.reshape(1, 1, 1, -1)
        if self.x_mode == "zscore":
            return (x - p1) / p2
        if self.x_mode == "minmax_01":
            return (x - p1) / p2
        if self.x_mode == "minmax_m11":
            return 2.0 * ((x - p1) / p2) - 1.0
        raise ValueError(f"Unsupported x_mode={self.x_mode}")

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.y_mode == "none":
            return y
        p1 = self.y_p1.reshape(1, 1, 1, -1)
        p2 = self.y_p2.reshape(1, 1, 1, -1)
        if self.y_mode == "zscore":
            return (y - p1) / p2
        if self.y_mode == "minmax_01":
            return (y - p1) / p2
        if self.y_mode == "minmax_m11":
            return 2.0 * ((y - p1) / p2) - 1.0
        raise ValueError(f"Unsupported y_mode={self.y_mode}")

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        if self.y_mode == "none":
            return y_norm
        p1 = self.y_p1.reshape(1, 1, 1)
        p2 = self.y_p2.reshape(1, 1, 1)
        if self.y_mode == "zscore":
            return y_norm * p2 + p1
        if self.y_mode == "minmax_01":
            return y_norm * p2 + p1
        if self.y_mode == "minmax_m11":
            return ((y_norm + 1.0) * 0.5) * p2 + p1
        raise ValueError(f"Unsupported y_mode={self.y_mode}")

    def save(self, path: str):
        np.savez_compressed(
            path,
            x_mode=np.array([self.x_mode]),
            y_mode=np.array([self.y_mode]),
            x_p1=np.array([] if self.x_p1 is None else self.x_p1, dtype=np.float32),
            x_p2=np.array([] if self.x_p2 is None else self.x_p2, dtype=np.float32),
            y_p1=np.array([] if self.y_p1 is None else self.y_p1, dtype=np.float32),
            y_p2=np.array([] if self.y_p2 is None else self.y_p2, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str):
        obj = np.load(path, allow_pickle=False)
        x_mode = str(obj["x_mode"][0])
        y_mode = str(obj["y_mode"][0])
        normalizer = cls(x_mode=x_mode, y_mode=y_mode)
        x_p1 = obj["x_p1"]
        x_p2 = obj["x_p2"]
        y_p1 = obj["y_p1"]
        y_p2 = obj["y_p2"]
        normalizer.x_p1 = x_p1 if x_p1.size > 0 else None
        normalizer.x_p2 = x_p2 if x_p2.size > 0 else None
        normalizer.y_p1 = y_p1 if y_p1.size > 0 else None
        normalizer.y_p2 = y_p2 if y_p2.size > 0 else None
        return normalizer
