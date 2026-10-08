import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from .data import MultiSourcePM25Core


_EPS = 1e-6
# Number of source/target dates read per streaming reduction chunk. Keeps the
# peak memory of normalization fitting bounded regardless of the fit range.
_FIT_CHUNK_DAYS = 32


def _iter_chunks(indices: np.ndarray, chunk_size: int):
    for i in range(0, len(indices), chunk_size):
        yield indices[i : i + chunk_size]


def _streaming_minmax(get_chunk, indices: np.ndarray, chunk_size: int = _FIT_CHUNK_DAYS):
    """Streaming min/max over finite values, reading ``indices`` in chunks.

    Non-finite values (NaN/inf) are ignored. When no finite value exists the
    neutral (0.0, 1.0) range is returned so the transform degenerates to the
    identity instead of producing NaN statistics.
    """
    mn = None
    mx = None
    for chunk in _iter_chunks(indices, chunk_size):
        arr = get_chunk(chunk)
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        vals = arr[finite]
        cmin = float(vals.min())
        cmax = float(vals.max())
        mn = cmin if mn is None else min(mn, cmin)
        mx = cmax if mx is None else max(mx, cmax)
    if mn is None:
        return 0.0, 1.0
    return mn, mx


def _streaming_mean_std(get_chunk, indices: np.ndarray, chunk_size: int = _FIT_CHUNK_DAYS):
    """Streaming mean/std (ddof=0) over finite values, reading in chunks.

    Matches ``np.nanmean``/``np.nanstd`` semantics for finite data while
    ignoring non-finite values. Accumulation is done in float64. When no finite
    value exists the neutral (0.0, 1.0) statistics are returned.
    """
    count = 0
    total = 0.0
    total_sq = 0.0
    for chunk in _iter_chunks(indices, chunk_size):
        arr = get_chunk(chunk)
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        vals = arr[finite].astype(np.float64, copy=False)
        count += int(vals.size)
        total += float(vals.sum())
        total_sq += float(np.square(vals).sum())
    if count == 0:
        return 0.0, 1.0
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    std = math.sqrt(var)
    return mean, max(std, _EPS)


@dataclass
class NormalizationModes:
    x_mode: str
    y_mode: str


def default_modes_for_model(model_name: str) -> NormalizationModes:
    name = model_name.lower()
    if name == "earthformer":
        return NormalizationModes(x_mode="zscore", y_mode="zscore")
    if name in ["iam4vp", "phydnet", "unet", "convlstm"]:
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
        # Target-date range the statistics were fitted on (provenance only;
        # not part of the fingerprint so existing banks stay compatible).
        self.fit_target_start = None
        self.fit_target_end = None
        # Input-schema provenance. ``input_channels`` / ``feature_names`` describe
        # the exact feature-plane layout the statistics were fitted for, so a
        # legacy 41-channel bank can never be silently reused once known-future
        # GFS channels are enabled.
        self.input_channels = None
        self.feature_names = None
        self.future_gfs_enabled = False
        self.future_gfs_steps = 0

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

        self.fit_target_start = str(train_target_start)
        self.fit_target_end = str(train_target_end)

        # Known-future GFS schema (absent on legacy cores -> disabled).
        future_enabled = bool(getattr(core, "future_gfs_enabled", False))
        future_steps = int(getattr(core, "future_gfs_steps", 0)) if future_enabled else 0

        if self.x_mode != "none":
            x_stats_1 = []
            x_stats_2 = []
            gfs_future_stats = {}
            gfs_var_order = []
            for name, ds, vars_ in core.source_specs:
                if future_enabled and name == "gfs":
                    # GFS statistics are fitted once per variable over the union
                    # of the historical input dates and the known-future target
                    # dates covered by the 2017-2018 fit samples. The per-lead
                    # channels then reuse the same statistics.
                    src_idx = np.unique(
                        np.concatenate(
                            [
                                core.time_index_map[name][in_days],
                                core.time_index_map[name][out_days],
                            ]
                        )
                    )
                else:
                    src_idx = np.unique(core.time_index_map[name][in_days])
                for v in vars_:
                    def _get_x(chunk, _name=name, _var=v):
                        return core.get_feature_days(_name, _var, chunk)

                    if self.x_mode == "zscore":
                        stat_1, stat_2 = _streaming_mean_std(_get_x, src_idx)
                    elif self.x_mode in ["minmax_01", "minmax_m11"]:
                        mn, mx = _streaming_minmax(_get_x, src_idx)
                        stat_1, stat_2 = mn, max(mx - mn, _EPS)
                    else:
                        raise ValueError(f"Unsupported x_mode={self.x_mode}")
                    x_stats_1.append(stat_1)
                    x_stats_2.append(stat_2)
                    if future_enabled and name == "gfs":
                        gfs_future_stats[v] = (stat_1, stat_2)
                        gfs_var_order.append(v)
            if future_enabled:
                # Lead-major known-future channels, each reusing its GFS
                # variable's statistics (identical across leads). Iterate the
                # core's authoritative GFS variable order so the statistics line
                # up with ``core.input_feature_names``.
                future_var_order = list(getattr(core, "gfs_vars", gfs_var_order))
                for _lead in range(future_steps):
                    for v in future_var_order:
                        stat_1, stat_2 = gfs_future_stats[v]
                        x_stats_1.append(stat_1)
                        x_stats_2.append(stat_2)
            self.x_p1 = np.array(x_stats_1, dtype=np.float32)
            self.x_p2 = np.array(x_stats_2, dtype=np.float32)

        self.input_channels = int(
            getattr(core, "input_channels", len(self.x_p1) if self.x_p1 is not None else 0)
        )
        self.future_gfs_enabled = future_enabled
        self.future_gfs_steps = future_steps
        names = getattr(core, "input_feature_names", None)
        self.feature_names = list(names) if names else None

        if self.y_mode != "none":
            def _get_y(chunk):
                return core.get_pm25_days(chunk)

            if self.y_mode == "zscore":
                mean, std = _streaming_mean_std(_get_y, out_days)
                self.y_p1 = np.array([mean], dtype=np.float32)
                self.y_p2 = np.array([std], dtype=np.float32)
            elif self.y_mode in ["minmax_01", "minmax_m11"]:
                mn, mx = _streaming_minmax(_get_y, out_days)
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

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"x_mode={self.x_mode}\ny_mode={self.y_mode}\n".encode("utf-8"))
        # Input-schema provenance is part of the fingerprint so a bank fitted
        # for a different channel layout (e.g. legacy 41 channels vs. 56 with
        # known-future GFS) can never collide.
        h.update(
            (
                f"input_channels={self.input_channels}\n"
                f"future_gfs_enabled={bool(self.future_gfs_enabled)}\n"
                f"future_gfs_steps={int(self.future_gfs_steps)}\n"
            ).encode("utf-8")
        )
        if self.feature_names:
            h.update(("feature_names=" + ",".join(self.feature_names) + "\n").encode("utf-8"))
        for name, arr in [
            ("x_p1", self.x_p1),
            ("x_p2", self.x_p2),
            ("y_p1", self.y_p1),
            ("y_p2", self.y_p2),
        ]:
            h.update(name.encode("utf-8"))
            if arr is None:
                h.update(b":none\n")
                continue
            value = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
            h.update(str(value.shape).encode("utf-8"))
            h.update(str(value.dtype).encode("utf-8"))
            h.update(value.tobytes(order="C"))
        return h.hexdigest()

    def validate_against_core(self, core) -> tuple:
        """Check these statistics match the core's input schema.

        Guards against silently reusing a bank fitted for a different input
        channel layout (e.g. legacy 41-channel stats with known-future GFS
        inputs enabled). Returns ``(ok, reason)``.
        """
        expected_channels = int(getattr(core, "input_channels", -1))
        if self.x_mode != "none":
            if self.x_p1 is None or self.x_p2 is None:
                return False, "x statistics are missing"
            if expected_channels >= 0 and (
                len(self.x_p1) != expected_channels or len(self.x_p2) != expected_channels
            ):
                return False, (
                    f"x statistics have {len(self.x_p1)}/{len(self.x_p2)} channels but core expects "
                    f"{expected_channels}"
                )
        core_future = bool(getattr(core, "future_gfs_enabled", False))
        core_steps = int(getattr(core, "future_gfs_steps", 0)) if core_future else 0
        if core_future:
            if not self.future_gfs_enabled:
                return False, (
                    "core enables known-future GFS inputs but statistics have no "
                    "future-GFS schema"
                )
            if int(self.future_gfs_steps) != core_steps:
                return False, (
                    f"future_gfs_steps mismatch: statistics={self.future_gfs_steps} "
                    f"core={core_steps}"
                )
        elif self.future_gfs_enabled:
            return False, (
                "statistics were fitted with known-future GFS inputs but core has "
                "them disabled"
            )
        expected_names = getattr(core, "input_feature_names", None)
        if self.feature_names and expected_names and self.feature_names != list(expected_names):
            return False, "input feature names/order differ from the fitted statistics"
        if core_future and expected_names and not self.feature_names:
            return False, "known-future GFS statistics are missing feature names/order"
        return True, ""

    def save(self, path: str):
        np.savez_compressed(
            path,
            x_mode=np.array([self.x_mode]),
            y_mode=np.array([self.y_mode]),
            x_p1=np.array([] if self.x_p1 is None else self.x_p1, dtype=np.float32),
            x_p2=np.array([] if self.x_p2 is None else self.x_p2, dtype=np.float32),
            y_p1=np.array([] if self.y_p1 is None else self.y_p1, dtype=np.float32),
            y_p2=np.array([] if self.y_p2 is None else self.y_p2, dtype=np.float32),
            fit_target_start=np.array([self.fit_target_start or ""]),
            fit_target_end=np.array([self.fit_target_end or ""]),
            input_channels=np.array(
                [-1 if self.input_channels is None else int(self.input_channels)],
                dtype=np.int64,
            ),
            future_gfs_enabled=np.array([bool(self.future_gfs_enabled)]),
            future_gfs_steps=np.array([int(self.future_gfs_steps)], dtype=np.int64),
            feature_names=np.array(
                ["\n".join(self.feature_names) if self.feature_names else ""]
            ),
            fingerprint=np.array([self.fingerprint()]),
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
        # Fit-range and input-schema provenance are optional: older npz files
        # do not contain them, in which case the schema stays unknown/legacy.
        files = set(obj.files)
        if "fit_target_start" in files:
            value = str(obj["fit_target_start"][0])
            normalizer.fit_target_start = value or None
        if "fit_target_end" in files:
            value = str(obj["fit_target_end"][0])
            normalizer.fit_target_end = value or None
        if "input_channels" in files:
            value = int(obj["input_channels"][0])
            normalizer.input_channels = value if value >= 0 else None
        if "future_gfs_enabled" in files:
            normalizer.future_gfs_enabled = bool(obj["future_gfs_enabled"][0])
        if "future_gfs_steps" in files:
            normalizer.future_gfs_steps = int(obj["future_gfs_steps"][0])
        if "feature_names" in files:
            raw = str(obj["feature_names"][0])
            normalizer.feature_names = raw.split("\n") if raw else None
        return normalizer
