from __future__ import annotations

import numpy as np
import pandas as pd


_SEASON4_NAMES = ["DJF", "MAM", "JJA", "SON"]
_MONTH12_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
_MONTH_TO_SEASON4_ID = {
    12: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
    11: 3,
}


def condition_names_for_scheme(condition_scheme: str) -> list[str]:
    scheme = condition_scheme.lower()
    if scheme in ("global", "single"):
        return ["global"]
    if scheme == "season4":
        return list(_SEASON4_NAMES)
    if scheme == "month12":
        return list(_MONTH12_NAMES)
    raise ValueError(f"Unsupported condition_scheme={condition_scheme}. Supported: season4, month12, global")


def _month_to_condition_id(month: int, condition_scheme: str) -> int:
    scheme = condition_scheme.lower()
    if scheme in ("global", "single"):
        return 0
    if scheme == "season4":
        if month not in _MONTH_TO_SEASON4_ID:
            raise ValueError(f"Invalid month={month}")
        return int(_MONTH_TO_SEASON4_ID[month])
    if scheme == "month12":
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month={month}")
        return int(month - 1)
    raise ValueError(f"Unsupported condition_scheme={condition_scheme}. Supported: season4, month12, global")


def condition_ids_for_start_indices(
    pm25_time: pd.DatetimeIndex,
    start_indices: np.ndarray,
    in_len: int,
    condition_scheme: str = "season4",
) -> np.ndarray:
    if start_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)

    target_indices = start_indices.astype(np.int64) + int(in_len)
    if np.any(target_indices < 0) or np.any(target_indices >= len(pm25_time)):
        raise ValueError("Target indices out of bounds when mapping conditions.")

    target_days = pm25_time[target_indices]
    out = np.empty((len(target_days),), dtype=np.int64)
    for i, day in enumerate(target_days):
        out[i] = _month_to_condition_id(int(day.month), condition_scheme=condition_scheme)
    return out
