import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
import pytorch_lightning as pl

from .data import MultiSourcePM25Core
from .data import PM25DataModule
from .lightning_module import PM25ForecastLitModule
from .normalization import PM25Normalizer


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tile_slices(length: int, tile: int, stride: int) -> List[slice]:
    out = []
    pos = 0
    while pos + tile <= length:
        out.append(slice(pos, pos + tile))
        pos += stride
    if not out:
        raise ValueError("tile must be <= length")
    if out[-1].stop < length:
        out.append(slice(length - tile, length))
    return out


def _blend_weight_2d(patch_h: int, patch_w: int, eps: float = 1e-3) -> np.ndarray:
    """Build a smooth 2D feathering window for patch blending.

    When patches overlap, weighted blending suppresses hard boundaries.
    """
    wy = np.hanning(patch_h) if patch_h > 1 else np.ones((1,), dtype=np.float32)
    wx = np.hanning(patch_w) if patch_w > 1 else np.ones((1,), dtype=np.float32)
    w = np.outer(wy, wx).astype(np.float32)
    w = np.maximum(w, eps)
    return w


def _to_int16_grid(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(np.rint(arr), 0, np.iinfo(np.int16).max)
    return arr.astype(np.int16, copy=False)


def build_trainer(
    work_dir: str,
    max_epochs: int,
    accelerator: str,
    devices: str,
    precision: int | str,
    dry_run_steps: int = 0,
    enable_val: bool = False,
) -> pl.Trainer:
    parsed_devices = int(devices) if str(devices).isdigit() else devices
    parsed_precision = int(precision) if str(precision).isdigit() else precision
    limit_train_batches = dry_run_steps if dry_run_steps > 0 else 1.0
    limit_val_batches = 1.0 if enable_val else 0
    return pl.Trainer(
        default_root_dir=work_dir,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=parsed_devices,
        precision=parsed_precision,  # type: ignore[arg-type]
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )


@torch.no_grad()
def predict_origin_patch_ensemble(
    model: torch.nn.Module,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    origin_start_idx: int,
    in_len: int,
    out_len: int,
    patch_h: int,
    patch_w: int,
    patch_stride_h: int,
    patch_stride_w: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    y_sum = np.zeros((out_len, core.height, core.width), dtype=np.float32)
    y_cnt = np.zeros((out_len, core.height, core.width), dtype=np.float32)
    patch_wt = _blend_weight_2d(patch_h, patch_w)[None, ...]

    h_slices = tile_slices(core.height, patch_h, patch_stride_h)
    w_slices = tile_slices(core.width, patch_w, patch_stride_w)

    for hs in h_slices:
        for ws in w_slices:
            x_patch, _ = core.get_window(
                int(origin_start_idx),
                in_len,
                out_len,
                lat_slice=hs,
                lon_slice=ws,
            )
            x_patch = normalizer.transform_x(x_patch)
            x_t = torch.from_numpy(x_patch[None]).float().to(device)
            y_hat = model(x_t).cpu().numpy()[0, ..., 0]
            y_hat = normalizer.inverse_y(y_hat)
            y_hat = np.maximum(y_hat, 0.0)
            y_sum[:, hs, ws] += y_hat * patch_wt
            y_cnt[:, hs, ws] += patch_wt

    return y_sum / np.clip(y_cnt, 1e-6, None)


@torch.no_grad()
def rolling_predict_2019(
    model: torch.nn.Module,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    in_len: int,
    out_len: int,
    patch_h: int,
    patch_w: int,
    patch_stride_h: int,
    patch_stride_w: int,
    output_dir: str,
):
    ensure_dir(output_dir)
    model.eval()
    device = next(model.parameters()).device

    year_start = pd.Timestamp("2019-01-01")
    year_end = pd.Timestamp("2019-12-31")
    year_days = pd.date_range(year_start, year_end, freq="D")
    day_to_ordinal = {d: i for i, d in enumerate(year_days)}

    out_dat = os.path.join(output_dir, "pm25_pred_2019.int16.dat")
    pred_mm = np.memmap(out_dat, mode="w+", dtype=np.int16, shape=(len(year_days), core.height, core.width))

    pending_sum: Dict[pd.Timestamp, np.ndarray] = {}
    pending_cnt: Dict[pd.Timestamp, np.ndarray] = {}

    starts = core.find_valid_start_indices(
        target_start="2019-01-01",
        target_end="2019-12-31",
        in_len=in_len,
        out_len=out_len,
    )

    h_slices = tile_slices(core.height, patch_h, patch_stride_h)
    w_slices = tile_slices(core.width, patch_w, patch_stride_w)
    patch_wt = _blend_weight_2d(patch_h, patch_w)[None, ...]

    for s in starts:
        y_sum = np.zeros((out_len, core.height, core.width), dtype=np.float32)
        y_cnt = np.zeros((out_len, core.height, core.width), dtype=np.float32)

        for hs in h_slices:
            for ws in w_slices:
                x_patch, _ = core.get_window(
                    int(s),
                    in_len,
                    out_len,
                    lat_slice=hs,
                    lon_slice=ws,
                )
                x_patch = normalizer.transform_x(x_patch)
                x_t = torch.from_numpy(x_patch[None]).float().to(device)
                y_hat = model(x_t).cpu().numpy()[0, ..., 0]
                y_hat = normalizer.inverse_y(y_hat)
                y_hat = np.maximum(y_hat, 0.0)
                y_sum[:, hs, ws] += y_hat * patch_wt
                y_cnt[:, hs, ws] += patch_wt

        y_full = y_sum / np.clip(y_cnt, 1e-6, None)

        target_days = core.pm25_time[s + in_len : s + in_len + out_len]
        for h in range(out_len):
            day = pd.Timestamp(target_days[h])
            if day < year_start or day > year_end:
                continue
            if day not in pending_sum:
                pending_sum[day] = np.zeros((core.height, core.width), dtype=np.float32)
                pending_cnt[day] = np.zeros((core.height, core.width), dtype=np.float32)
            pending_sum[day] += y_full[h]
            pending_cnt[day] += 1.0

        current_start = pd.Timestamp(core.pm25_time[s])
        finalize_threshold = current_start + pd.Timedelta(days=in_len)
        to_finalize = [d for d in pending_sum.keys() if d < finalize_threshold]
        for d in sorted(to_finalize):
            day_pred = pending_sum[d] / np.clip(pending_cnt[d], 1e-6, None)
            pred_mm[day_to_ordinal[d]] = _to_int16_grid(day_pred)
            del pending_sum[d]
            del pending_cnt[d]

    for d in sorted(pending_sum.keys()):
        day_pred = pending_sum[d] / np.clip(pending_cnt[d], 1e-6, None)
        pred_mm[day_to_ordinal[d]] = _to_int16_grid(day_pred)

    pred_mm.flush()

    da = xr.DataArray(
        pred_mm.astype(np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": year_days.values,
            "lat": core.pm25_ds["lat"].values,
            "lon": core.pm25_ds["lon"].values,
        },
        name="pm25_pred",
    )
    xr.Dataset({"pm25_pred": da}).to_zarr(os.path.join(output_dir, "pm25_pred_2019_rolling.zarr"), mode="w")

    da = xr.DataArray(
        pred_mm.astype(np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": year_days.values,
            "lat": core.pm25_ds["lat"].values,
            "lon": core.pm25_ds["lon"].values,
        },
        name="pm25_pred",
    )
    xr.Dataset({"pm25_pred": da}).to_zarr(os.path.join(output_dir, "pm25_pred_2019.zarr"), mode="w")


def _build_window_bounds(center_date: pd.Timestamp, window_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    train_end = center_date - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=window_days - 1)
    return train_start, train_end


def _fit_model_for_window(
    model_name: str,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    model_kwargs: dict,
    in_len: int,
    out_len: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    batch_size: int,
    num_workers: int,
    patch_h: int,
    patch_w: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    dry_run_steps: int,
    accelerator: str,
    devices: str,
    precision: str,
    work_dir: str,
    refit_normalizer: bool,
    enable_val: bool,
    prefetch_factor: int,
    pin_memory: bool,
    shm_cache_enabled: bool,
    shm_cache_dir: str,
    shm_cache_max_items: int,
    shm_cache_min_free_gb: float,
    lit_model: PM25ForecastLitModule | None = None,
) -> PM25ForecastLitModule:
    train_start_str = train_start.strftime("%Y-%m-%d")
    train_end_str = train_end.strftime("%Y-%m-%d")

    if refit_normalizer:
        normalizer.fit(
            core=core,
            in_len=in_len,
            out_len=out_len,
            train_target_start=train_start_str,
            train_target_end=train_end_str,
        )

    train_val_start = max(train_start, train_end - pd.Timedelta(days=6))
    val_start_str = train_val_start.strftime("%Y-%m-%d")
    val_end_str = train_end_str

    if lit_model is None:
        lit_model = PM25ForecastLitModule(
            model_name=model_name,
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            in_channels=core.input_channels,
            lr=lr,
            weight_decay=weight_decay,
            model_kwargs=model_kwargs,
        )

    dm = PM25DataModule(
        core=core,
        in_len=in_len,
        out_len=out_len,
        train_target_start=train_start_str,
        train_target_end=train_end_str,
        val_target_start=val_start_str,
        val_target_end=val_end_str,
        batch_size=batch_size,
        num_workers=num_workers,
        patch_h=patch_h,
        patch_w=patch_w,
        normalizer=normalizer,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shm_cache_enabled=shm_cache_enabled,
        shm_cache_dir=shm_cache_dir,
        shm_cache_max_items=shm_cache_max_items,
        shm_cache_min_free_gb=shm_cache_min_free_gb,
    )

    trainer = build_trainer(
        work_dir=work_dir,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        dry_run_steps=dry_run_steps,
        enable_val=enable_val,
    )
    trainer.fit(lit_model, datamodule=dm)
    return lit_model


def rolling_retrain_predict_2019(
    model_name: str,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    model_kwargs: dict,
    in_len: int,
    out_len: int,
    patch_h: int,
    patch_w: int,
    patch_stride_h: int,
    patch_stride_w: int,
    output_dir: str,
    rolling_train_window_days: int,
    rolling_step_days: int,
    train_lr: float,
    train_weight_decay: float,
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    dry_run_steps: int,
    accelerator: str,
    devices: str,
    precision: str,
    rolling_refit_norm_every_n_windows: int,
    rolling_enable_val: bool,
    prefetch_factor: int,
    pin_memory: bool,
    shm_cache_enabled: bool,
    shm_cache_dir: str,
    shm_cache_max_items: int,
    shm_cache_min_free_gb: float,
    train_start_date: str,
    predict_start_date: str,
    predict_end_date: str,
    train_first_window_only: bool = False,
):
    ensure_dir(output_dir)

    year_start = pd.Timestamp(predict_start_date)
    year_end = pd.Timestamp(predict_end_date)
    year_days = pd.date_range(year_start, year_end, freq="D")
    day_to_ordinal = {d: i for i, d in enumerate(year_days)}

    out_dat = os.path.join(output_dir, "pm25_pred_2019_rolling.int16.dat")
    pred_mm = np.memmap(out_dat, mode="w+", dtype=np.int16, shape=(len(year_days), core.height, core.width))

    pending_sum: Dict[pd.Timestamp, np.ndarray] = {}
    pending_cnt: Dict[pd.Timestamp, np.ndarray] = {}

    # Start origins earlier by in_len days so the first target can be year_start.
    # Example: in_len=5, predict_start=2019-01-01 => first origin=2018-12-27, targets 2019-01-01..2019-01-03.
    origin_start = year_start - pd.Timedelta(days=in_len)
    origin_end = year_end - pd.Timedelta(days=out_len - 1)
    origin_dates = pd.date_range(origin_start, origin_end, freq=f"{rolling_step_days}D")

    lit_model: PM25ForecastLitModule | None = None
    model_file = os.path.join(output_dir, "rolling_last.ckpt")

    for win_idx, origin_date in enumerate(origin_dates):
        train_start, train_end = _build_window_bounds(origin_date, rolling_train_window_days)
        if train_start < pd.Timestamp(train_start_date):
            train_start = pd.Timestamp(train_start_date)

        if rolling_refit_norm_every_n_windows <= 0:
            refit_normalizer = (win_idx == 0)
        else:
            refit_normalizer = (win_idx % rolling_refit_norm_every_n_windows == 0)

        lit_model = _fit_model_for_window(
            model_name=model_name,
            core=core,
            normalizer=normalizer,
            model_kwargs=model_kwargs,
            in_len=in_len,
            out_len=out_len,
            train_start=train_start,
            train_end=train_end,
            batch_size=batch_size,
            num_workers=num_workers,
            patch_h=patch_h,
            patch_w=patch_w,
            lr=train_lr,
            weight_decay=train_weight_decay,
            max_epochs=max_epochs,
            dry_run_steps=dry_run_steps,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            work_dir=output_dir,
            refit_normalizer=refit_normalizer,
            enable_val=rolling_enable_val,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            shm_cache_enabled=shm_cache_enabled,
            shm_cache_dir=shm_cache_dir,
            shm_cache_max_items=shm_cache_max_items,
            shm_cache_min_free_gb=shm_cache_min_free_gb,
            lit_model=lit_model,
        )

        y_full = predict_origin_patch_ensemble(
            model=lit_model,
            core=core,
            normalizer=normalizer,
            origin_start_idx=int(np.where(core.pm25_time == origin_date)[0][0]),
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            patch_stride_h=patch_stride_h,
            patch_stride_w=patch_stride_w,
        )

        target_days = core.pm25_time[
            int(np.where(core.pm25_time == origin_date)[0][0]) + in_len : int(np.where(core.pm25_time == origin_date)[0][0]) + in_len + out_len
        ]
        for h in range(out_len):
            day = pd.Timestamp(target_days[h])
            if day < year_start or day > year_end:
                continue
            if day not in pending_sum:
                pending_sum[day] = np.zeros((core.height, core.width), dtype=np.float32)
                pending_cnt[day] = np.zeros((core.height, core.width), dtype=np.float32)
            pending_sum[day] += y_full[h]
            pending_cnt[day] += 1.0

        finalize_threshold = origin_date + pd.Timedelta(days=in_len)
        to_finalize = [d for d in pending_sum.keys() if d < finalize_threshold]
        for d in sorted(to_finalize):
            day_pred = pending_sum[d] / np.clip(pending_cnt[d], 1e-6, None)
            pred_mm[day_to_ordinal[d]] = _to_int16_grid(day_pred)
            del pending_sum[d]
            del pending_cnt[d]

        lit_model.to("cpu")
        torch.save(lit_model.state_dict(), model_file)

        if train_first_window_only:
            break

    for d in sorted(pending_sum.keys()):
        day_pred = pending_sum[d] / np.clip(pending_cnt[d], 1e-6, None)
        pred_mm[day_to_ordinal[d]] = _to_int16_grid(day_pred)

    pred_mm.flush()

