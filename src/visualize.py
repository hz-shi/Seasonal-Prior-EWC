import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_prediction(pred_path: str, time_len: int, h: int, w: int) -> xr.DataArray:
    if pred_path.endswith(".zarr"):
        ds = xr.open_zarr(pred_path)
        if "pm25_pred" in ds:
            return ds["pm25_pred"]
        raise ValueError(f"'pm25_pred' not found in {pred_path}")

    dtype = np.int16 if pred_path.endswith(".int16.dat") else np.float32
    arr = np.memmap(pred_path, mode="r", dtype=dtype, shape=(time_len, h, w))
    return xr.DataArray(arr, dims=("time", "lat", "lon"))


def parse_args():
    p = argparse.ArgumentParser(description="Visualize PM2.5 rolling predictions against ground truth")
    p.add_argument("--pred_path", type=str, required=True, help="Path to pm25_pred_2019_rolling.zarr, .int16.dat, or .float32.dat")
    p.add_argument("--pm25_path", type=str, required=True, help="Path to ground-truth pm25.zarr")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--start_date", type=str, default="2019-01-01")
    p.add_argument("--end_date", type=str, default="2019-12-31")
    p.add_argument("--plot_date", type=str, default="2019-07-01")
    p.add_argument("--vmax", type=float, default=-1.0, help="Color scale max for PM2.5 maps; <=0 means auto p99")
    p.add_argument("--map_stride", type=int, default=2, help="Spatial stride for map plotting; 1 means full resolution")
    p.add_argument("--metric_stride", type=int, default=4, help="Spatial stride for metrics; 1 means full resolution")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    gt_ds = xr.open_zarr(args.pm25_path)
    gt = gt_ds["pm25"].sel(time=slice(args.start_date, args.end_date)).astype(np.float32)
    t = gt.sizes["time"]
    h = gt.sizes["lat"]
    w = gt.sizes["lon"]

    pred = load_prediction(args.pred_path, t, h, w)
    if "time" not in pred.coords:
        pred = pred.assign_coords(time=gt.time.values, lat=gt.lat.values, lon=gt.lon.values)
    else:
        pred = pred.sel(time=slice(args.start_date, args.end_date))

    pred = pred.astype(np.float32)

    metric_stride = max(1, int(args.metric_stride))
    map_stride = max(1, int(args.map_stride))

    gt_metric = gt[:, ::metric_stride, ::metric_stride]
    pred_metric = pred[:, ::metric_stride, ::metric_stride]

    err = pred_metric - gt_metric
    abs_err = np.abs(err)

    mae = float(abs_err.mean().values)
    rmse = float(np.sqrt((err ** 2).mean().values))
    bias = float(err.mean().values)

    with open(os.path.join(args.output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"BIAS: {bias:.6f}\n")

    plot_day = pd.Timestamp(args.plot_date)
    pred_day = pred.sel(time=plot_day).isel(lat=slice(None, None, map_stride), lon=slice(None, None, map_stride))
    gt_day = gt.sel(time=plot_day).isel(lat=slice(None, None, map_stride), lon=slice(None, None, map_stride))
    ae_day = np.abs(pred_day - gt_day)

    vmax = args.vmax if args.vmax > 0 else float(np.nanpercentile(gt_metric.values, 99))
    emax = float(np.nanpercentile(abs_err.values, 99))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    im0 = axes[0].imshow(gt_day.values, cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title(f"Ground Truth {plot_day.date()}")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_day.values, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title(f"Prediction {plot_day.date()}")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(ae_day.values, cmap="magma", vmin=0, vmax=emax)
    axes[2].set_title(f"Absolute Error {plot_day.date()}")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"PM2.5 Comparison | MAE={mae:.3f} RMSE={rmse:.3f} BIAS={bias:.3f}")
    fig.savefig(os.path.join(args.output_dir, "map_compare.png"), dpi=150)
    plt.close(fig)

    # Time series at representative points
    hm = gt.sizes["lat"]
    wm = gt.sizes["lon"]
    pts = [
        (hm // 2, wm // 2, "center"),
        (hm // 4, wm // 4, "q1"),
        (3 * hm // 4, 3 * wm // 4, "q3"),
    ]
    fig, axes = plt.subplots(len(pts), 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    time_vals = pd.to_datetime(gt.time.values)
    for ax, (ri, ci, tag) in zip(axes, pts):
        gt_ts = gt[:, ri, ci].values
        pd_ts = pred[:, ri, ci].values
        ax.plot(time_vals, gt_ts, label="gt", linewidth=1.2)
        ax.plot(time_vals, pd_ts, label="pred", linewidth=1.2)
        ax.set_title(f"Point {tag} (row={ri}, col={ci})")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    fig.savefig(os.path.join(args.output_dir, "timeseries_compare.png"), dpi=150)
    plt.close(fig)

    # Domain-wide daily MAE curve
    daily_mae = abs_err.mean(dim=("lat", "lon"))
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.plot(time_vals, daily_mae.values, color="tab:red", linewidth=1.3)
    ax.set_title("Daily Domain MAE")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "daily_mae.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
