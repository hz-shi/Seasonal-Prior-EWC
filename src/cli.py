import argparse
import json
import os
import tomllib
from typing import Dict

import pytorch_lightning as pl
import torch

from .data import DataPaths, MultiSourcePM25Core, PM25DataModule
from .lightning_module import PM25ForecastLitModule
from .normalization import PM25Normalizer, default_modes_for_model
from .rolling import ensure_dir, rolling_predict_2019


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()

    cfg = load_config(pre_args.config) if pre_args.config else {}

    p = argparse.ArgumentParser(description="Train/predict PM2.5 with multi-source zarr and pluggable models")
    p.add_argument("--config", type=str, default=pre_args.config)

    p.add_argument("--year_path", type=str, default=None)
    p.add_argument("--eral_path", type=str, default=None)
    p.add_argument("--erap_path", type=str, default=None)
    p.add_argument("--gfs_path", type=str, default=None)
    p.add_argument("--meic_path", type=str, default=None)
    p.add_argument("--pm25_path", type=str, default=None)

    p.add_argument("--model_name", type=str, choices=["earthformer", "iam4vp", "phydnet"], default=None)
    p.add_argument("--mode", type=str, choices=["train", "predict", "train_predict", "rolling_train_predict"], default=None)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--work_dir", type=str, default=None)

    p.add_argument("--in_len", type=int, default=5)
    p.add_argument("--out_len", type=int, default=3)
    p.add_argument("--rolling_train_window_days", type=int, default=60)
    p.add_argument("--rolling_step_days", type=int, default=1)

    p.add_argument("--train_target_start", type=str, default="2017-01-06")
    p.add_argument("--train_target_end", type=str, default="2018-12-31")
    p.add_argument("--val_target_start", type=str, default="2020-01-01")
    p.add_argument("--val_target_end", type=str, default="2020-12-31")

    p.add_argument("--patch_h", type=int, default=None)
    p.add_argument("--patch_w", type=int, default=None)
    p.add_argument("--patch_stride_h", type=int, default=0)
    p.add_argument("--patch_stride_w", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--no_pin_memory", action="store_true")
    p.add_argument("--shm_cache_enabled", action="store_true")
    p.add_argument("--shm_cache_dir", type=str, default="/dev/shm/pm25_window_cache")
    p.add_argument("--shm_cache_max_items", type=int, default=0)
    p.add_argument("--shm_cache_min_free_gb", type=float, default=2.0)

    p.add_argument("--zarr_parallel_backend", type=str, choices=["serial", "thread"], default="serial")
    p.add_argument("--zarr_read_workers", type=int, default=1)

    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--dry_run_steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--base_units", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--earthformer_enc_depth", nargs="+", type=int, default=None)
    p.add_argument("--earthformer_dec_depth", nargs="+", type=int, default=None)
    p.add_argument("--earthformer_downsample", type=int, default=2)
    p.add_argument("--earthformer_num_global_vectors", type=int, default=0)
    p.add_argument("--earthformer_z_init_method", type=str, default="nearest_interp")
    p.add_argument("--earthformer_initial_downsample_scale", type=int, default=2)
    p.add_argument("--earthformer_initial_downsample_conv_layers", type=int, default=2)
    p.add_argument("--earthformer_final_upsample_conv_layers", type=int, default=2)
    p.add_argument("--earthformer_attn_drop", type=float, default=0.0)
    p.add_argument("--earthformer_proj_drop", type=float, default=0.0)
    p.add_argument("--earthformer_ffn_drop", type=float, default=0.0)
    p.add_argument("--iam4vp_hid_s", type=int, default=64)
    p.add_argument("--iam4vp_hid_t", type=int, default=512)
    p.add_argument("--iam4vp_n_s", type=int, default=4)
    p.add_argument("--iam4vp_n_t", type=int, default=6)

    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="1")
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--cuda_visible_devices", type=str, default="")

    p.add_argument("--x_norm_mode", type=str, default="auto", choices=["auto", "none", "zscore", "minmax_01", "minmax_m11"])
    p.add_argument("--y_norm_mode", type=str, default="auto", choices=["auto", "none", "zscore", "minmax_01", "minmax_m11"])
    p.add_argument("--norm_stats_path", type=str, default="")
    p.add_argument("--rolling_refit_norm_every_n_windows", type=int, default=0)
    p.add_argument("--rolling_enable_val", action="store_true")

    if cfg:
        p.set_defaults(**cfg)

    args = p.parse_args(remaining)
    validate_args(args)
    return args


def load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "rb") as f:
        if ext == ".toml":
            cfg = tomllib.load(f)
        elif ext == ".json":
            cfg = json.loads(f.read().decode("utf-8"))
        else:
            raise ValueError("Unsupported config extension. Use .toml or .json")

    return flatten_config(cfg)


def flatten_config(cfg: Dict) -> Dict:
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def validate_args(args):
    required_common = [
        "year_path",
        "eral_path",
        "erap_path",
        "gfs_path",
        "meic_path",
        "pm25_path",
        "model_name",
        "mode",
        "work_dir",
        "patch_h",
        "patch_w",
    ]
    missing = [k for k in required_common if getattr(args, k) in [None, ""]]
    if missing:
        raise ValueError(f"Missing required arguments (CLI or config): {missing}")


def build_core(args) -> MultiSourcePM25Core:
    paths = DataPaths(
        year_path=args.year_path,
        eral_path=args.eral_path,
        erap_path=args.erap_path,
        gfs_path=args.gfs_path,
        meic_path=args.meic_path,
        pm25_path=args.pm25_path,
    )
    return MultiSourcePM25Core(
        paths,
        zarr_parallel_backend=args.zarr_parallel_backend,
        zarr_read_workers=args.zarr_read_workers,
    )


def build_trainer(args) -> pl.Trainer:
    devices = int(args.devices) if args.devices.isdigit() else args.devices
    precision = int(args.precision) if str(args.precision).isdigit() else args.precision
    limit_train_batches = args.dry_run_steps if args.dry_run_steps > 0 else 1.0
    limit_val_batches = 0 if args.dry_run_steps > 0 else 1.0
    return pl.Trainer(
        default_root_dir=args.work_dir,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=precision,  # type: ignore[arg-type]
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )


def build_model_kwargs(args) -> Dict[str, object]:
    out: Dict[str, object] = {
        "base_units": args.base_units,
        "num_heads": args.num_heads,
        "earthformer_downsample": args.earthformer_downsample,
        "earthformer_num_global_vectors": args.earthformer_num_global_vectors,
        "earthformer_z_init_method": args.earthformer_z_init_method,
        "earthformer_initial_downsample_scale": args.earthformer_initial_downsample_scale,
        "earthformer_initial_downsample_conv_layers": args.earthformer_initial_downsample_conv_layers,
        "earthformer_final_upsample_conv_layers": args.earthformer_final_upsample_conv_layers,
        "earthformer_attn_drop": args.earthformer_attn_drop,
        "earthformer_proj_drop": args.earthformer_proj_drop,
        "earthformer_ffn_drop": args.earthformer_ffn_drop,
        "hid_s": args.iam4vp_hid_s,
        "hid_t": args.iam4vp_hid_t,
        "n_s": args.iam4vp_n_s,
        "n_t": args.iam4vp_n_t,
    }
    if args.earthformer_enc_depth is not None:
        out["earthformer_enc_depth"] = list(args.earthformer_enc_depth)
    if args.earthformer_dec_depth is not None:
        out["earthformer_dec_depth"] = list(args.earthformer_dec_depth)
    return out


def main():
    args = parse_args()

    cuda_visible_devices = getattr(args, "cuda_visible_devices", "")
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    ensure_dir(args.work_dir)

    if args.no_pin_memory:
        args.pin_memory = False

    core = build_core(args)
    model_kwargs = build_model_kwargs(args)

    default_modes = default_modes_for_model(args.model_name)
    x_mode = default_modes.x_mode if args.x_norm_mode == "auto" else args.x_norm_mode
    y_mode = default_modes.y_mode if args.y_norm_mode == "auto" else args.y_norm_mode
    norm_stats_path = args.norm_stats_path or os.path.join(args.work_dir, "normalizer_stats.npz")

    normalizer = PM25Normalizer(x_mode=x_mode, y_mode=y_mode)

    lit_model = PM25ForecastLitModule(
        model_name=args.model_name,
        in_len=args.in_len,
        out_len=args.out_len,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        in_channels=core.input_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        model_kwargs=model_kwargs,
    )

    if args.mode in ["train", "train_predict"]:
        normalizer.fit(
            core=core,
            in_len=args.in_len,
            out_len=args.out_len,
            train_target_start=args.train_target_start,
            train_target_end=args.train_target_end,
        )
        normalizer.save(norm_stats_path)

        dm = PM25DataModule(
            core=core,
            in_len=args.in_len,
            out_len=args.out_len,
            train_target_start=args.train_target_start,
            train_target_end=args.train_target_end,
            val_target_start=args.val_target_start,
            val_target_end=args.val_target_end,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            normalizer=normalizer,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            shm_cache_enabled=args.shm_cache_enabled,
            shm_cache_dir=args.shm_cache_dir,
            shm_cache_max_items=args.shm_cache_max_items,
            shm_cache_min_free_gb=args.shm_cache_min_free_gb,
        )
        trainer = build_trainer(args)
        trainer.fit(lit_model, datamodule=dm)

        checkpoint_cb = trainer.checkpoint_callback
        if checkpoint_cb is not None:
            best = getattr(checkpoint_cb, "best_model_path", "")
            if best:
                args.ckpt_path = best

    if args.mode in ["predict", "train_predict"]:
        if args.mode == "predict":
            if not os.path.exists(norm_stats_path):
                raise FileNotFoundError(
                    f"Normalization stats not found: {norm_stats_path}. "
                    "Run training first or pass --norm_stats_path."
                )
            normalizer = PM25Normalizer.load(norm_stats_path)

        if args.ckpt_path:
            lit_model = PM25ForecastLitModule.load_from_checkpoint(args.ckpt_path)

        if args.patch_stride_h <= 0:
            args.patch_stride_h = args.patch_h
        if args.patch_stride_w <= 0:
            args.patch_stride_w = args.patch_w

        lit_model = lit_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        rolling_predict_2019(
            model=lit_model,
            core=core,
            normalizer=normalizer,
            in_len=args.in_len,
            out_len=args.out_len,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            patch_stride_h=args.patch_stride_h,
            patch_stride_w=args.patch_stride_w,
            output_dir=args.work_dir,
        )

    if args.mode == "rolling_train_predict":
        from .rolling import rolling_retrain_predict_2019

        if args.patch_stride_h <= 0:
            args.patch_stride_h = args.patch_h
        if args.patch_stride_w <= 0:
            args.patch_stride_w = args.patch_w

        rolling_retrain_predict_2019(
            model_name=args.model_name,
            core=core,
            normalizer=normalizer,
            model_kwargs=model_kwargs,
            in_len=args.in_len,
            out_len=args.out_len,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            patch_stride_h=args.patch_stride_h,
            patch_stride_w=args.patch_stride_w,
            output_dir=args.work_dir,
            rolling_train_window_days=args.rolling_train_window_days,
            rolling_step_days=args.rolling_step_days,
            train_lr=args.lr,
            train_weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_epochs=args.max_epochs,
            dry_run_steps=args.dry_run_steps,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            rolling_refit_norm_every_n_windows=args.rolling_refit_norm_every_n_windows,
            rolling_enable_val=args.rolling_enable_val,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            shm_cache_enabled=args.shm_cache_enabled,
            shm_cache_dir=args.shm_cache_dir,
            shm_cache_max_items=args.shm_cache_max_items,
            shm_cache_min_free_gb=args.shm_cache_min_free_gb,
            train_start_date="2018-01-01",
            predict_start_date="2019-01-01",
            predict_end_date="2019-12-31",
            train_first_window_only=args.dry_run_steps > 0,
        )


if __name__ == "__main__":
    main()
