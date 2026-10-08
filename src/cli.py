import argparse
import hashlib
import json
import os
import subprocess
import tomllib
import warnings
from datetime import datetime
from typing import Any, Dict

warnings.filterwarnings(
    "ignore",
    message=(
        r"`isinstance\(treespec, LeafSpec\)` is deprecated, use "
        r"`isinstance\(treespec, TreeSpec\) and treespec\.is_leaf\(\)` instead\."
    ),
    module=r"pytorch_lightning\.utilities\._pytree",
)

def _str2bool(v):
    if v is None or isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy

from .data import DataPaths, MultiSourcePM25Core, PM25DataModule
from .lightning_module import PM25ForecastLitModule
from .normalization import PM25Normalizer, default_modes_for_model
from .reg.condition_aware_ewc import (
    ConditionAwareEWC,
    EWCMetadataMismatchError,
    build_condition_aware_ewc_bank,
    load_condition_aware_ewc,
    warm_condition_cache,
)
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

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_name", type=str, choices=["earthformer", "iam4vp", "phydnet", "unet", "convlstm"], default=None)
    p.add_argument("--mode", type=str, choices=["train", "predict", "train_predict", "rolling_train_predict"], default=None)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--work_dir", type=str, default=None)
    p.add_argument("--predict_start_date", type=str, default="2019-01-01")
    p.add_argument("--predict_end_date", type=str, default="2019-12-31")
    p.add_argument("--predict_output_prefix", type=str, default="pm25_pred_2019")

    p.add_argument("--in_len", type=int, default=5)
    p.add_argument("--out_len", type=int, default=3)
    p.add_argument("--rolling_train_window_days", type=int, default=60)
    p.add_argument("--rolling_step_days", type=int, default=1)
    p.add_argument("--rolling_train_start_date", type=str, default="2018-01-01")
    p.add_argument("--resume_rolling", action="store_true")
    p.add_argument("--no_resume_rolling", action="store_true")
    p.add_argument("--rolling_start_window_idx", type=int, default=0)
    p.add_argument("--train_first_window_only", type=str, default=None, nargs="?", const="true")

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
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--shm_cache_enabled", nargs="?", const=True, default=False, type=_str2bool)
    p.add_argument("--shm_cache_dir", type=str, default="/dev/shm/pm25_window_cache")
    p.add_argument("--shm_cache_max_items", type=int, default=0)
    p.add_argument("--shm_cache_min_free_gb", type=float, default=2.0)
    p.add_argument("--shm_cache_x_dtype", type=str, choices=["float32", "float16"], default="float32")
    p.add_argument("--cuda_prefetch", action="store_true")
    p.add_argument("--dataloader_multiprocessing_context", type=str, default="")
    p.add_argument("--train_patches_per_sample", type=int, default=1)

    p.add_argument("--zarr_parallel_backend", type=str, choices=["serial", "thread"], default="serial")
    p.add_argument("--zarr_read_workers", type=int, default=1)
    p.add_argument("--preload_to_memory", type=_str2bool, default=None)
    p.add_argument("--preload_workers", type=int, default=0)
    # Known-future GFS covariates: append forecast-day t+1..t+out_len GFS fields
    # as extra feature planes. Default false preserves the legacy 41-channel
    # input; the paper configs opt in explicitly.
    p.add_argument("--future_gfs_enabled", nargs="?", const=True, default=False, type=_str2bool)

    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--log_every_n_steps", type=int, default=1)
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
    p.add_argument("--unet_base_channels", type=int, default=32)
    p.add_argument("--unet_depth", type=int, default=4)
    p.add_argument("--unet_dropout", type=float, default=0.0)
    p.add_argument("--convlstm_hidden_dims", nargs="+", type=int, default=[32, 32])
    p.add_argument("--convlstm_kernel_size", type=int, default=3)

    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="1")
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--matmul_precision", type=str, choices=["", "highest", "high", "medium"], default="")
    p.add_argument("--cuda_visible_devices", type=str, default="")

    p.add_argument("--x_norm_mode", type=str, default="auto", choices=["auto", "none", "zscore", "minmax_01", "minmax_m11"])
    p.add_argument("--y_norm_mode", type=str, default="auto", choices=["auto", "none", "zscore", "minmax_01", "minmax_m11"])
    p.add_argument("--norm_stats_path", type=str, default="")
    p.add_argument("--rolling_refit_norm_every_n_windows", type=int, default=0)
    p.add_argument("--rolling_enable_val", action="store_true")
    p.add_argument("--ca_ewc_enabled", action="store_true")
    p.add_argument("--ca_ewc_lambda", type=float, default=0.0)
    p.add_argument("--ca_ewc_condition_scheme", type=str, choices=["season4", "month12", "single", "global"], default="season4")
    p.add_argument("--ca_ewc_exclude_param_patterns", nargs="*", default=[])
    p.add_argument("--ca_ewc_bank_path", type=str, default="")
    p.add_argument("--ca_ewc_auto_build_bank", action="store_true")
    p.add_argument("--ca_ewc_offline_max_epochs", type=int, default=1)
    p.add_argument("--ca_ewc_offline_max_steps_per_condition", type=int, default=0)
    p.add_argument("--ca_ewc_fisher_batches", type=int, default=32)
    p.add_argument("--ca_ewc_offline_lr", type=float, default=1e-4)
    p.add_argument("--ca_ewc_offline_weight_decay", type=float, default=1e-4)
    p.add_argument("--ca_ewc_bank_dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--ca_ewc_bank_train_start", type=str, default="2017-01-01")
    p.add_argument("--ca_ewc_bank_train_end", type=str, default="2018-12-31")
    p.add_argument("--ca_ewc_online_update_enabled", action="store_true")
    p.add_argument("--ca_ewc_online_fisher_batches", type=int, default=4)
    p.add_argument("--ca_ewc_online_theta_alpha", type=float, default=0.05)
    p.add_argument("--ca_ewc_online_omega_alpha", type=float, default=0.1)
    p.add_argument("--ca_ewc_online_update_every_n_windows", type=int, default=1)
    p.add_argument("--ca_ewc_online_save_every_n_windows", type=int, default=1)
    p.add_argument("--ca_ewc_transition_ranges", type=str, default="")
    p.add_argument("--ca_ewc_transition_lambda_scale", type=float, default=1.0)
    p.add_argument("--ca_ewc_cache_warmup", type=_str2bool, nargs="?", const=True, default=None)
    p.add_argument("--ca_ewc_cache_num_workers", type=int, default=8)
    p.add_argument("--ca_ewc_device_cache_enabled", action="store_true")

    p.add_argument("--reviewer_method", type=str, choices=["legacy", "baseline", "standard_ewc", "seasonal_only", "seasonal_prior_ewc"], default="legacy")
    p.add_argument("--reviewer_fisher_variant", type=str, choices=["empirical", "shuffled", "uniform", "inverse"], default="empirical")
    p.add_argument("--reviewer_fisher_seed", type=int, default=0)
    p.add_argument("--reviewer_standard_bank_path", type=str, default="")

    p.add_argument("--parameter_analysis_enabled", action="store_true")
    p.add_argument("--parameter_analysis_bank_path", type=str, default="")
    p.add_argument("--parameter_analysis_dir", type=str, default="")
    p.add_argument("--parameter_analysis_top_k", type=int, default=50000)
    p.add_argument("--parameter_analysis_dtype", type=str, choices=["float32", "float16"], default="float32")

    if cfg:
        p.set_defaults(**cfg)

    args = p.parse_args(remaining)
    if getattr(args, "no_resume_rolling", False):
        # Explicitly disable rolling resume even when a config sets
        # resume_rolling=true. --no_resume_rolling always wins.
        args.resume_rolling = False
    if getattr(args, "train_first_window_only", None) is None and "train_first_window_only" in cfg:
        args.train_first_window_only = cfg["train_first_window_only"]
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
        preload_to_memory=args.preload_to_memory,
        preload_workers=args.preload_workers,
        future_gfs_enabled=bool(getattr(args, "future_gfs_enabled", False)),
        future_gfs_steps=int(args.out_len),
    )


def build_trainer(args) -> pl.Trainer:
    devices = _parse_devices_arg(args.devices)
    precision = int(args.precision) if str(args.precision).isdigit() else args.precision
    limit_train_batches = args.dry_run_steps if args.dry_run_steps > 0 else 1.0
    limit_val_batches = 0 if args.dry_run_steps > 0 else 1.0
    strategy = None
    accelerator = str(args.accelerator).lower()
    if accelerator in ["auto", "gpu", "cuda"] and torch.cuda.is_available() and _count_devices(devices) > 1:
        major, minor = torch.cuda.get_device_capability(0)
        if major < 5:
            # On Kepler-era cards (e.g., K80), NCCL object collectives can fail with
            # "CUDA error: named symbol not found". Gloo avoids that code path.
            strategy = DDPStrategy(
                process_group_backend="gloo",
                find_unused_parameters=True,
            )
            print(
                f"Detected legacy GPU capability {major}.{minor} with multi-GPU; "
                "using DDP(gloo backend, find_unused_parameters=True) to avoid "
                "NCCL CUDA symbol errors and tolerate conditionally unused parameters."
            )
    trainer_kwargs = {
        "default_root_dir": args.work_dir,
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": devices,
        "precision": precision,  # type: ignore[arg-type]
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": max(1, int(args.log_every_n_steps)),
    }
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    return pl.Trainer(**trainer_kwargs)


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
        "unet_base_channels": args.unet_base_channels,
        "unet_depth": args.unet_depth,
        "unet_dropout": args.unet_dropout,
        "convlstm_hidden_dims": args.convlstm_hidden_dims,
        "convlstm_kernel_size": args.convlstm_kernel_size,
    }
    if args.earthformer_enc_depth is not None:
        out["earthformer_enc_depth"] = list(args.earthformer_enc_depth)
    if args.earthformer_dec_depth is not None:
        out["earthformer_dec_depth"] = list(args.earthformer_dec_depth)
    return out


def _parse_devices_arg(raw):
    if isinstance(raw, int):
        return raw
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [int(tok.strip()) for tok in inner.split(",")]
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if parts and all(p.isdigit() for p in parts):
            return [int(p) for p in parts]
    return raw


def _count_devices(devices) -> int:
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    return 1


def _global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    for key in ("RANK", "GLOBAL_RANK", "SLURM_PROCID", "LOCAL_RANK"):
        rank_env = os.environ.get(key)
        if rank_env is not None:
            try:
                return int(rank_env)
            except ValueError:
                pass
    return 0


def _is_global_zero_process() -> bool:
    return _global_rank() == 0


def resolve_ca_ewc_build_device(args) -> torch.device:
    accelerator = str(args.accelerator).lower()
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if accelerator in ["auto", "gpu", "cuda"]:
        if not torch.cuda.is_available():
            if accelerator in ["gpu", "cuda"]:
                raise RuntimeError("accelerator requests GPU/CUDA, but CUDA is not available.")
            return torch.device("cpu")
        devices = _parse_devices_arg(args.devices)
        if isinstance(devices, int):
            raise ValueError(
                "CA-EWC bank build requires an explicit CUDA device list, e.g. devices=[5]; "
                "bare devices=5 means device count and is ambiguous."
            )
        if isinstance(devices, (list, tuple)) and devices:
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None and local_rank_env.isdigit():
                lr = int(local_rank_env)
                if lr < len(devices):
                    return torch.device(f"cuda:{int(devices[lr])}")
            return torch.device(f"cuda:{int(devices[0])}")
        raise ValueError(
            f"CA-EWC bank build requires an explicit CUDA device list, e.g. devices=[5]; "
            f"got unsupported devices format: {args.devices}"
        )
    return torch.device("cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_probe_cuda() -> None:
    """Probe CUDA capability and set local rank device.

    Must be called AFTER CPU-only cache warm-up (if any) to avoid fork
    deadlocks from CUDA-initialized parent processes.
    """
    if not torch.cuda.is_available():
        return
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None and local_rank_env.isdigit():
        torch.cuda.set_device(int(local_rank_env))
    dev_idx = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(dev_idx)
    if major < 5:
        torch.backends.cudnn.enabled = False
        print(
            f"Detected legacy GPU capability {major}.{minor}; "
            "disabled cuDNN to avoid unsupported-architecture runtime errors."
        )


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _source_file_hashes() -> Dict[str, str]:
    root = _repo_root()
    rel_paths = [
        "src/cli.py",
        "src/data.py",
        "src/lightning_module.py",
        "src/normalization.py",
        "src/reg/condition_aware_ewc.py",
        "src/reg/conditions.py",
    ]
    out = {}
    for rel_path in rel_paths:
        path = os.path.join(root, rel_path)
        out[rel_path] = _file_sha256(path) if os.path.exists(path) else "missing"
    return out


def build_ca_ewc_bank_metadata(
    args,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    lit_model: PM25ForecastLitModule,
) -> Dict[str, Any]:
    model_kwargs = build_model_kwargs(args)
    parameter_schema = {
        n: list(p.shape)
        for n, p in lit_model.model.named_parameters()
        if p.requires_grad
    }
    data_paths = {
        "year_path": args.year_path,
        "eral_path": args.eral_path,
        "erap_path": args.erap_path,
        "gfs_path": args.gfs_path,
        "meic_path": args.meic_path,
        "pm25_path": args.pm25_path,
    }
    metadata = {
        "metadata_version": 1,
        "model_name": args.model_name,
        "model_kwargs": _jsonable(model_kwargs),
        "in_channels": int(core.input_channels),
        "in_len": int(args.in_len),
        "out_len": int(args.out_len),
        "patch_h": int(args.patch_h),
        "patch_w": int(args.patch_w),
        "parameter_schema_hash": _stable_hash(parameter_schema),
        "trainable_parameter_count": int(sum(p.numel() for p in lit_model.model.parameters() if p.requires_grad)),
        "normalizer_x_mode": normalizer.x_mode,
        "normalizer_y_mode": normalizer.y_mode,
        "normalizer_fingerprint": normalizer.fingerprint(),
        "condition_scheme": args.ca_ewc_condition_scheme,
        "ca_ewc_exclude_param_patterns": _as_string_list(args.ca_ewc_exclude_param_patterns),
        "ca_ewc_bank_train_start": args.ca_ewc_bank_train_start,
        "ca_ewc_bank_train_end": args.ca_ewc_bank_train_end,
        "ca_ewc_bank_dtype": args.ca_ewc_bank_dtype,
        "shm_cache_x_dtype": getattr(args, "shm_cache_x_dtype", "float32"),
        "ca_ewc_offline_max_epochs": int(args.ca_ewc_offline_max_epochs),
        "ca_ewc_offline_lr": float(args.ca_ewc_offline_lr),
        "ca_ewc_offline_weight_decay": float(args.ca_ewc_offline_weight_decay),
        "ca_ewc_offline_max_steps_per_condition": int(args.ca_ewc_offline_max_steps_per_condition),
        "ca_ewc_fisher_batches": int(args.ca_ewc_fisher_batches),
        "batch_size": int(args.batch_size),
        "data_paths": _jsonable(data_paths),
        "data_paths_hash": _stable_hash(data_paths),
        "git_commit": _git_commit(),
        "source_file_hashes": _source_file_hashes(),
        "bank_build_contract_version": 1,
    }
    metadata["metadata_hash"] = _stable_hash(metadata)
    return metadata


def _archive_existing_bank(bank_path: str) -> str | None:
    if not os.path.exists(bank_path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_path = f"{bank_path}.stale_metadata.{ts}"
    os.replace(bank_path, archived_path)
    return archived_path


def _build_ca_ewc_bank(
    args,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    lit_model: PM25ForecastLitModule,
    metadata: Dict[str, Any],
) -> None:
    bank_device = resolve_ca_ewc_build_device(args)
    lit_model.model = lit_model.model.to(bank_device)
    build_condition_aware_ewc_bank(
        model=lit_model.model,
        core=core,
        normalizer=normalizer,
        in_len=args.in_len,
        out_len=args.out_len,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        train_target_start=args.ca_ewc_bank_train_start,
        train_target_end=args.ca_ewc_bank_train_end,
        condition_scheme=args.ca_ewc_condition_scheme,
        bank_path=args.ca_ewc_bank_path,
        offline_lr=args.ca_ewc_offline_lr,
        offline_weight_decay=args.ca_ewc_offline_weight_decay,
        offline_max_epochs=args.ca_ewc_offline_max_epochs,
        offline_max_steps_per_condition=args.ca_ewc_offline_max_steps_per_condition,
        fisher_batches=args.ca_ewc_fisher_batches,
        batch_size=args.batch_size,
        shm_cache_enabled=args.shm_cache_enabled,
        shm_cache_dir=args.shm_cache_dir,
        shm_cache_max_items=args.shm_cache_max_items,
        shm_cache_min_free_gb=args.shm_cache_min_free_gb,
        shm_cache_x_dtype=args.shm_cache_x_dtype,
        bank_dtype=args.ca_ewc_bank_dtype,
        precision=args.precision,
        excluded_param_patterns=_as_string_list(args.ca_ewc_exclude_param_patterns),
        metadata=metadata,
    )


def apply_reviewer_method(args) -> None:
    """根据 reviewer_method 解析并应用实际 EWC 运行配置。

    五种模式：
    - legacy: 完全按旧 ca_ewc_enabled / 旧 scheme / 旧 lambda 行为，不改变任何逻辑。
    - baseline: 本运行禁用 EWC bank/penalty，不改变其它训练参数。
    - standard_ewc: 启用 EWC，强制 scheme=single，importance_variant 取
      reviewer_fisher_variant（默认 empirical）。始终使用 per-run runtime bank
      （<work_dir>/ca_ewc_bank_runtime.pt），shared source bank 仅由 runner 复制、
      CLI 绝不写入；若 runtime bank 缺失，沿用既有兼容校验/构建机制
      （EWCMetadataMismatchError + auto_build），不误加载 season4 bank。
    - seasonal_only: 启用 EWC，强制 season4，importance_variant=uniform
      （等权 seasonal-anchor mean squared parameter distance，无 Fisher weighting）。
    - seasonal_prior_ewc: 启用 EWC，强制 season4，importance_variant 取
      reviewer_fisher_variant。

    通过修改 args 上的 EWC 相关字段，使 train/predict/train_predict/rolling
    所有调用路径行为一致。旧配置（无 reviewer_method 键）默认 legacy，
    完全保持旧行为。
    """
    method = getattr(args, "reviewer_method", "legacy") or "legacy"
    variant = getattr(args, "reviewer_fisher_variant", "empirical") or "empirical"
    seed = int(getattr(args, "reviewer_fisher_seed", 0) or 0)

    # 默认值：与旧行为一致（legacy）
    enabled = bool(args.ca_ewc_enabled)
    scheme = args.ca_ewc_condition_scheme
    bank_path = args.ca_ewc_bank_path
    importance_variant = "empirical"
    importance_seed = 0

    if method == "legacy":
        pass
    elif method == "baseline":
        enabled = False
    elif method == "standard_ewc":
        enabled = True
        scheme = "single"
        importance_variant = variant
        importance_seed = seed
        # standard_ewc always runs against the per-run runtime bank
        # (<work_dir>/ca_ewc_bank_runtime.pt). The shared source bank
        # (reviewer_standard_bank_path) is only ever copied by the runner and
        # must never be written to by the CLI, so it is intentionally not used
        # as the runtime bank path here.
        bank_path = os.path.join(args.work_dir, "ca_ewc_bank_runtime.pt")
    elif method == "seasonal_only":
        enabled = True
        scheme = "season4"
        importance_variant = "uniform"
        importance_seed = seed
    elif method == "seasonal_prior_ewc":
        enabled = True
        scheme = "season4"
        importance_variant = variant
        importance_seed = seed
    else:
        raise ValueError(
            f"Unsupported reviewer_method={method!r}. "
            "Choose from legacy, baseline, standard_ewc, seasonal_only, seasonal_prior_ewc."
        )

    args.ca_ewc_enabled = enabled
    args.ca_ewc_condition_scheme = scheme
    args.ca_ewc_bank_path = bank_path
    args._reviewer_importance_variant = importance_variant
    args._reviewer_importance_seed = importance_seed

    if _is_global_zero_process():
        print(
            f"[reviewer] method={method} enabled={enabled} scheme={scheme} "
            f"variant={importance_variant} seed={importance_seed} lambda={args.ca_ewc_lambda}"
        )


def maybe_prepare_condition_aware_ewc(
    args,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    lit_model: PM25ForecastLitModule,
) -> ConditionAwareEWC | None:
    if not args.ca_ewc_enabled:
        return None
    if args.ca_ewc_lambda <= 0:
        raise ValueError("ca_ewc_enabled=true requires ca_ewc_lambda > 0.")
    if not normalizer.fitted:
        raise ValueError("Normalizer must be fitted before enabling Condition-Aware EWC.")

    bank_path = args.ca_ewc_bank_path or os.path.join(args.work_dir, "ca_ewc_bank.pt")
    args.ca_ewc_bank_path = bank_path
    expected_metadata = build_ca_ewc_bank_metadata(args, core=core, normalizer=normalizer, lit_model=lit_model)
    importance_variant = getattr(args, "_reviewer_importance_variant", "empirical")
    importance_seed = getattr(args, "_reviewer_importance_seed", 0)

    if not os.path.exists(bank_path):
        if not args.ca_ewc_auto_build_bank:
            raise FileNotFoundError(
                f"Condition-Aware EWC bank not found at: {bank_path}. "
                "Enable --ca_ewc_auto_build_bank to build it automatically."
            )
        _build_ca_ewc_bank(args, core=core, normalizer=normalizer, lit_model=lit_model, metadata=expected_metadata)

    try:
        return load_condition_aware_ewc(
            model=lit_model.model,
            bank_path=bank_path,
            lambda_=args.ca_ewc_lambda,
            bank_dtype=args.ca_ewc_bank_dtype,
            precision=args.precision,
            expected_metadata=expected_metadata,
            excluded_param_patterns=_as_string_list(args.ca_ewc_exclude_param_patterns),
            importance_variant=importance_variant,
            importance_seed=importance_seed,
            device_cache_enabled=bool(getattr(args, "ca_ewc_device_cache_enabled", False)),
        )
    except EWCMetadataMismatchError as exc:
        if not args.ca_ewc_auto_build_bank:
            raise
        archived_path = _archive_existing_bank(bank_path)
        if _is_global_zero_process():
            print(f"Existing Condition-Aware EWC bank metadata mismatch; rebuilding {bank_path}.")
            if archived_path is not None:
                print(f"Archived stale EWC bank to: {archived_path}")
            print(str(exc))
        _build_ca_ewc_bank(args, core=core, normalizer=normalizer, lit_model=lit_model, metadata=expected_metadata)
        return load_condition_aware_ewc(
            model=lit_model.model,
            bank_path=bank_path,
            lambda_=args.ca_ewc_lambda,
            bank_dtype=args.ca_ewc_bank_dtype,
            precision=args.precision,
            expected_metadata=expected_metadata,
            excluded_param_patterns=_as_string_list(args.ca_ewc_exclude_param_patterns),
            importance_variant=importance_variant,
            importance_seed=importance_seed,
            device_cache_enabled=bool(getattr(args, "ca_ewc_device_cache_enabled", False)),
        )


def main():
    args = parse_args()
    apply_reviewer_method(args)
    if getattr(args, "seed", None) is not None:
        pl.seed_everything(args.seed, workers=True)


    cuda_visible_devices = getattr(args, "cuda_visible_devices", "")
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    # Defer ALL torch.cuda.* calls until after CPU-only cache warm-up.
    # warm_condition_cache() forks DataLoader workers; fork after CUDA init
    # can deadlock.  CUDA probing happens in _maybe_probe_cuda() called
    # right before the first CUDA-using code path.
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)

    if args.ca_ewc_enabled and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    ensure_dir(args.work_dir)
    os.environ.setdefault("PM25_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if _is_global_zero_process():
        print(f"PM25_RUN_ID={os.environ['PM25_RUN_ID']}")

    if args.no_pin_memory:
        args.pin_memory = False
    os.environ.setdefault("PM25_ZARR_READ_WORKERS_IN_DATALOADER", str(max(1, int(args.zarr_read_workers))))

    core = build_core(args)
    if _is_global_zero_process():
        print(
            f"[cli] input_channels={core.input_channels} "
            f"history_channels={core.history_channels} "
            f"known_future_channels={core.known_future_channels} "
            f"future_gfs_enabled={core.future_gfs_enabled} "
            f"horizon={core.future_gfs_steps}"
        )
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
    condition_ewc = None

    if args.mode in ["train", "train_predict"]:
        _maybe_probe_cuda()
        normalizer.fit(
            core=core,
            in_len=args.in_len,
            out_len=args.out_len,
            train_target_start=args.train_target_start,
            train_target_end=args.train_target_end,
        )
        if _is_global_zero_process():
            normalizer.save(norm_stats_path)
        lit_model.set_y_transform_from_normalizer(normalizer)
        condition_ewc = maybe_prepare_condition_aware_ewc(args, core=core, normalizer=normalizer, lit_model=lit_model)
        lit_model.condition_ewc = condition_ewc

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
            persistent_workers=args.persistent_workers,
            shm_cache_enabled=args.shm_cache_enabled,
            shm_cache_dir=args.shm_cache_dir,
            shm_cache_max_items=args.shm_cache_max_items,
            shm_cache_min_free_gb=args.shm_cache_min_free_gb,
            shm_cache_x_dtype=args.shm_cache_x_dtype,
            condition_scheme=args.ca_ewc_condition_scheme,
            train_patches_per_sample=args.train_patches_per_sample,
            cuda_prefetch=args.cuda_prefetch,
            dataloader_multiprocessing_context=args.dataloader_multiprocessing_context,
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
            schema_ok, schema_reason = normalizer.validate_against_core(core)
            if not schema_ok:
                raise ValueError(
                    f"Normalization stats at {norm_stats_path} are incompatible with "
                    f"the current input schema: {schema_reason}. Refit or provide "
                    "matching stats."
                )

        if args.ckpt_path:
            lit_model = PM25ForecastLitModule.load_from_checkpoint(args.ckpt_path)

        if args.patch_stride_h <= 0:
            args.patch_stride_h = args.patch_h
        if args.patch_stride_w <= 0:
            args.patch_stride_w = args.patch_w

        runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lit_model = lit_model.to(runtime_device)
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
            infer_batch_size=1,
            predict_start_date=args.predict_start_date,
            predict_end_date=args.predict_end_date,
            output_prefix=args.predict_output_prefix,
        )

    if args.mode == "rolling_train_predict":
        from .rolling import rolling_retrain_predict_2019

        if args.patch_stride_h <= 0:
            args.patch_stride_h = args.patch_h
        if args.patch_stride_w <= 0:
            args.patch_stride_w = args.patch_w

        # Unified normalization for every rolling variant (No/Standard/Seasonal):
        # load the configured stats if present, otherwise fit once on the
        # historical bank range (2017-2018 by default) and freeze them. This
        # matches the manuscript protocol (statistics estimated exclusively from
        # 2017-2018 and held fixed) and avoids re-fitting per window.
        expected_fit_range = (args.ca_ewc_bank_train_start, args.ca_ewc_bank_train_end)
        if not normalizer.fitted and os.path.exists(norm_stats_path):
            loaded_normalizer = PM25Normalizer.load(norm_stats_path)
            loaded_fit_range = (
                loaded_normalizer.fit_target_start, loaded_normalizer.fit_target_end
            )
            schema_ok, schema_reason = loaded_normalizer.validate_against_core(core)
            if (
                loaded_normalizer.fitted
                and loaded_fit_range == expected_fit_range
                and loaded_normalizer.x_mode == normalizer.x_mode
                and loaded_normalizer.y_mode == normalizer.y_mode
                and schema_ok
            ):
                normalizer = loaded_normalizer
                if _is_global_zero_process():
                    print(f"Loaded rolling normalizer stats from: {norm_stats_path}")
            elif _is_global_zero_process():
                print(
                    "Ignoring rolling normalizer stats with incompatible mode, fit range "
                    "or input schema: "
                    f"file=({loaded_normalizer.x_mode},{loaded_normalizer.y_mode}) "
                    f"current=({normalizer.x_mode},{normalizer.y_mode}), "
                    f"file_range={loaded_fit_range}, expected_range={expected_fit_range}, "
                    f"schema_ok={schema_ok} ({schema_reason})"
                )
        if not normalizer.fitted:
            normalizer.fit(
                core=core,
                in_len=args.in_len,
                out_len=args.out_len,
                train_target_start=args.ca_ewc_bank_train_start,
                train_target_end=args.ca_ewc_bank_train_end,
            )
            if _is_global_zero_process():
                normalizer.save(norm_stats_path)
        if _is_global_zero_process():
            print(
                "[cli] fixed rolling normalization: "
                f"range={normalizer.fit_target_start}..{normalizer.fit_target_end}, "
                f"fingerprint={normalizer.fingerprint()}"
            )

        if args.ca_ewc_enabled:
            # CPU-only cache warm-up: must run BEFORE ANY torch.cuda.* call
            # to avoid fork deadlocks from CUDA-initialized parent process.
            if getattr(args, "ca_ewc_cache_warmup", False) and args.shm_cache_enabled:
                print("[cli] warming up SHM cache for EWC bank (CPU-only, no CUDA)", flush=True)
                warm_condition_cache(
                    core=core,
                    normalizer=normalizer,
                    in_len=args.in_len,
                    out_len=args.out_len,
                    patch_h=args.patch_h,
                    patch_w=args.patch_w,
                    train_target_start=args.ca_ewc_bank_train_start,
                    train_target_end=args.ca_ewc_bank_train_end,
                    condition_scheme=args.ca_ewc_condition_scheme,
                    batch_size=args.batch_size,
                    shm_cache_enabled=args.shm_cache_enabled,
                    shm_cache_dir=args.shm_cache_dir,
                    shm_cache_max_items=args.shm_cache_max_items,
                    shm_cache_min_free_gb=args.shm_cache_min_free_gb,
                    shm_cache_x_dtype=args.shm_cache_x_dtype,
                    num_workers=getattr(args, "ca_ewc_cache_num_workers", 8),
                    multiprocessing_context=args.dataloader_multiprocessing_context,
                )
                print("[cli] SHM cache warm-up done", flush=True)

        # Probe CUDA AFTER warm-up; first torch.cuda.* call in this mode.
        _maybe_probe_cuda()

        if args.ca_ewc_enabled:
            condition_ewc = maybe_prepare_condition_aware_ewc(args, core=core, normalizer=normalizer, lit_model=lit_model)

        # Fairness reseed: re-seed right before rolling training so every
        # non-legacy reviewer method (baseline included) starts from the same
        # deterministic RNG state regardless of EWC bank preparation above.
        reviewer_method = str(getattr(args, "reviewer_method", "legacy") or "legacy")
        if reviewer_method != "legacy" and getattr(args, "seed", None) is not None:
            pl.seed_everything(args.seed, workers=True)
            if _is_global_zero_process():
                print(f"[cli] fairness reseed: pl.seed_everything({args.seed}, workers=True) for reviewer_method={reviewer_method}")

        train_first_window_only_val = args.dry_run_steps > 0
        if getattr(args, "train_first_window_only", None) is not None:
            val = args.train_first_window_only
            if isinstance(val, bool):
                train_first_window_only_val = val
            elif isinstance(val, str):
                val_lower = val.lower().strip()
                if val_lower in ("true", "1", "yes", "on"):
                    train_first_window_only_val = True
                elif val_lower in ("false", "0", "no", "off"):
                    train_first_window_only_val = False
                else:
                    raise ValueError(f"Invalid boolean value for train_first_window_only: {val}")
            else:
                raise ValueError(f"Invalid type for train_first_window_only: {type(val)}")

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
            log_every_n_steps=args.log_every_n_steps,
            dry_run_steps=args.dry_run_steps,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            rolling_refit_norm_every_n_windows=args.rolling_refit_norm_every_n_windows,
            rolling_enable_val=args.rolling_enable_val,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            shm_cache_enabled=args.shm_cache_enabled,
            shm_cache_dir=args.shm_cache_dir,
            shm_cache_max_items=args.shm_cache_max_items,
            shm_cache_min_free_gb=args.shm_cache_min_free_gb,
            shm_cache_x_dtype=args.shm_cache_x_dtype,
            cuda_prefetch=args.cuda_prefetch,
            dataloader_multiprocessing_context=args.dataloader_multiprocessing_context,
            condition_scheme=args.ca_ewc_condition_scheme,
            train_patches_per_sample=args.train_patches_per_sample,
            normalizer_stats_path=norm_stats_path,
            condition_ewc=condition_ewc,
            ca_ewc_bank_path=args.ca_ewc_bank_path,
            ca_ewc_online_update_enabled=args.ca_ewc_online_update_enabled,
            ca_ewc_online_fisher_batches=args.ca_ewc_online_fisher_batches,
            ca_ewc_online_theta_alpha=args.ca_ewc_online_theta_alpha,
            ca_ewc_online_omega_alpha=args.ca_ewc_online_omega_alpha,
            ca_ewc_online_update_every_n_windows=args.ca_ewc_online_update_every_n_windows,
            ca_ewc_online_save_every_n_windows=args.ca_ewc_online_save_every_n_windows,
            ca_ewc_transition_ranges=args.ca_ewc_transition_ranges,
            ca_ewc_transition_lambda_scale=args.ca_ewc_transition_lambda_scale,
            train_start_date=args.rolling_train_start_date,
            predict_start_date=args.predict_start_date,
            predict_end_date=args.predict_end_date,
            train_first_window_only=train_first_window_only_val,
            seed=getattr(args, "seed", None),
            resume_rolling=args.resume_rolling,
            start_window_idx=args.rolling_start_window_idx,
            parameter_analysis_enabled=args.parameter_analysis_enabled,
            parameter_analysis_bank_path=args.parameter_analysis_bank_path,
            parameter_analysis_dir=args.parameter_analysis_dir,
            parameter_analysis_top_k=args.parameter_analysis_top_k,
            parameter_analysis_dtype=args.parameter_analysis_dtype,
            reviewer_method=str(getattr(args, "reviewer_method", "legacy") or "legacy"),
            reviewer_fisher_variant=str(
                getattr(args, "_reviewer_importance_variant", None)
                or getattr(args, "reviewer_fisher_variant", "empirical")
                or "empirical"
            ),
            reviewer_fisher_seed=int(getattr(args, "reviewer_fisher_seed", 0) or 0),
        )


if __name__ == "__main__":
    main()
