import os
import csv
import gc
import json
import sys
import traceback
import math
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger

from .data import MultiSourcePM25Core
from .data import PM25DataModule
from .lightning_module import PM25ForecastLitModule
from .normalization import PM25Normalizer
from .reg.condition_aware_ewc import ConditionAwareEWC

_RANK_LOGGING_CONFIGURED = False


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


def _daily_mae_rmse(day_pred: np.ndarray, day_true: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(day_true) & (day_true > 0) & np.isfinite(day_pred)
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        return float("nan"), float("nan"), 0
    diff = day_pred[valid] - day_true[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse, valid_count


def _horizon_metrics(day_pred: np.ndarray, day_true: np.ndarray) -> tuple[float, float, float, float, int]:
    """Per-horizon metrics for one origin's raw prediction vs truth.

    Uses the same valid mask as _daily_mae_rmse. Returns
    (mae, rmse, r2, bias, valid_pixels). R2 = 1 - SSE/SST where SST is the sum
    of squared deviations of the valid truth from its spatial mean; NaN when
    SST <= 0 (degenerate constant field).
    """
    valid = np.isfinite(day_true) & (day_true > 0) & np.isfinite(day_pred)
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    pred = day_pred[valid]
    true = day_true[valid]
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    bias = float(np.mean(diff))
    sse = float(np.sum(diff * diff))
    sst = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return mae, rmse, r2, bias, valid_count


def _derive_reviewer_metadata(
    model_name: str,
    seed: int | None,
    condition_ewc: ConditionAwareEWC | None,
    condition_scheme: str,
    ca_ewc_base_lambda: float,
    reviewer_method: str = "legacy",
    reviewer_fisher_variant: str = "",
) -> dict:
    """Derive reviewer run metadata from inputs available to rolling.

    When reviewer_method / reviewer_fisher_variant are explicitly provided,
    they are used directly. Otherwise the method is inferred from the EWC object
    state (bank condition_scheme + importance_variant).
    """
    clean_method = str(reviewer_method or "legacy").strip()
    if clean_method == "baseline":
        method = "baseline"
        fisher_variant = ""
    elif clean_method == "legacy":
        method = "legacy"
        if condition_ewc is not None:
            fisher_variant = str(reviewer_fisher_variant or getattr(condition_ewc, "importance_variant", "") or "empirical")
        else:
            fisher_variant = str(reviewer_fisher_variant or "")
    elif clean_method == "seasonal_only":
        method = "seasonal_only"
        fisher_variant = "uniform"
    elif clean_method in ("standard_ewc", "seasonal_prior_ewc"):
        method = clean_method
        if reviewer_fisher_variant:
            fisher_variant = str(reviewer_fisher_variant)
        elif condition_ewc is not None:
            fisher_variant = str(getattr(condition_ewc, "importance_variant", "") or "empirical")
        else:
            fisher_variant = "empirical"
    elif clean_method:
        method = clean_method
        fisher_variant = str(reviewer_fisher_variant or "")
    elif condition_ewc is None:
        method = "legacy"
        fisher_variant = str(reviewer_fisher_variant or "")
    else:
        scheme = str(
            getattr(getattr(condition_ewc, "bank", None), "condition_scheme", "") or condition_scheme or ""
        )
        variant = str(getattr(condition_ewc, "importance_variant", "") or "empirical")
        if scheme == "single":
            method = "standard_ewc"
        elif scheme == "season4":
            method = "seasonal_only" if variant == "uniform" else "seasonal_prior_ewc"
        else:
            method = "legacy"
        fisher_variant = variant

    seasonal_prior = method in ("seasonal_only", "seasonal_prior_ewc")
    return {
        "model": str(model_name),
        "method": method,
        "seed": seed,
        "ewc_lambda": 0.0 if method == "baseline" else float(ca_ewc_base_lambda),
        "seasonal_prior": seasonal_prior,
        "fisher_variant": fisher_variant,
    }


def _validate_reviewer_csv_metadata(csv_path: str, current_meta: dict) -> None:
    """Validate that an existing reviewer CSV matches the current run metadata.

    Inspects the first data row of the CSV. If model, method, seed, lambda,
    or fisher_variant mismatch, raises RuntimeError to prevent corrupted metrics.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return
        first_row = next(reader, None)
        if not first_row or len(first_row) < len(header):
            return

    row_dict = dict(zip(header, first_row))

    def _float_equal(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol

    checks = [
        ("model", str(current_meta.get("model", "")), str(row_dict.get("model", "")), lambda a, b: a == b),
        ("method", str(current_meta.get("method", "")), str(row_dict.get("method", "")), lambda a, b: a == b),
        ("seed", str(current_meta.get("seed", "")), str(row_dict.get("seed", "")), lambda a, b: a == b),
        (
            "lambda",
            float(current_meta.get("ewc_lambda", 0.0)),
            float(row_dict.get("lambda", row_dict.get("ewc_lambda", 0.0)) or 0.0),
            _float_equal,
        ),
        (
            "fisher_variant",
            str(current_meta.get("fisher_variant", "")),
            str(row_dict.get("fisher_variant", "")),
            lambda a, b: a == b,
        ),
    ]

    for field_name, curr_val, file_val, comp_fn in checks:
        if field_name not in row_dict:
            continue
        try:
            if not comp_fn(curr_val, file_val):
                raise RuntimeError(
                    f"Reviewer CSV metadata mismatch for '{field_name}': "
                    f"existing CSV has '{file_val}', current run has '{curr_val}'. "
                    f"Refusing to resume or append into mismatched CSV: {csv_path}"
                )
        except ValueError as exc:
            raise RuntimeError(
                f"Reviewer CSV metadata parse error on field '{field_name}': {exc}"
            ) from exc




def _compute_state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> tuple[str, int, int]:
    """Compute a deterministic SHA256 digest over the model state_dict.

    Traverses keys in sorted order. For each key, feeds:
      - key name (length prefix + UTF-8 bytes)
      - dtype string
      - tensor shape
      - raw underlying bytes of contiguous CPU tensor in 1MB chunks.
    bfloat16 tensors are viewed as int16 to access raw memoryview bytes.
    Does not mutate tensor storage, gradient history, or RNG state.

    Returns:
        (hex_digest, tensor_count, parameter_count)
    """
    hasher = hashlib.sha256()
    total_elements = 0
    chunk_size = 1024 * 1024  # 1MB chunking

    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        t = v.detach().cpu()
        hasher.update(len(k).to_bytes(4, byteorder="big"))
        hasher.update(k.encode("utf-8"))
        hasher.update(str(t.dtype).encode("utf-8"))
        hasher.update(str(list(t.shape)).encode("utf-8"))

        t_contig = t.contiguous()
        if t_contig.dtype == torch.bfloat16:
            t_contig = t_contig.view(torch.int16)
        mv = memoryview(t_contig.numpy()).cast("B")
        for i in range(0, len(mv), chunk_size):
            hasher.update(mv[i : i + chunk_size])

        total_elements += t.numel()

    return hasher.hexdigest(), len(state_dict), total_elements


def _audit_initial_state_dict(
    work_dir: str,
    state_dict: dict[str, torch.Tensor],
    model: str,
    method: str,
    seed: int,
    window_idx: int = 0,
) -> dict[str, Any]:
    """Audit model initial state_dict prior to optimizer stepping in window 0.

    Writes reviewer_initialization_audit.json in work_dir. If the file already exists,
    verifies metadata and SHA256 consistency; raises RuntimeError on mismatch.
    Only called for reviewer_method != 'legacy' on rank 0.
    """
    os.makedirs(work_dir, exist_ok=True)
    audit_file = os.path.join(work_dir, "reviewer_initialization_audit.json")

    sha256_hash, tensor_count, param_count = _compute_state_dict_sha256(state_dict)

    if os.path.exists(audit_file):
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read existing initialization audit file {audit_file}: {exc}"
            ) from exc

        mismatches: list[str] = []
        if str(existing.get("model", "")).strip().lower() != str(model).strip().lower():
            mismatches.append(f"model: expected {model}, got {existing.get('model')}")
        if str(existing.get("method", "")).strip().lower() != str(method).strip().lower():
            mismatches.append(f"method: expected {method}, got {existing.get('method')}")
        if int(existing.get("seed", -1)) != int(seed):
            mismatches.append(f"seed: expected {seed}, got {existing.get('seed')}")
        if int(existing.get("window_idx", -1)) != int(window_idx):
            mismatches.append(f"window_idx: expected {window_idx}, got {existing.get('window_idx')}")
        if str(existing.get("state_dict_sha256", "")) != sha256_hash:
            mismatches.append(
                f"state_dict_sha256: expected {sha256_hash}, got {existing.get('state_dict_sha256')}"
            )

        if mismatches:
            raise RuntimeError(
                f"Reviewer initialization audit mismatch in {audit_file}: {'; '.join(mismatches)}"
            )
        return existing

    payload = {
        "model": str(model),
        "method": str(method),
        "seed": int(seed),
        "hash_algorithm": "sha256",
        "state_dict_sha256": sha256_hash,
        "tensor_count": int(tensor_count),
        "parameter_count": int(param_count),
        "window_idx": int(window_idx),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    tmp_file = audit_file + f".tmp.{os.getpid()}"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, audit_file)
    return payload


def _read_reviewer_done_keys(csv_path: str) -> set[tuple[str, str, str]]:
    """Read (origin_date, forecast_date, horizon) keys already present in the reviewer CSV.

    Used for resume safety: rows already written are never duplicated. Returns
    an empty set when the file does not exist. Backwards-compatible with 14-column
    CSVs (origin_date falls back to empty string).
    """
    done: set[tuple[str, str, str]] = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return done
        if "origin_date" in header and "forecast_date" in header and "horizon" in header:
            idx_orig = header.index("origin_date")
            idx_fc = header.index("forecast_date")
            idx_hz = header.index("horizon")
            for row in reader:
                if len(row) > max(idx_orig, idx_fc, idx_hz):
                    done.add((row[idx_orig], row[idx_fc], row[idx_hz]))
        else:
            # Legacy 14-column CSV without origin_date column
            for row in reader:
                if len(row) >= 8 and row[6] != "forecast_date":
                    done.add(("", row[6], row[7]))
    return done


def build_trainer(
    work_dir: str,
    max_epochs: int,
    log_every_n_steps: int,
    accelerator: str,
    devices: str,
    precision: int | str,
    dry_run_steps: int = 0,
    enable_val: bool = False,
    logger_version: str | None = None,
    callbacks: list | None = None,
) -> pl.Trainer:
    parsed_devices = _parse_devices_arg(devices)
    parsed_precision = int(precision) if str(precision).isdigit() else precision
    limit_train_batches = dry_run_steps if dry_run_steps > 0 else 1.0
    limit_val_batches = 1.0 if enable_val else 0
    strategy = None
    accelerator_name = str(accelerator).lower()
    if accelerator_name in ["auto", "gpu", "cuda"] and torch.cuda.is_available() and _count_devices(parsed_devices) > 1:
        major, minor = torch.cuda.get_device_capability(0)
        if major < 5:
            strategy = DDPStrategy(
                process_group_backend="gloo",
                find_unused_parameters=True,
            )
            print(
                f"Detected legacy GPU capability {major}.{minor} with multi-GPU; "
                "using DDP(gloo backend, find_unused_parameters=True) to avoid "
                "NCCL CUDA symbol errors and tolerate conditionally unused parameters."
            )
    logger_obj: bool | CSVLogger = True
    if logger_version is not None:
        logger_obj = CSVLogger(save_dir=work_dir, name="rolling_logs", version=logger_version)

    trainer_kwargs = {
        "default_root_dir": work_dir,
        "max_epochs": max_epochs,
        "accelerator": accelerator,
        "devices": parsed_devices,
        "precision": parsed_precision,  # type: ignore[arg-type]
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": max(1, int(log_every_n_steps)),
        "logger": logger_obj,
        "enable_checkpointing": False,
        "enable_model_summary": False,
    }
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    return pl.Trainer(**trainer_kwargs)


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


def _resolve_runtime_device(accelerator: str, devices=None) -> torch.device:
    name = str(accelerator).lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name in ["auto", "gpu", "cuda"] and torch.cuda.is_available():
        parsed = _parse_devices_arg(devices) if devices is not None else None
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None and local_rank_env.isdigit():
                idx = int(local_rank_env)
                return torch.device(f"cuda:{parsed[idx]}")
            return torch.device(f"cuda:{parsed[0]}")
        return torch.device("cuda")
    return torch.device("cpu")


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


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _setup_rank_logging(output_dir: str) -> None:
    global _RANK_LOGGING_CONFIGURED
    if _RANK_LOGGING_CONFIGURED:
        return
    run_id = os.environ.get("PM25_RUN_ID")
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["PM25_RUN_ID"] = run_id
    log_dir = os.path.join(output_dir, "rank_logs", run_id)
    os.makedirs(log_dir, exist_ok=True)
    rank = _global_rank()
    log_path = os.path.join(log_dir, f"rank_{rank}.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    print(f"[rank {rank}] logging to {log_path}", file=log_file)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    _RANK_LOGGING_CONFIGURED = True


def _load_rolling_state(state_path: str) -> dict:
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_rolling_state(state_path: str, state: dict) -> None:
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, state_path)


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
    infer_batch_size: int = 1,
    predict_start_date: str = "2019-01-01",
    predict_end_date: str = "2019-12-31",
    output_prefix: str = "pm25_pred_2019",
):
    ensure_dir(output_dir)
    model.eval()
    device = next(model.parameters()).device

    year_start = pd.Timestamp(predict_start_date)
    year_end = pd.Timestamp(predict_end_date)
    year_days = pd.date_range(year_start, year_end, freq="D")
    day_to_ordinal = {d: i for i, d in enumerate(year_days)}

    out_dat_final = os.path.join(output_dir, f"{output_prefix}.int16.dat")
    if os.path.exists(out_dat_final):
        os.remove(out_dat_final)
    pred_mm = np.memmap(out_dat_final, mode="w+", dtype=np.int16, shape=(len(year_days), core.height, core.width))

    pending_sum: Dict[pd.Timestamp, np.ndarray] = {}
    pending_cnt: Dict[pd.Timestamp, np.ndarray] = {}

    starts = core.find_valid_start_indices(
        target_start=predict_start_date,
        target_end=predict_end_date,
        in_len=in_len,
        out_len=out_len,
    )

    h_slices = tile_slices(core.height, patch_h, patch_stride_h)
    w_slices = tile_slices(core.width, patch_w, patch_stride_w)
    patch_wt = _blend_weight_2d(patch_h, patch_w)[None, ...]
    infer_batch_size = max(1, int(infer_batch_size))

    run_ok = False
    try:
        for s in starts:
            y_sum = np.zeros((out_len, core.height, core.width), dtype=np.float32)
            y_cnt = np.zeros((out_len, core.height, core.width), dtype=np.float32)

            patch_inputs: list[tuple[slice, slice, np.ndarray]] = []
            for hs in h_slices:
                for ws in w_slices:
                    x_patch, _ = core.get_window(
                        int(s),
                        in_len,
                        out_len,
                        lat_slice=hs,
                        lon_slice=ws,
                    )
                    patch_inputs.append((hs, ws, normalizer.transform_x(x_patch)))

            for i in range(0, len(patch_inputs), infer_batch_size):
                chunk = patch_inputs[i : i + infer_batch_size]
                x_np = np.stack([x for _, _, x in chunk], axis=0)
                x_t = torch.from_numpy(x_np).float().to(device, non_blocking=True)
                y_hat_batch = model(x_t).cpu().numpy()[..., 0]
                for j, (hs, ws, _) in enumerate(chunk):
                    y_hat = normalizer.inverse_y(y_hat_batch[j])
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
            # A day d receives its last contribution at origin date (d - in_len).
            # Finalize when d <= current_origin + in_len to avoid one-window lag.
            to_finalize = [d for d in pending_sum.keys() if d <= finalize_threshold]
            for d in sorted(to_finalize):
                day_pred = pending_sum[d] / np.clip(pending_cnt[d], 1e-6, None)
                pred_mm[day_to_ordinal[d]] = _to_int16_grid(day_pred)
                del pending_sum[d]
                del pending_cnt[d]
            if to_finalize:
                # Persist finalized day-level predictions for real-time inspection.
                pred_mm.flush()

        for d in sorted(pending_sum.keys()):
            day_pred = pending_sum[d] / np.clip(pending_cnt[d], 1e-6, None)
            pred_mm[day_to_ordinal[d]] = _to_int16_grid(day_pred)

        pred_mm.flush()
        run_ok = True
    finally:
        try:
            pred_mm.flush()
        except Exception:
            pass
        del pred_mm
        if not run_ok and os.path.exists(out_dat_final):
            os.remove(out_dat_final)

    da = xr.DataArray(
        np.memmap(out_dat_final, mode="r", dtype=np.int16, shape=(len(year_days), core.height, core.width)).astype(np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": year_days.values,
            "lat": core.pm25_lat,
            "lon": core.pm25_lon,
        },
        name="pm25_pred",
    )
    xr.Dataset({"pm25_pred": da}).to_zarr(os.path.join(output_dir, f"{output_prefix}_rolling.zarr"), mode="w")

    da = xr.DataArray(
        np.memmap(out_dat_final, mode="r", dtype=np.int16, shape=(len(year_days), core.height, core.width)).astype(np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": year_days.values,
            "lat": core.pm25_lat,
            "lon": core.pm25_lon,
        },
        name="pm25_pred",
    )
    xr.Dataset({"pm25_pred": da}).to_zarr(os.path.join(output_dir, f"{output_prefix}.zarr"), mode="w")


def _parse_ca_ewc_transition_ranges(spec: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ranges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for raw_part in str(spec or "").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid CA-EWC transition range {part!r}; expected MM-DD:MM-DD.")
        start_s, end_s = [x.strip() for x in part.split(":", 1)]
        start = pd.Timestamp(f"2000-{start_s}")
        end = pd.Timestamp(f"2000-{end_s}")
        ranges.append(((int(start.month), int(start.day)), (int(end.month), int(end.day))))
    return ranges


def _date_matches_month_day_range(day: pd.Timestamp, start: tuple[int, int], end: tuple[int, int]) -> bool:
    md = (int(day.month), int(day.day))
    if start <= end:
        return start <= md <= end
    return md >= start or md <= end


def _ca_ewc_window_lambda_scale(
    target_start: pd.Timestamp,
    out_len: int,
    transition_ranges: list[tuple[tuple[int, int], tuple[int, int]]],
    transition_lambda_scale: float,
) -> float:
    if not transition_ranges:
        return 1.0
    target_days = pd.date_range(target_start, target_start + pd.Timedelta(days=out_len - 1), freq="D")
    for day in target_days:
        for start, end in transition_ranges:
            if _date_matches_month_day_range(pd.Timestamp(day), start, end):
                return float(transition_lambda_scale)
    return 1.0


def _build_window_bounds(
    origin_date: pd.Timestamp, window_days: int, in_len: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """End the historical target window immediately before the first forecast.

    ``origin_date`` is the first of the ``in_len`` historical predictor days,
    not the forecast date. Those predictor days are already observed when the
    forecast is issued, so they may also lie within the validation target range.
    """
    first_forecast_date = pd.Timestamp(origin_date) + pd.Timedelta(days=in_len)
    train_end = first_forecast_date - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=window_days - 1)
    return train_start, train_end


@dataclass
class WindowTargetProtocol:
    """Resolved train/validation target-date ranges for one rolling window."""

    train_target_start: pd.Timestamp
    train_target_end: pd.Timestamp
    val_target_start: pd.Timestamp | None
    val_target_end: pd.Timestamp | None
    train_target_days: int
    val_target_days: int
    train_samples: int
    val_samples: int

    def as_dict(self) -> dict:
        return {
            "train_target_start": self.train_target_start.strftime("%Y-%m-%d"),
            "train_target_end": self.train_target_end.strftime("%Y-%m-%d"),
            "val_target_start": (
                self.val_target_start.strftime("%Y-%m-%d") if self.val_target_start is not None else None
            ),
            "val_target_end": (
                self.val_target_end.strftime("%Y-%m-%d") if self.val_target_end is not None else None
            ),
            "train_target_days": int(self.train_target_days),
            "val_target_days": int(self.val_target_days),
            "train_samples": int(self.train_samples),
            "val_samples": int(self.val_samples),
        }


def _count_target_samples(target_days: int, out_len: int) -> int:
    return max(0, int(target_days) - int(out_len) + 1)


def resolve_window_target_protocol(
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    in_len: int,
    out_len: int,
    enable_val: bool,
    val_days: int = 7,
) -> WindowTargetProtocol:
    """Resolve mutually exclusive train/validation target-date ranges.

    Manuscript protocol (Section 3.3): within a 60-day optimization window the
    first 53 target days are used for optimization and the latest 7 target days
    for validation. When ``enable_val`` is true the training target range ends
    one day before the validation range starts, so the two *target-date* ranges
    are strictly disjoint. This is stronger than de-duplicating sample start
    indices: it guarantees no 3-day label window straddles the boundary. When
    ``enable_val`` is false the full window is used for training.

    Raises ValueError when the window is too short to reserve ``val_days``
    validation target days plus at least one training sample.
    """
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    if train_end < train_start:
        raise ValueError(f"train_end {train_end} precedes train_start {train_start}")

    if not enable_val:
        days = int((train_end - train_start).days) + 1
        return WindowTargetProtocol(
            train_target_start=train_start,
            train_target_end=train_end,
            val_target_start=None,
            val_target_end=None,
            train_target_days=days,
            val_target_days=0,
            train_samples=_count_target_samples(days, out_len),
            val_samples=0,
        )

    if val_days < out_len:
        raise ValueError(
            f"validation window of {val_days} target days is too short for out_len={out_len}"
        )
    val_target_end = train_end
    val_target_start = train_end - pd.Timedelta(days=val_days - 1)
    train_target_end = val_target_start - pd.Timedelta(days=1)
    train_target_days = int((train_target_end - train_start).days) + 1
    if train_target_days < out_len:
        raise ValueError(
            f"training target range {train_start:%Y-%m-%d}..{train_target_end:%Y-%m-%d} "
            f"({train_target_days} days) is too short for out_len={out_len} after "
            f"reserving {val_days} validation target days"
        )
    return WindowTargetProtocol(
        train_target_start=train_start,
        train_target_end=train_target_end,
        val_target_start=val_target_start,
        val_target_end=val_target_end,
        train_target_days=train_target_days,
        val_target_days=val_days,
        train_samples=_count_target_samples(train_target_days, out_len),
        val_samples=_count_target_samples(val_days, out_len),
    )


def _target_day_index_set(core, start_indices, in_len: int, out_len: int) -> set[int]:
    days: set[int] = set()
    for s in start_indices:
        for d in range(int(s) + in_len, int(s) + in_len + out_len):
            days.add(int(d))
    return days


def assert_window_protocol_disjoint(core, protocol: WindowTargetProtocol, in_len: int, out_len: int) -> dict:
    """Verify actual train/val sample target days do not overlap.

    Returns a record with resolved dates, actual sample counts and the overlap
    size (must be 0). Raises RuntimeError when target days overlap.
    """
    train_starts = core.find_valid_start_indices(
        target_start=protocol.train_target_start.strftime("%Y-%m-%d"),
        target_end=protocol.train_target_end.strftime("%Y-%m-%d"),
        in_len=in_len,
        out_len=out_len,
    )
    record = protocol.as_dict()
    record["train_samples_actual"] = int(len(train_starts))
    if protocol.val_target_start is None:
        record["val_samples_actual"] = 0
        record["target_day_overlap"] = 0
        return record
    val_starts = core.find_valid_start_indices(
        target_start=protocol.val_target_start.strftime("%Y-%m-%d"),
        target_end=protocol.val_target_end.strftime("%Y-%m-%d"),
        in_len=in_len,
        out_len=out_len,
    )
    record["val_samples_actual"] = int(len(val_starts))
    train_days = _target_day_index_set(core, train_starts, in_len, out_len)
    val_days = _target_day_index_set(core, val_starts, in_len, out_len)
    overlap = train_days & val_days
    record["target_day_overlap"] = int(len(overlap))
    if overlap:
        raise RuntimeError(
            f"train/val target days overlap for window protocol {record}: "
            f"{sorted(overlap)[:10]}"
        )
    return record


class BestValStateTracker:
    """Track the lowest-validation-loss model state as a CPU copy.

    The stored state is a detached CPU clone of ``state_dict`` so it never
    aliases the live (possibly GPU-resident) parameters. Non-persistent buffers
    (e.g. the y-transform scale/offset) and plain attributes such as
    ``condition_ewc`` are not part of ``state_dict`` and are therefore preserved
    when the best state is restored.
    """

    def __init__(self):
        self.best_val_loss: float | None = None
        self.best_epoch: int | None = None
        self.best_state: dict[str, torch.Tensor] | None = None

    def update(self, val_loss, state_dict, epoch: int | None = None) -> bool:
        try:
            value = float(val_loss)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        if self.best_val_loss is None or value < self.best_val_loss:
            self.best_val_loss = value
            self.best_epoch = epoch
            self.best_state = {
                k: v.detach().to("cpu", copy=True) for k, v in state_dict.items()
            }
            return True
        return False


class BestValCheckpointCallback(pl.Callback):
    """Keep the best-validation-loss state in memory (no disk checkpoints).

    Only acts when validation is enabled. ``val_loss`` is logged with
    ``sync_dist=True`` so every rank observes the same value and makes the same
    keep/discard decision, keeping DDP replicas consistent without any
    rank-specific branching.
    """

    def __init__(self, tracker: BestValStateTracker):
        super().__init__()
        self.tracker = tracker

    def on_validation_epoch_end(self, trainer, pl_module):
        if not getattr(trainer, "enable_validation", True):
            return
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return
        epoch = getattr(trainer, "current_epoch", None)
        self.tracker.update(val_loss, pl_module.state_dict(), epoch=epoch)


def _append_window_protocol_record(work_dir: str, window_idx: int, record: dict) -> None:
    """Append one JSON line describing a window's train/val target protocol."""
    path = os.path.join(work_dir, "rolling_window_protocols.jsonl")
    payload = {"window_idx": int(window_idx), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


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
    log_every_n_steps: int,
    dry_run_steps: int,
    accelerator: str,
    devices: str,
    precision: str,
    work_dir: str,
    refit_normalizer: bool,
    enable_val: bool,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    shm_cache_enabled: bool,
    shm_cache_dir: str,
    shm_cache_max_items: int,
    shm_cache_min_free_gb: float,
    condition_scheme: str,
    train_patches_per_sample: int,
    normalizer_stats_path: str,
    condition_ewc: ConditionAwareEWC | None,
    shm_cache_x_dtype: str = "float32",
    cuda_prefetch: bool = False,
    dataloader_multiprocessing_context: str = "",
    lit_model: PM25ForecastLitModule | None = None,
    logger_version: str | None = None,
    initial_state_path: str | None = None,
    seed: int | None = None,
    window_idx: int = 0,
    reviewer_method: str = "legacy",
    base_seed: int | None = None,
) -> PM25ForecastLitModule:
    train_start_str = train_start.strftime("%Y-%m-%d")
    train_end_str = train_end.strftime("%Y-%m-%d")
    if seed is not None:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)

    if refit_normalizer:
        normalizer.fit(
            core=core,
            in_len=in_len,
            out_len=out_len,
            train_target_start=train_start_str,
            train_target_end=train_end_str,
        )
        if normalizer_stats_path and _is_global_zero_process():
            normalizer.save(normalizer_stats_path)

    protocol = resolve_window_target_protocol(
        train_start=train_start,
        train_end=train_end,
        in_len=in_len,
        out_len=out_len,
        enable_val=enable_val,
    )
    train_target_start_str = protocol.train_target_start.strftime("%Y-%m-%d")
    train_target_end_str = protocol.train_target_end.strftime("%Y-%m-%d")
    if protocol.val_target_start is not None:
        val_start_str = protocol.val_target_start.strftime("%Y-%m-%d")
        val_end_str = protocol.val_target_end.strftime("%Y-%m-%d")
    else:
        # Validation disabled: keep a valid (unused) range for the datamodule.
        val_start_str = max(train_start, train_end - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        val_end_str = train_end_str

    if enable_val:
        val_starts = core.find_valid_start_indices(
            target_start=val_start_str,
            target_end=val_end_str,
            in_len=in_len,
            out_len=out_len,
        )
        if len(val_starts) == 0:
            raise RuntimeError(
                f"enable_val=True but no validation samples exist in "
                f"{val_start_str}..{val_end_str}; refusing to run a window without validation."
            )

    protocol_record = assert_window_protocol_disjoint(core, protocol, in_len, out_len)
    if _is_global_zero_process():
        print(
            f"[window_protocol] window={window_idx} "
            f"{json.dumps(protocol_record, ensure_ascii=False, sort_keys=True)}"
        )
        _append_window_protocol_record(work_dir, window_idx, protocol_record)

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
            condition_ewc=condition_ewc,
        )
        if initial_state_path and os.path.exists(initial_state_path):
            state_dict = torch.load(initial_state_path, map_location="cpu")
            lit_model.load_state_dict(state_dict)
    else:
        lit_model.condition_ewc = condition_ewc

    lit_model.set_y_transform_from_normalizer(normalizer)

    dm = PM25DataModule(
        core=core,
        in_len=in_len,
        out_len=out_len,
        train_target_start=train_target_start_str,
        train_target_end=train_target_end_str,
        val_target_start=val_start_str,
        val_target_end=val_end_str,
        batch_size=batch_size,
        num_workers=num_workers,
        patch_h=patch_h,
        patch_w=patch_w,
        normalizer=normalizer,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shm_cache_enabled=shm_cache_enabled,
        shm_cache_dir=shm_cache_dir,
        shm_cache_max_items=shm_cache_max_items,
        shm_cache_min_free_gb=shm_cache_min_free_gb,
        shm_cache_x_dtype=shm_cache_x_dtype,
        cuda_prefetch=cuda_prefetch,
        condition_scheme=condition_scheme,
        train_patches_per_sample=train_patches_per_sample,
        dataloader_multiprocessing_context=dataloader_multiprocessing_context,
    )

    clean_rev_method = str(reviewer_method or "legacy").strip().lower()
    if window_idx == 0 and clean_rev_method != "legacy" and _is_global_zero_process():
        audit_seed = base_seed if base_seed is not None else (seed if seed is not None else 0)
        _audit_initial_state_dict(
            work_dir=work_dir,
            state_dict=lit_model.state_dict(),
            model=model_name,
            method=clean_rev_method,
            seed=int(audit_seed),
            window_idx=window_idx,
        )

    tracker = BestValStateTracker() if enable_val else None
    callbacks = [BestValCheckpointCallback(tracker)] if tracker is not None else None
    trainer = build_trainer(
        work_dir=work_dir,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        dry_run_steps=dry_run_steps,
        enable_val=enable_val,
        logger_version=logger_version,
        callbacks=callbacks,
    )
    try:
        trainer.fit(lit_model, datamodule=dm)
    finally:
        dm.shutdown_workers()
        del trainer
        del dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if tracker is not None:
        if tracker.best_state is None:
            raise RuntimeError(
                "enable_val=True but no finite val_loss was recorded; refusing to "
                "report a successful window fit."
            )
        # Restore the lowest-validation-loss weights into the same lit_model so
        # condition_ewc, the non-persistent y-transform buffers and cross-window
        # parameter inheritance are preserved. The optimizer is rebuilt next
        # window, so its state is intentionally not restored.
        lit_model.load_state_dict(tracker.best_state)
        if _is_global_zero_process():
            print(
                f"[window_protocol] window={window_idx} restored best val_loss="
                f"{tracker.best_val_loss:.6g} at epoch={tracker.best_epoch}"
            )
    return lit_model




_SEASON4_NAMES = ["DJF", "MAM", "JJA", "SON"]
_MONTH_TO_SEASON4 = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def _season_idx_from_date(dt: pd.Timestamp) -> int:
    return _MONTH_TO_SEASON4[dt.month]


def _season_name_from_idx(idx: int) -> str:
    return _SEASON4_NAMES[idx]


def _param_group(key: str) -> str:
    k = key.lower()
    if "stem" in k or "input_proj" in k or "embed" in k:
        return "stem"
    if "down" in k or "encoder" in k or "enc_" in k:
        return "down_blocks"
    if "bottleneck" in k or "bridge" in k or "mid" in k or "neck" in k:
        return "bottleneck"
    if "up" in k or "decoder" in k or "dec_" in k:
        return "up_blocks"
    if "head" in k or "output" in k or "pred" in k or "out_proj" in k:
        return "head"
    if "spatial" in k or "spa_" in k or "s_att" in k or "s_conv" in k:
        return "spatial"
    if "temporal" in k or "tem_" in k or "t_att" in k or "t_conv" in k:
        return "temporal"
    return "other"


def _compute_drift_for_group(
    current_sd: dict,
    theta_star: dict,
    omega: dict,
    season_idx: int,
    group: str | None,
) -> tuple[float, float, float, int]:
    sum_sq = 0.0
    sum_fisher_sq = 0.0
    sum_omega = 0.0
    n_params = 0
    for key in theta_star:
        if group is not None and _param_group(key) != group:
            continue
        if key not in current_sd:
            continue
        if key not in omega:
            continue
        cur = current_sd[key].float().flatten()
        anchor = theta_star[key][season_idx].float().flatten()
        om = omega[key][season_idx].float().flatten()
        min_len = min(len(cur), len(anchor), len(om))
        if min_len == 0:
            continue
        cur = cur[:min_len]
        anchor = anchor[:min_len]
        om = om[:min_len]
        diff = cur - anchor
        sum_sq += (diff ** 2).sum().item()
        sum_fisher_sq += (om * diff ** 2).sum().item()
        sum_omega += om.sum().item()
        n_params += min_len
    return sum_sq, sum_fisher_sq, sum_omega, n_params


@torch.no_grad()
def _run_parameter_analysis(
    lit_model,
    bank: dict,
    win_idx: int,
    origin_date: pd.Timestamp,
    predict_target_start: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    analysis_dir: str,
    top_k: int,
    dtype_str: str,
    is_first_window: bool,
) -> None:
    theta_star = bank["theta_star"]
    omega = bank["omega"]
    season_idx = _season_idx_from_date(predict_target_start)
    season = _season_name_from_idx(season_idx)

    # Extract model state dict matching bank key format
    model_sd = {}
    for n, p in lit_model.model.named_parameters():
        model_sd[n] = p.data

    # -- Drift CSV --
    csv_path = os.path.join(analysis_dir, "window_parameter_drift.csv")
    groups = ["all", "stem", "down_blocks", "bottleneck", "up_blocks",
              "head", "spatial", "temporal", "other"]
    rows = []
    for g in groups:
        sum_sq, sum_fisher_sq, sum_omega, n_params = _compute_drift_for_group(
            model_sd, theta_star, omega, season_idx, None if g == "all" else g
        )
        if n_params == 0 and g != "all":
            continue
        uw_l2 = math.sqrt(sum_sq / n_params) if n_params > 0 else float("nan")
        fw_l2 = math.sqrt(sum_fisher_sq / (sum_omega + 1e-8)) if n_params > 0 else float("nan")
        rows.append({
            "window_idx": win_idx,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "origin_date": origin_date.strftime("%Y-%m-%d"),
            "predict_target_start": predict_target_start.strftime("%Y-%m-%d"),
            "season_idx": season_idx,
            "season": season,
            "group": g,
            "n_params": n_params,
            "unweighted_l2": uw_l2,
            "fisher_weighted_l2": fw_l2,
        })

    write_header = is_first_window or not os.path.exists(csv_path)
    with open(csv_path, "a" if not write_header else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "window_idx", "train_start", "train_end", "origin_date",
            "predict_target_start", "season_idx", "season", "group",
            "n_params", "unweighted_l2", "fisher_weighted_l2",
        ])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # -- PCA trajectory selection (only on first window) --
    traj_dir = os.path.join(analysis_dir, "trajectory_vectors")
    os.makedirs(traj_dir, exist_ok=True)
    sel_path = os.path.join(analysis_dir, "trajectory_selection.json")

    if is_first_window:
        # Select global top-K positions by omega mean across conditions without
        # materializing every parameter element as a Python tuple.
        candidate_scores: list[torch.Tensor] = []
        candidate_keys: list[str] = []
        candidate_indices: list[torch.Tensor] = []
        per_tensor_k = max(1, int(top_k))
        for key in omega:
            if key not in model_sd:
                continue
            cur_shape = model_sd[key].shape
            bank_shape = omega[key].shape[1:]  # (n_conditions, *param_shape)
            if cur_shape != bank_shape:
                continue
            om_mean = omega[key].float().mean(dim=0).flatten()  # (n_params,)
            if om_mean.numel() == 0:
                continue
            local_k = min(per_tensor_k, int(om_mean.numel()))
            scores, indices = torch.topk(om_mean, k=local_k, largest=True, sorted=False)
            candidate_scores.append(scores.cpu())
            candidate_indices.append(indices.cpu())
            candidate_keys.extend([key] * local_k)

        selected = []
        if candidate_scores:
            all_scores = torch.cat(candidate_scores, dim=0)
            global_k = min(max(1, int(top_k)), int(all_scores.numel()))
            top_scores, top_pos = torch.topk(all_scores, k=global_k, largest=True, sorted=True)

            offsets = []
            offset = 0
            for indices in candidate_indices:
                n = int(indices.numel())
                offsets.append((offset, offset + n, indices))
                offset += n

            for score, pos in zip(top_scores.tolist(), top_pos.tolist()):
                for start, end, indices in offsets:
                    if start <= pos < end:
                        selected.append((candidate_keys[pos], int(indices[pos - start].item()), float(score)))
                        break

        sel_info = {
            "top_k": top_k,
            "n_selected": len(selected),
            "entries": [
                {"key": s[0], "flat_idx": s[1], "omega_mean": s[2]}
                for s in selected
            ],
        }
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(sel_info, f, indent=2)

    # -- Trajectory vector for this window --
    if os.path.exists(sel_path):
        with open(sel_path, "r", encoding="utf-8") as f:
            sel_info = json.load(f)
        entries = sel_info["entries"]
        np_dtype = np.float32 if dtype_str == "float32" else np.float16

        values = np.zeros(len(entries), dtype=np_dtype)
        for i, e in enumerate(entries):
            key = e["key"]
            flat_idx = e["flat_idx"]
            if key in model_sd:
                param = model_sd[key].float().flatten()
                if flat_idx < len(param):
                    values[i] = param[flat_idx].item()

        vec_path = os.path.join(traj_dir, f"window_{win_idx:04d}.npz")
        np.savez(
            vec_path,
            window_idx=win_idx,
            values=values,
            predict_target_start=predict_target_start.strftime("%Y-%m-%d"),
            season_idx=season_idx,
            season=season,
        )


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
    log_every_n_steps: int,
    dry_run_steps: int,
    accelerator: str,
    devices: str,
    precision: str,
    rolling_refit_norm_every_n_windows: int,
    rolling_enable_val: bool,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    shm_cache_enabled: bool,
    shm_cache_dir: str,
    shm_cache_max_items: int,
    shm_cache_min_free_gb: float,
    condition_scheme: str,
    train_patches_per_sample: int,
    normalizer_stats_path: str,
    condition_ewc: ConditionAwareEWC | None,
    ca_ewc_bank_path: str,
    ca_ewc_transition_ranges: str,
    ca_ewc_transition_lambda_scale: float,
    ca_ewc_online_update_enabled: bool,
    ca_ewc_online_fisher_batches: int,
    ca_ewc_online_theta_alpha: float,
    ca_ewc_online_omega_alpha: float,
    ca_ewc_online_update_every_n_windows: int,
    ca_ewc_online_save_every_n_windows: int,
    train_start_date: str,
    predict_start_date: str,
    predict_end_date: str,
    shm_cache_x_dtype: str = "float32",
    cuda_prefetch: bool = False,
    dataloader_multiprocessing_context: str = "",
    train_first_window_only: bool = False,
    seed: int | None = None,
    resume_rolling: bool = False,
    start_window_idx: int = 0,
    parameter_analysis_enabled: bool = False,
    parameter_analysis_bank_path: str = "",
    parameter_analysis_dir: str = "",
    parameter_analysis_top_k: int = 50000,
    parameter_analysis_dtype: str = "float32",
    reviewer_method: str = "legacy",
    reviewer_fisher_variant: str = "empirical",
    reviewer_fisher_seed: int = 0,
):
    ensure_dir(output_dir)
    _setup_rank_logging(output_dir)
    is_global_zero = _is_global_zero_process()

    # Parameter analysis setup
    pa_bank = None
    pa_dir = ""
    if parameter_analysis_enabled:
        pa_dir = parameter_analysis_dir or os.path.join(output_dir, "parameter_analysis")
        ensure_dir(pa_dir)
        bank_src = parameter_analysis_bank_path
        if not bank_src and condition_ewc is not None:
            bank_src = getattr(condition_ewc, "bank_path", "") or ""
        if not bank_src:
            raise ValueError(
                "parameter_analysis_enabled but no bank path provided. "
                "Set parameter_analysis_bank_path or use condition_ewc."
            )
        if not os.path.exists(bank_src):
            raise FileNotFoundError(f"Parameter analysis bank not found: {bank_src}")
        pa_bank = torch.load(bank_src, map_location="cpu")
        if is_global_zero:
            print(f"[parameter_analysis] Loaded bank from {bank_src}")
            print(f"[parameter_analysis] Saving to {pa_dir}")

    runtime_device = _resolve_runtime_device(accelerator, devices)
    ca_ewc_transition_specs = _parse_ca_ewc_transition_ranges(ca_ewc_transition_ranges)
    ca_ewc_base_lambda = float(getattr(condition_ewc, "lambda_", 0.0)) if condition_ewc is not None else 0.0
    reviewer_meta = _derive_reviewer_metadata(
        model_name=model_name,
        seed=seed,
        condition_ewc=condition_ewc,
        condition_scheme=condition_scheme,
        ca_ewc_base_lambda=ca_ewc_base_lambda,
        reviewer_method=reviewer_method,
        reviewer_fisher_variant=reviewer_fisher_variant,
    )

    year_start = pd.Timestamp(predict_start_date)
    year_end = pd.Timestamp(predict_end_date)
    year_days = pd.date_range(year_start, year_end, freq="D")
    day_to_ordinal = {d: i for i, d in enumerate(year_days)}

    output_prefix = f"pm25_pred_{year_start.year}"
    out_dat_final = os.path.join(output_dir, f"{output_prefix}_rolling.int16.dat")
    state_file = os.path.join(output_dir, "rolling_state.json")
    normalizer_stats_path = normalizer_stats_path or os.path.join(output_dir, "normalizer_stats.npz")
    if (not normalizer.fitted) and os.path.exists(normalizer_stats_path):
        loaded_normalizer = PM25Normalizer.load(normalizer_stats_path)
        if loaded_normalizer.x_mode == normalizer.x_mode and loaded_normalizer.y_mode == normalizer.y_mode:
            normalizer = loaded_normalizer
            if is_global_zero:
                print(f"Loaded rolling normalizer stats from: {normalizer_stats_path}")
        elif is_global_zero:
            print(
                "Ignoring rolling normalizer stats with mode mismatch: "
                f"file=({loaded_normalizer.x_mode},{loaded_normalizer.y_mode}) "
                f"current=({normalizer.x_mode},{normalizer.y_mode})"
            )
    if normalizer.fitted and is_global_zero:
        normalizer.save(normalizer_stats_path)
    pred_mm = None
    state = _load_rolling_state(state_file) if resume_rolling else {}
    if resume_rolling and state:
        state_output = str(state.get("output_dat", ""))
        mismatches = []
        if state_output and os.path.abspath(state_output) != os.path.abspath(out_dat_final):
            mismatches.append(f"output_dat={state_output}")
        if state.get("predict_start_date") not in (None, predict_start_date):
            mismatches.append(f"predict_start_date={state.get('predict_start_date')}")
        if state.get("predict_end_date") not in (None, predict_end_date):
            mismatches.append(f"predict_end_date={state.get('predict_end_date')}")
        if mismatches:
            if is_global_zero:
                print(
                    "Ignoring rolling resume state that does not match this run: "
                    + "; ".join(mismatches)
                )
            state = {}
            resume_rolling = False
    if resume_rolling and state and not os.path.exists(out_dat_final):
        if is_global_zero:
            print(
                f"Warning: resume_rolling requested but {out_dat_final} is missing; "
                "starting from window 0 instead of using rolling_state.json."
            )
        state = {}
        resume_rolling = False
    resume_from_state = int(state.get("last_completed_window_idx", -1)) + 1 if state else 0
    first_window_idx = max(0, int(start_window_idx), resume_from_state)
    if is_global_zero:
        mode = "r+" if resume_rolling and os.path.exists(out_dat_final) else "w+"
        if mode == "w+" and os.path.exists(out_dat_final):
            os.remove(out_dat_final)
        pred_mm = np.memmap(out_dat_final, mode=mode, dtype=np.int16, shape=(len(year_days), core.height, core.width))

    pending_sum: Dict[pd.Timestamp, np.ndarray] = {}
    pending_cnt: Dict[pd.Timestamp, np.ndarray] = {}

    # Start origins earlier by in_len days so the first target can be year_start.
    # Example: in_len=5, predict_start=2019-01-01 => first origin=2018-12-27, targets 2019-01-01..2019-01-03.
    origin_start = year_start - pd.Timedelta(days=in_len)
    origin_end = year_end - pd.Timedelta(days=out_len - 1)
    origin_dates = pd.date_range(origin_start, origin_end, freq=f"{rolling_step_days}D")
    origin_date_set = {pd.Timestamp(d) for d in origin_dates}
    last_origin_for_day: Dict[pd.Timestamp, pd.Timestamp] = {}
    for day in year_days:
        candidate_origins = [
            pd.Timestamp(day) - pd.Timedelta(days=in_len + horizon)
            for horizon in range(out_len)
        ]
        valid_origins = [origin for origin in candidate_origins if origin in origin_date_set]
        if valid_origins:
            last_origin_for_day[pd.Timestamp(day)] = max(valid_origins)
    day_to_pm25_idx = {pd.Timestamp(d): i for i, d in enumerate(core.pm25_time)}
    last_win_idx = -1

    lit_model: PM25ForecastLitModule | None = None
    model_file = os.path.join(output_dir, "rolling_last.ckpt")
    daily_metrics_csv = os.path.join(output_dir, "rolling_daily_metrics.csv")
    windows_csv = os.path.join(output_dir, "rolling_windows.csv")
    reviewer_csv = os.path.join(output_dir, "reviewer_daily_horizon_metrics.csv")
    reviewer_done: set[tuple[str, str, str]] = set()
    if is_global_zero:
        write_headers = not resume_rolling or not os.path.exists(daily_metrics_csv)
        with open(daily_metrics_csv, "a" if resume_rolling else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_headers:
                writer.writerow(["date", "mae", "rmse", "valid_pixels", "window_idx_finalized_by"])
        write_headers = not resume_rolling or not os.path.exists(windows_csv)
        with open(windows_csv, "a" if resume_rolling else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_headers:
                writer.writerow(
                    [
                        "window_idx",
                        "train_start",
                        "train_end",
                        "origin_date",
                        "predict_target_start",
                        "predict_target_end",
                        "log_dir",
                    ]
                )
        if not os.path.exists(reviewer_csv) or os.path.getsize(reviewer_csv) == 0:
            with open(reviewer_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model",
                        "method",
                        "seed",
                        "ewc_lambda",
                        "seasonal_prior",
                        "fisher_variant",
                        "origin_date",
                        "forecast_date",
                        "horizon",
                        "mae",
                        "rmse",
                        "r2",
                        "bias",
                        "valid_pixels",
                        "window_idx",
                    ]
                )
                f.flush()
        else:
            _validate_reviewer_csv_metadata(reviewer_csv, reviewer_meta)
        reviewer_done = _read_reviewer_done_keys(reviewer_csv)
    _distributed_barrier()

    run_ok = False
    try:
        for win_idx, origin_date in enumerate(origin_dates):
            if win_idx < first_window_idx:
                continue
            last_win_idx = int(win_idx)
            train_start, train_end = _build_window_bounds(
                origin_date, rolling_train_window_days, in_len
            )
            if train_start < pd.Timestamp(train_start_date):
                train_start = pd.Timestamp(train_start_date)

            if condition_ewc is not None:
                # Keep the same normalization used for CA-EWC bank construction.
                # Re-fitting to per-window stats changes the input/output scaling and makes
                # EWC anchors (theta*) inconsistent with the current parameterization.
                refit_normalizer = False
            elif rolling_refit_norm_every_n_windows <= 0:
                refit_normalizer = not normalizer.fitted
            else:
                refit_normalizer = (win_idx % rolling_refit_norm_every_n_windows == 0) or (not normalizer.fitted)

            if condition_ewc is not None:
                pred_target_start = origin_date + pd.Timedelta(days=in_len)
                lambda_scale = _ca_ewc_window_lambda_scale(
                    pred_target_start,
                    out_len,
                    ca_ewc_transition_specs,
                    ca_ewc_transition_lambda_scale,
                )
                condition_ewc.lambda_ = ca_ewc_base_lambda * lambda_scale
                if is_global_zero and lambda_scale != 1.0:
                    pred_target_end = pred_target_start + pd.Timedelta(days=out_len - 1)
                    print(
                        f"[ca_ewc_transition] window={win_idx} targets="
                        f"{pred_target_start:%Y-%m-%d}:{pred_target_end:%Y-%m-%d} "
                        f"lambda={condition_ewc.lambda_:.6g} scale={lambda_scale:.6g}"
                    )

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
                log_every_n_steps=log_every_n_steps,
                dry_run_steps=dry_run_steps,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                work_dir=output_dir,
                refit_normalizer=refit_normalizer,
                enable_val=rolling_enable_val,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                shm_cache_enabled=shm_cache_enabled,
                shm_cache_dir=shm_cache_dir,
                shm_cache_max_items=shm_cache_max_items,
                shm_cache_min_free_gb=shm_cache_min_free_gb,
                shm_cache_x_dtype=shm_cache_x_dtype,
                cuda_prefetch=cuda_prefetch,
                dataloader_multiprocessing_context=dataloader_multiprocessing_context,
                condition_scheme=condition_scheme,
                train_patches_per_sample=train_patches_per_sample,
                normalizer_stats_path=normalizer_stats_path,
                condition_ewc=condition_ewc,
                lit_model=lit_model,
                logger_version=f"window_{win_idx:04d}",
                initial_state_path=model_file if resume_rolling and lit_model is None else None,
                seed=(seed + win_idx) if seed is not None else None,
                window_idx=win_idx,
                reviewer_method=reviewer_method,
                base_seed=seed,
            )
            lit_model = lit_model.to(runtime_device)
            if is_global_zero:
                with open(windows_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    pred_start = (origin_date + pd.Timedelta(days=in_len)).strftime("%Y-%m-%d")
                    pred_end = (origin_date + pd.Timedelta(days=in_len + out_len - 1)).strftime("%Y-%m-%d")
                    writer.writerow(
                        [
                            int(win_idx),
                            train_start.strftime("%Y-%m-%d"),
                            train_end.strftime("%Y-%m-%d"),
                            origin_date.strftime("%Y-%m-%d"),
                            pred_start,
                            pred_end,
                            os.path.join(output_dir, "rolling_logs", f"window_{win_idx:04d}"),
                        ]
                    )

            update_every = max(1, int(ca_ewc_online_update_every_n_windows))
            should_update_ewc = ((win_idx + 1) % update_every) == 0
            if (
                is_global_zero
                and condition_ewc is not None
                and ca_ewc_online_update_enabled
                and should_update_ewc
            ):
                updated = condition_ewc.online_update_from_window(
                    model=lit_model.model,
                    core=core,
                    normalizer=normalizer,
                    in_len=in_len,
                    out_len=out_len,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    window_target_start=train_start.strftime("%Y-%m-%d"),
                    window_target_end=train_end.strftime("%Y-%m-%d"),
                    batch_size=batch_size,
                    shm_cache_enabled=shm_cache_enabled,
                    shm_cache_dir=shm_cache_dir,
                    shm_cache_max_items=shm_cache_max_items,
                    shm_cache_min_free_gb=shm_cache_min_free_gb,
                    shm_cache_x_dtype=shm_cache_x_dtype,
                    fisher_batches=ca_ewc_online_fisher_batches,
                    theta_alpha=ca_ewc_online_theta_alpha,
                    omega_alpha=ca_ewc_online_omega_alpha,
                )
                save_every = max(1, int(ca_ewc_online_save_every_n_windows))
                if updated > 0 and ((win_idx + 1) % save_every == 0):
                    condition_ewc.save_bank(ca_ewc_bank_path)

            if is_global_zero:
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

                # Reviewer per-horizon metrics: raw y_full[h] vs truth for this
                # origin, appended immediately (independent of daily aggregation).
                for h in range(out_len):
                    day = pd.Timestamp(target_days[h])
                    if day < year_start or day > year_end:
                        continue
                    day_idx = day_to_pm25_idx.get(day)
                    if day_idx is None:
                        continue
                    key = (origin_date, day.strftime("%Y-%m-%d"), f"Day{h + 1}")
                    if key in reviewer_done:
                        continue
                    day_true = core.get_pm25_day(int(day_idx))
                    mae, rmse, r2, bias, valid_pixels = _horizon_metrics(y_full[h], day_true)
                    with open(reviewer_csv, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                reviewer_meta["model"],
                                reviewer_meta["method"],
                                reviewer_meta["seed"],
                                reviewer_meta["ewc_lambda"],
                                reviewer_meta["seasonal_prior"],
                                reviewer_meta["fisher_variant"],
                                key[0],
                                key[1],
                                key[2],
                                mae,
                                rmse,
                                r2,
                                bias,
                                valid_pixels,
                                int(win_idx),
                            ]
                        )
                        f.flush()
                    reviewer_done.add(key)

                # Finalize a day once its last scheduled contributing origin has run.
                # With rolling_step_days > 1, not every potential origin exists; assuming
                # daily origins leaves later horizons in memory and loses them on resume.
                to_finalize = [
                    d for d in pending_sum.keys()
                    if last_origin_for_day.get(d, origin_date) <= origin_date
                ]
                for d in sorted(to_finalize):
                    day_pred = pending_sum[d] / np.clip(pending_cnt[d], 1e-6, None)
                    if pred_mm is None:
                        raise RuntimeError("pred_mm is unexpectedly None on global rank 0.")
                    pred_mm[day_to_ordinal[d]] = _to_int16_grid(day_pred)
                    day_idx = day_to_pm25_idx.get(d)
                    if day_idx is not None:
                        day_true = core.get_pm25_day(int(day_idx))
                        mae, rmse, valid_pixels = _daily_mae_rmse(day_pred, day_true)
                        with open(daily_metrics_csv, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([d.strftime("%Y-%m-%d"), mae, rmse, valid_pixels, int(win_idx)])
                    del pending_sum[d]
                    del pending_cnt[d]
                if to_finalize:
                    # Persist finalized day-level predictions for real-time inspection.
                    pred_mm.flush()

                lit_model.to("cpu")
                torch.save(lit_model.state_dict(), model_file)
                _write_rolling_state(
                    state_file,
                    {
                        "status": "running",
                        "last_completed_window_idx": int(win_idx),
                        "last_completed_origin_date": origin_date.strftime("%Y-%m-%d"),
                        "predict_start_date": predict_start_date,
                        "predict_end_date": predict_end_date,
                        "output_dat": out_dat_final,
                    },
                )

                # Parameter analysis
                if pa_bank is not None and is_global_zero:
                    predict_target_start_date = origin_date + pd.Timedelta(days=in_len)
                    _run_parameter_analysis(
                        lit_model=lit_model,
                        bank=pa_bank,
                        win_idx=win_idx,
                        origin_date=origin_date,
                        predict_target_start=predict_target_start_date,
                        train_start=train_start,
                        train_end=train_end,
                        analysis_dir=pa_dir,
                        top_k=parameter_analysis_top_k,
                        dtype_str=parameter_analysis_dtype,
                        is_first_window=(win_idx == first_window_idx),
                    )

            _distributed_barrier()

            if train_first_window_only:
                break

        if is_global_zero:
            for d in sorted(pending_sum.keys()):
                day_pred = pending_sum[d] / np.clip(pending_cnt[d], 1e-6, None)
                if pred_mm is None:
                    raise RuntimeError("pred_mm is unexpectedly None on global rank 0.")
                pred_mm[day_to_ordinal[d]] = _to_int16_grid(day_pred)
                day_idx = day_to_pm25_idx.get(d)
                if day_idx is not None:
                    day_true = core.get_pm25_day(int(day_idx))
                    mae, rmse, valid_pixels = _daily_mae_rmse(day_pred, day_true)
                    with open(daily_metrics_csv, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([d.strftime("%Y-%m-%d"), mae, rmse, valid_pixels, int(last_win_idx)])

            pred_mm.flush()
            _write_rolling_state(
                state_file,
                {
                    "status": "completed",
                    "last_completed_window_idx": int(last_win_idx),
                    "predict_start_date": predict_start_date,
                    "predict_end_date": predict_end_date,
                    "output_dat": out_dat_final,
                },
            )
        run_ok = True
    except Exception:
        if is_global_zero:
            _write_rolling_state(
                state_file,
                {
                    "status": "failed",
                    "last_completed_window_idx": int(last_win_idx),
                    "predict_start_date": predict_start_date,
                    "predict_end_date": predict_end_date,
                    "output_dat": out_dat_final,
                    "error": traceback.format_exc(),
                },
            )
        raise
    finally:
        if pred_mm is not None:
            try:
                pred_mm.flush()
            except Exception:
                pass
            del pred_mm
        if is_global_zero and (not run_ok):
            print(f"Run failed; preserving partial output file if present: {out_dat_final}")

    if is_global_zero and condition_ewc is not None and ca_ewc_online_update_enabled:
        condition_ewc.save_bank(ca_ewc_bank_path)
    _distributed_barrier()
