from __future__ import annotations

import fnmatch
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from ..data import MultiSourcePM25Core, PM25WindowDataset
from ..normalization import PM25Normalizer
from .conditions import condition_ids_for_start_indices, condition_names_for_scheme


BANK_COMPATIBILITY_KEYS = frozenset({
    "model_name",
    "model_kwargs",
    "in_channels",
    "in_len",
    "out_len",
    "patch_h",
    "patch_w",
    "parameter_schema_hash",
    "trainable_parameter_count",
    "normalizer_x_mode",
    "normalizer_y_mode",
    "normalizer_fingerprint",
    "condition_scheme",
    "ca_ewc_exclude_param_patterns",
    "ca_ewc_bank_train_start",
    "ca_ewc_bank_train_end",
    "ca_ewc_bank_dtype",
    "shm_cache_x_dtype",
    "ca_ewc_offline_lr",
    "ca_ewc_offline_weight_decay",
    "ca_ewc_offline_max_epochs",
    "ca_ewc_offline_max_steps_per_condition",
    "ca_ewc_fisher_batches",
    "batch_size",
    "data_paths_hash",
    "bank_build_contract_version",
})

def _masked_l1_mse(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    y_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if y_valid is None:
        valid = (y > 0) & torch.isfinite(y) & torch.isfinite(y_hat)
    else:
        valid = y_valid.bool() & torch.isfinite(y) & torch.isfinite(y_hat)
    if torch.any(valid):
        y_hat_v = y_hat[valid]
        y_v = y[valid]
        l1 = torch.nn.functional.l1_loss(y_hat_v, y_v)
        mse = torch.nn.functional.mse_loss(y_hat_v, y_v)
    else:
        z = y_hat.sum() * 0.0
        l1 = z
        mse = z
    return l1, mse


def _amp_context(device: torch.device, precision):
    p = str(precision).lower()
    if device.type != "cuda":
        return nullcontext()
    if p in {"bf16-mixed", "bf16", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if p in {"16-mixed", "16", "fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _loss_for_batch(model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    x = torch.nan_to_num(batch["x"], nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
    y = torch.nan_to_num(batch["y"], nan=0.0, posinf=0.0, neginf=0.0).to(device, dtype=torch.float32, non_blocking=True)
    y_valid = batch.get("y_valid")
    if y_valid is not None:
        y_valid = y_valid.to(device=device, dtype=torch.bool)
    y_hat = torch.nan_to_num(model(x), nan=0.0, posinf=0.0, neginf=0.0)
    l1, mse = _masked_l1_mse(y_hat, y, y_valid=y_valid)
    return l1 + mse


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported bank_dtype={dtype_name}. Choose from float32, float16, bfloat16.")


class EWCMetadataMismatchError(ValueError):
    pass


def _metadata_repr(value: Any, max_len: int = 240) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _normalize_param_patterns(patterns: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not patterns:
        return ()
    return tuple(str(p).strip() for p in patterns if str(p).strip())


def _matches_any_param_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


_IMPORTANCE_VARIANTS = ("empirical", "shuffled", "uniform", "inverse")


def _reverse_condition_ranks(
    omega_dict: Dict[str, torch.Tensor],
    condition_indices: Sequence[int],
    target_dict: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    """Perform global rank reversal for specified conditions across all omega parameter tensors.

    For each condition index:
      1. Concatenate all parameter slices for this condition in bank.omega's stable parameter order.
      2. Check for NaN/Inf in source values; raise ValueError if found.
      3. Sort ascending stably: values, order = torch.sort(flat, ascending=True, stable=True).
      4. Invert ranks: inverse_flat = torch.empty_like(flat); inverse_flat[order] = values.flip(0).
      5. Check for NaN/Inf in result; raise ValueError if found.
      6. Write back inverted slices to their respective parameter tensors at condition index c_idx,
         strictly preserving value multiset, shape, dtype, and device. Absolute ban on 1/Omega.
    """
    if target_dict is None:
        result: Dict[str, torch.Tensor] = {
            name: torch.empty_like(t) for name, t in omega_dict.items()
        }
    else:
        result = target_dict

    param_names = list(omega_dict.keys())
    for c_idx in condition_indices:
        slices = [omega_dict[name][c_idx].reshape(-1) for name in param_names]
        if not slices:
            continue
        flat = torch.cat(slices, dim=0)

        if not torch.isfinite(flat).all():
            raise ValueError(f"Source omega contains NaN or Inf for condition index {c_idx}.")

        sorted_values, order = torch.sort(flat, descending=False, stable=True)
        inverse_flat = torch.empty_like(flat)
        inverse_flat[order] = sorted_values.flip(0)

        if not torch.isfinite(inverse_flat).all():
            raise ValueError(f"Result omega contains NaN or Inf for condition index {c_idx}.")

        offset = 0
        for name in param_names:
            orig_shape = omega_dict[name][c_idx].shape
            num_elem = orig_shape.numel()
            result[name][c_idx].copy_(
                inverse_flat[offset : offset + num_elem].reshape(orig_shape)
            )
            offset += num_elem

    return result


def _derive_importance_omega(
    bank: ConditionAwareEWCBank,
    variant: str,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Derive the runtime omega from the bank's empirical omega.

    - empirical: use the bank omega as-is (no copy, no mutation).
    - shuffled: deep-copy each tensor and deterministically shuffle its elements
      in place with a single CPU generator seeded by `seed`, preserving each
      tensor's value multiset (and dtype/shape).
    - uniform: ones_like of each tensor.
    - inverse: global rank reversal across all parameter tensors for each condition,
      strictly preserving the value multiset, shape, dtype, and device without mutation of bank.omega.

    The bank object and its tensors are never mutated, so one empirical bank can
    back multiple variants simultaneously.
    """
    variant = str(variant).lower()
    if variant not in _IMPORTANCE_VARIANTS:
        raise ValueError(
            f"Unsupported importance_variant={variant}. "
            f"Choose from {', '.join(_IMPORTANCE_VARIANTS)}."
        )
    if variant == "empirical":
        return bank.omega
    if variant == "uniform":
        return {name: torch.ones_like(t) for name, t in bank.omega.items()}
    if variant == "inverse":
        if not bank.omega:
            return {}
        num_conds = next(iter(bank.omega.values())).shape[0]
        for name, t in bank.omega.items():
            if t.shape[0] != num_conds:
                raise ValueError(
                    f"Condition dimension mismatch in bank.omega for {name}: "
                    f"expected {num_conds}, got {t.shape[0]}"
                )
        return _reverse_condition_ranks(bank.omega, list(range(num_conds)))

    # shuffled
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    derived: Dict[str, torch.Tensor] = {}
    for name, t in bank.omega.items():
        flat = t.detach().clone().reshape(-1)
        perm = torch.randperm(flat.numel(), generator=gen, device="cpu")
        derived[name] = flat[perm].reshape(t.shape)
    return derived


@dataclass
class ConditionAwareEWCBank:
    condition_scheme: str
    condition_names: list[str]
    theta_star: Dict[str, torch.Tensor]  # [C, ...]
    omega: Dict[str, torch.Tensor]  # [C, ...]
    metadata: Dict[str, Any] | None = None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "version": 2,
            "condition_scheme": self.condition_scheme,
            "condition_names": self.condition_names,
            "theta_star": self.theta_star,
            "omega": self.omega,
            "metadata": dict(self.metadata or {}),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "ConditionAwareEWCBank":
        payload = torch.load(path, map_location="cpu")
        return cls(
            condition_scheme=str(payload["condition_scheme"]),
            condition_names=list(payload["condition_names"]),
            theta_star=dict(payload["theta_star"]),
            omega=dict(payload["omega"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def metadata_mismatches(self, expected_metadata: Dict[str, Any]) -> list[str]:
        actual_metadata = self.metadata or {}
        if not actual_metadata:
            return ["bank metadata is missing"]

        mismatches = []
        for key in sorted(BANK_COMPATIBILITY_KEYS & set(expected_metadata.keys())):
            expected = expected_metadata[key]
            if key not in actual_metadata:
                mismatches.append(f"{key}: missing, expected={_metadata_repr(expected)}")
                continue
            actual = actual_metadata[key]
            if _metadata_repr(actual, max_len=100000) != _metadata_repr(expected, max_len=100000):
                mismatches.append(
                    f"{key}: actual={_metadata_repr(actual)}, expected={_metadata_repr(expected)}"
                )
        return mismatches


@dataclass
class _CachedConditionTensors:
    """Device-resident theta/omega for one (device, dtype, condition) plus the
    denominator scalar and its validity.

    The denominator is computed once with the exact same chunk boundaries and
    accumulation order as the uncached path, then cached as a device scalar and
    a Python bool, so later penalty calls need no per-parameter
    ``isfinite``/``item`` host synchronization.
    """

    theta: torch.Tensor
    omega: torch.Tensor
    den: torch.Tensor
    den_valid: bool


class ConditionAwareEWC:
    def __init__(
        self,
        bank: ConditionAwareEWCBank,
        lambda_: float,
        excluded_param_patterns: list[str] | tuple[str, ...] | None = None,
        precision: str | int = "32",
        importance_variant: str = "empirical",
        importance_seed: int = 0,
        device_cache_enabled: bool = False,
    ):
        if lambda_ < 0:
            raise ValueError("ca_ewc_lambda must be >= 0.")
        self.bank = bank
        self.lambda_ = float(lambda_)
        self.excluded_param_patterns = _normalize_param_patterns(excluded_param_patterns)
        self.precision = precision
        self.importance_variant = str(importance_variant).lower()
        self.importance_seed = int(importance_seed)
        self.device_cache_enabled = bool(device_cache_enabled)
        # Cache key: (str(device), dtype, condition_index) -> {param_name: tensors}.
        # The CPU bank stays authoritative; this only mirrors active conditions.
        self._device_cache: Dict[Tuple[str, torch.dtype, int], Dict[str, _CachedConditionTensors]] = {}
        self._refresh_runtime_importance()

    def _refresh_runtime_importance(
        self,
        updated_conditions: Sequence[int] | None = None,
    ) -> None:
        # Any omega refresh (full or condition-targeted) invalidates the device
        # cache for the affected conditions so stale device copies are never
        # reused. This is also the choke point reached after an online bank
        # update, which mutates both theta_star and omega in place.
        self._invalidate_device_cache(updated_conditions)
        if self.importance_variant == "inverse" and updated_conditions is not None:
            if hasattr(self, "_omega") and self._omega:
                _reverse_condition_ranks(
                    self.bank.omega,
                    condition_indices=updated_conditions,
                    target_dict=self._omega,
                )
                return
        self._omega = _derive_importance_omega(
            bank=self.bank,
            variant=self.importance_variant,
            seed=self.importance_seed,
        )

    def _is_excluded_param(self, name: str) -> bool:
        return _matches_any_param_pattern(name, self.excluded_param_patterns)

    def _invalidate_device_cache(
        self,
        condition_indices: Sequence[int] | None = None,
    ) -> None:
        """Drop cached device tensors for the given conditions.

        ``condition_indices=None`` clears the whole cache. Device and dtype are
        part of the cache key, so a device/dtype change never reuses an entry.
        """
        cache = getattr(self, "_device_cache", None)
        if not cache:
            return
        if condition_indices is None:
            cache.clear()
            return
        drop = {int(c) for c in condition_indices}
        for key in [k for k in cache.keys() if k[2] in drop]:
            del cache[key]

    def clear_device_cache(self) -> None:
        """Explicitly release every cached device tensor."""
        self._device_cache.clear()

    def _param_term_cached(
        self,
        p: torch.Tensor,
        name: str,
        c_idx: int,
        theta_cpu: torch.Tensor,
        omega_cpu: torch.Tensor,
        eps: torch.Tensor,
        chunk_elems: int = 262144,
    ) -> torch.Tensor:
        """Device-cached variant of ``_param_term_chunked``.

        theta/omega for (device, dtype, condition) are materialised on the
        compute device once and reused. The numerator is accumulated over the
        exact same chunk boundaries and in the same order as the uncached path,
        and the denominator is computed once with that same chunk order and
        cached as a device scalar plus a validity flag, so subsequent calls
        perform no per-parameter ``isfinite``/``item`` host sync.
        """
        device = p.device
        dtype = p.dtype
        key = (str(device), dtype, int(c_idx))
        entry = self._device_cache.get(key)
        if entry is None:
            entry = {}
            self._device_cache[key] = entry

        cached = entry.get(name)
        if cached is None:
            theta_gpu = theta_cpu.reshape(-1).to(device=device, dtype=dtype, non_blocking=True)
            omega_gpu = torch.clamp(
                omega_cpu.reshape(-1).to(device=device, dtype=dtype, non_blocking=True),
                min=0.0,
            )
            den = torch.zeros((), device=device, dtype=dtype)
            n = int(omega_gpu.numel())
            for start in range(0, n, chunk_elems):
                end = min(start + chunk_elems, n)
                den = den + omega_gpu[start:end].sum()
            den_valid = bool(torch.isfinite(den) and float(den.item()) > 0.0)
            cached = _CachedConditionTensors(
                theta=theta_gpu,
                omega=omega_gpu,
                den=den,
                den_valid=den_valid,
            )
            entry[name] = cached

        p_flat = p.reshape(-1)
        num = torch.zeros((), device=device, dtype=dtype)
        n = int(p_flat.numel())
        theta_flat = cached.theta
        omega_flat = cached.omega
        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            p_chunk = p_flat[start:end]
            theta_chunk = theta_flat[start:end]
            omega_chunk = omega_flat[start:end]
            diff = p_chunk - theta_chunk
            num = num + (omega_chunk * diff * diff).sum()
            del theta_chunk, omega_chunk, diff
        if cached.den_valid:
            return num / (cached.den + eps.to(dtype=dtype))
        return torch.zeros((), device=device, dtype=dtype)

    @staticmethod
    def _param_term_chunked(
        p: torch.Tensor,
        theta_cpu: torch.Tensor,
        omega_cpu: torch.Tensor,
        eps: torch.Tensor,
        chunk_elems: int = 262144,
    ) -> torch.Tensor:
        p_flat = p.reshape(-1)
        theta_flat = theta_cpu.reshape(-1)
        omega_flat = omega_cpu.reshape(-1)
        num = torch.zeros((), device=p.device, dtype=p.dtype)
        den = torch.zeros((), device=p.device, dtype=p.dtype)
        n = int(p_flat.numel())
        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            p_chunk = p_flat[start:end]
            theta_chunk = theta_flat[start:end].to(device=p.device, dtype=p.dtype, non_blocking=True)
            omega_chunk = torch.clamp(omega_flat[start:end].to(device=p.device, dtype=p.dtype, non_blocking=True), min=0.0)
            diff = p_chunk - theta_chunk
            num = num + (omega_chunk * diff * diff).sum()
            den = den + omega_chunk.sum()
            del theta_chunk, omega_chunk, diff
        if torch.isfinite(den) and float(den.item()) > 0.0:
            return num / (den + eps.to(dtype=p.dtype))
        return torch.zeros((), device=p.device, dtype=p.dtype)

    @staticmethod
    def _check_alpha(name: str, value: float) -> float:
        v = float(value)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
        return v

    def validate_model(self, model: torch.nn.Module) -> None:
        model_shapes = {
            n: tuple(p.shape)
            for n, p in model.named_parameters()
            if p.requires_grad and not self._is_excluded_param(n)
        }
        for name, shape in model_shapes.items():
            if name not in self.bank.theta_star or name not in self.bank.omega:
                raise KeyError(f"Parameter '{name}' missing in Condition-Aware EWC bank.")
            theta = self.bank.theta_star[name]
            omega = self.bank.omega[name]
            if theta.ndim < 1 or omega.ndim < 1:
                raise ValueError(f"Bank tensor rank invalid for parameter '{name}'.")
            if tuple(theta.shape[1:]) != shape or tuple(omega.shape[1:]) != shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{name}': "
                    f"model={shape}, theta={tuple(theta.shape[1:])}, omega={tuple(omega.shape[1:])}"
                )
            if theta.shape[0] != len(self.bank.condition_names) or omega.shape[0] != len(self.bank.condition_names):
                raise ValueError(f"Condition dimension mismatch for parameter '{name}'.")

    def penalty(self, model: torch.nn.Module, condition_ids: torch.Tensor) -> torch.Tensor:
        if condition_ids.numel() == 0:
            device = next(model.parameters()).device
            return torch.zeros((), device=device)

        device = next(model.parameters()).device

        cond = condition_ids.detach().view(-1).to(torch.long).cpu()
        num_conditions = len(self.bank.condition_names)
        valid = cond[(cond >= 0) & (cond < num_conditions)]
        if valid.numel() == 0:
            return torch.zeros((), device=device)

        counts = torch.bincount(valid, minlength=num_conditions).to(device=device, dtype=torch.float32)
        weights = counts / counts.sum()
        active_conditions = torch.nonzero(weights > 0, as_tuple=False).view(-1)

        total = torch.zeros((), device=device)
        eps = torch.tensor(1e-12, device=device, dtype=torch.float32)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if self._is_excluded_param(name):
                continue
            theta_store = self.bank.theta_star.get(name)
            omega_store = self._omega.get(name)
            if theta_store is None or omega_store is None:
                continue

            for c in active_conditions:
                c_idx = int(c.item())
                if self.device_cache_enabled:
                    param_term = self._param_term_cached(
                        p=p,
                        name=name,
                        c_idx=c_idx,
                        theta_cpu=theta_store[c_idx],
                        omega_cpu=omega_store[c_idx],
                        eps=eps,
                    )
                else:
                    # Use chunked transfer to avoid allocating full bank tensors on GPU.
                    param_term = self._param_term_chunked(
                        p=p,
                        theta_cpu=theta_store[c_idx],
                        omega_cpu=omega_store[c_idx],
                        eps=eps,
                    )
                total = total + weights[c_idx] * param_term

        return 0.5 * self.lambda_ * total

    def save_bank(self, path: str) -> None:
        self.bank.save(path)

    def online_update_from_window(
        self,
        *,
        model: torch.nn.Module,
        core: MultiSourcePM25Core,
        normalizer: PM25Normalizer,
        in_len: int,
        out_len: int,
        patch_h: int,
        patch_w: int,
        window_target_start: str,
        window_target_end: str,
        batch_size: int,
        shm_cache_enabled: bool,
        shm_cache_dir: str,
        shm_cache_max_items: int,
        shm_cache_min_free_gb: float,
        fisher_batches: int,
        theta_alpha: float,
        omega_alpha: float,
        shm_cache_x_dtype: str = "float32",
    ) -> int:
        if fisher_batches <= 0:
            raise ValueError("ca_ewc_online_fisher_batches must be > 0.")
        theta_alpha = self._check_alpha("ca_ewc_online_theta_alpha", theta_alpha)
        omega_alpha = self._check_alpha("ca_ewc_online_omega_alpha", omega_alpha)
        if (theta_alpha <= 0.0) and (omega_alpha <= 0.0):
            return 0

        starts_np = core.find_valid_start_indices(
            target_start=window_target_start,
            target_end=window_target_end,
            in_len=in_len,
            out_len=out_len,
        )
        if starts_np.size == 0:
            return 0

        cond_ids_np = condition_ids_for_start_indices(
            pm25_time=core.pm25_time,
            start_indices=starts_np,
            in_len=in_len,
            condition_scheme=self.bank.condition_scheme,
        )
        starts = torch.from_numpy(starts_np.astype("int64"))
        cond_ids = torch.from_numpy(cond_ids_np.astype("int64"))
        unique_conditions = torch.unique(cond_ids).tolist()

        device = next(model.parameters()).device
        was_training = model.training
        model.train()

        updated_count = 0
        updated_condition_indices = set()
        for c_raw in unique_conditions:
            c_idx = int(c_raw)
            if c_idx < 0 or c_idx >= len(self.bank.condition_names):
                continue

            cond_starts = starts[cond_ids == c_idx]
            if cond_starts.numel() == 0:
                continue

            loader = _build_loader_for_condition(
                core=core,
                start_indices=cond_starts,
                normalizer=normalizer,
                in_len=in_len,
                out_len=out_len,
                patch_h=patch_h,
                patch_w=patch_w,
                condition_scheme=self.bank.condition_scheme,
                batch_size=batch_size,
                shm_cache_enabled=shm_cache_enabled,
                shm_cache_dir=shm_cache_dir,
                shm_cache_max_items=shm_cache_max_items,
                shm_cache_min_free_gb=shm_cache_min_free_gb,
                shm_cache_x_dtype=self.bank.metadata.get("shm_cache_x_dtype", "float32"),
            )

            fisher_accum = {
                n: torch.zeros_like(p, device=device, dtype=torch.float32)
                for n, p in model.named_parameters()
                if p.requires_grad and not self._is_excluded_param(n)
            }
            fisher_count = 0
            for batch in loader:
                model.zero_grad(set_to_none=True)
                with _amp_context(device, self.precision):
                    loss = _loss_for_batch(model, batch, device=device)
                loss.backward()
                for n, p in model.named_parameters():
                    if not p.requires_grad or p.grad is None or self._is_excluded_param(n):
                        continue
                    fisher_accum[n] += p.grad.detach().to(torch.float32).pow(2)
                fisher_count += 1
                if fisher_count >= fisher_batches:
                    break

            if fisher_count <= 0:
                continue

            with torch.no_grad():
                for n, p in model.named_parameters():
                    if not p.requires_grad or self._is_excluded_param(n):
                        continue

                    theta_store = self.bank.theta_star[n]
                    omega_store = self.bank.omega[n]
                    theta_old = theta_store[c_idx].to(torch.float32)
                    omega_old = omega_store[c_idx].to(torch.float32)
                    theta_new = p.detach().cpu().to(torch.float32)
                    omega_new = (fisher_accum[n] / float(fisher_count)).detach().cpu().to(torch.float32)

                    if theta_alpha > 0.0:
                        theta_blend = (1.0 - theta_alpha) * theta_old + theta_alpha * theta_new
                        theta_store[c_idx].copy_(theta_blend.to(theta_store.dtype))
                    if omega_alpha > 0.0:
                        omega_blend = (1.0 - omega_alpha) * omega_old + omega_alpha * omega_new
                        omega_store[c_idx].copy_(omega_blend.to(omega_store.dtype))
            updated_count += 1
            updated_condition_indices.add(c_idx)

        if updated_count > 0:
            self._refresh_runtime_importance(updated_conditions=sorted(updated_condition_indices))

        if not was_training:
            model.eval()
        return updated_count


def _build_loader_for_condition(
    *,
    core: MultiSourcePM25Core,
    start_indices: torch.Tensor,
    normalizer: PM25Normalizer,
    in_len: int,
    out_len: int,
    patch_h: int,
    patch_w: int,
    condition_scheme: str,
    batch_size: int,
    shm_cache_enabled: bool,
    shm_cache_dir: str,
    shm_cache_max_items: int,
    shm_cache_min_free_gb: float,
    shm_cache_x_dtype: str = "float32",
    num_workers: int = 0,
    pin_memory: bool = False,
    multiprocessing_context: str = "",
) -> DataLoader:
    dataset = PM25WindowDataset(
        core=core,
        start_indices=start_indices.cpu().numpy(),
        in_len=in_len,
        out_len=out_len,
        patch_h=patch_h,
        patch_w=patch_w,
        random_patch=True,
        normalizer=normalizer,
        shm_cache_enabled=shm_cache_enabled,
        shm_cache_dir=shm_cache_dir,
        shm_cache_max_items=shm_cache_max_items,
        shm_cache_min_free_gb=shm_cache_min_free_gb,
        shm_cache_x_dtype=shm_cache_x_dtype,
        condition_scheme=condition_scheme,
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
    }
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["pin_memory"] = pin_memory
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2
        if multiprocessing_context:
            import multiprocessing
            loader_kwargs["multiprocessing_context"] = multiprocessing.get_context(multiprocessing_context)
    else:
        loader_kwargs["num_workers"] = 0
        loader_kwargs["pin_memory"] = False
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = None
    return DataLoader(dataset, **loader_kwargs)


def warm_condition_cache(
    *,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    in_len: int,
    out_len: int,
    patch_h: int,
    patch_w: int,
    train_target_start: str,
    train_target_end: str,
    condition_scheme: str,
    batch_size: int,
    shm_cache_enabled: bool,
    shm_cache_dir: str,
    shm_cache_max_items: int,
    shm_cache_min_free_gb: float,
    shm_cache_x_dtype: str = "float32",
    num_workers: int = 8,
    multiprocessing_context: str = "",
) -> None:
    """Iterate condition-grouped datasets to populate SHM cache with num_workers>0.

    Must be called BEFORE any CUDA context is initialized (no .to(cuda), no
    Trainer, no torch.cuda calls). After warm-up, the bank build can use
    num_workers=0 and hit warm cache.
    """
    if not shm_cache_enabled:
        return

    all_starts_np = core.find_valid_start_indices(
        target_start=train_target_start,
        target_end=train_target_end,
        in_len=in_len,
        out_len=out_len,
    )
    if all_starts_np.size == 0:
        return

    all_starts = torch.from_numpy(all_starts_np.astype("int64"))
    cond_ids_np = condition_ids_for_start_indices(
        pm25_time=core.pm25_time,
        start_indices=all_starts_np,
        in_len=in_len,
        condition_scheme=condition_scheme,
    )
    cond_ids = torch.from_numpy(cond_ids_np.astype("int64"))
    condition_names = condition_names_for_scheme(condition_scheme)

    total_cached = 0
    for c_idx in range(len(condition_names)):
        mask = cond_ids == c_idx
        cond_starts = all_starts[mask]
        if cond_starts.numel() == 0:
            continue

        loader = _build_loader_for_condition(
            core=core,
            start_indices=cond_starts,
            normalizer=normalizer,
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            condition_scheme=condition_scheme,
            batch_size=batch_size,
            shm_cache_enabled=shm_cache_enabled,
            shm_cache_dir=shm_cache_dir,
            shm_cache_max_items=shm_cache_max_items,
            shm_cache_min_free_gb=shm_cache_min_free_gb,
            shm_cache_x_dtype=shm_cache_x_dtype,
            num_workers=num_workers,
            pin_memory=False,
            multiprocessing_context=multiprocessing_context,
        )
        n_samples = len(loader.dataset)
        print(f"[ewc-warmup] condition {condition_names[c_idx]}: "
              f"{n_samples} samples, num_workers={num_workers}", flush=True)
        for _ in loader:
            total_cached += 1
        print(f"[ewc-warmup] condition {condition_names[c_idx]} done "
              f"({total_cached} total batches processed)", flush=True)

    print(f"[ewc-warmup] cache warm-up complete, {total_cached} batches "
          f"across {len(condition_names)} conditions", flush=True)


def build_condition_aware_ewc_bank(
    *,
    model: torch.nn.Module,
    core: MultiSourcePM25Core,
    normalizer: PM25Normalizer,
    in_len: int,
    out_len: int,
    patch_h: int,
    patch_w: int,
    train_target_start: str,
    train_target_end: str,
    condition_scheme: str,
    bank_path: str,
    offline_lr: float,
    offline_weight_decay: float,
    offline_max_epochs: int,
    offline_max_steps_per_condition: int,
    fisher_batches: int,
    batch_size: int,
    shm_cache_enabled: bool,
    shm_cache_dir: str,
    shm_cache_max_items: int,
    shm_cache_min_free_gb: float,
    bank_dtype: str,
    shm_cache_x_dtype: str = "float32",
    precision: str | int = "32",
    excluded_param_patterns: list[str] | tuple[str, ...] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> ConditionAwareEWCBank:
    if not normalizer.fitted:
        raise ValueError("Normalizer must be fitted before building Condition-Aware EWC bank.")
    if fisher_batches <= 0:
        raise ValueError("ca_ewc_fisher_batches must be > 0.")

    condition_names = condition_names_for_scheme(condition_scheme)
    save_dtype = _resolve_dtype(bank_dtype)
    excluded_patterns = _normalize_param_patterns(excluded_param_patterns)

    all_starts_np = core.find_valid_start_indices(
        target_start=train_target_start,
        target_end=train_target_end,
        in_len=in_len,
        out_len=out_len,
    )
    if all_starts_np.size == 0:
        raise ValueError("No valid windows found for Condition-Aware EWC bank construction.")

    all_starts = torch.from_numpy(all_starts_np.astype("int64"))
    cond_ids_np = condition_ids_for_start_indices(
        pm25_time=core.pm25_time,
        start_indices=all_starts_np,
        in_len=in_len,
        condition_scheme=condition_scheme,
    )
    cond_ids = torch.from_numpy(cond_ids_np.astype("int64"))

    device = next(model.parameters()).device
    base_state_cpu = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    was_training = model.training
    trainable_names = [
        n for n, p in model.named_parameters()
        if p.requires_grad and not _matches_any_param_pattern(n, excluded_patterns)
    ]
    theta_buf: Dict[str, list[torch.Tensor]] = {n: [] for n in trainable_names}
    omega_buf: Dict[str, list[torch.Tensor]] = {n: [] for n in trainable_names}

    try:
        for c_idx in range(len(condition_names)):
            mask = cond_ids == c_idx
            cond_starts = all_starts[mask]

            if cond_starts.numel() == 0:
                for n in trainable_names:
                    base_param = base_state_cpu[n].detach().to(torch.float32)
                    theta_buf[n].append(base_param)
                    omega_buf[n].append(torch.zeros_like(base_param))
                continue

            model.load_state_dict(base_state_cpu)
            model.to(device)
            model.train()
            model.zero_grad(set_to_none=True)

            optimizer = torch.optim.AdamW(model.parameters(), lr=offline_lr, weight_decay=offline_weight_decay)
            loader = _build_loader_for_condition(
                core=core,
                start_indices=cond_starts,
                normalizer=normalizer,
                in_len=in_len,
                out_len=out_len,
                patch_h=patch_h,
                patch_w=patch_w,
                condition_scheme=condition_scheme,
                batch_size=batch_size,
                shm_cache_enabled=shm_cache_enabled,
                shm_cache_dir=shm_cache_dir,
                shm_cache_max_items=shm_cache_max_items,
                shm_cache_min_free_gb=shm_cache_min_free_gb,
                shm_cache_x_dtype=shm_cache_x_dtype,
            )

            max_epochs = max(1, int(offline_max_epochs))
            max_steps = max(0, int(offline_max_steps_per_condition))
            n_batches = len(loader)
            print(f"[ewc-bank] condition={condition_names[c_idx]} phase=train "
                  f"batches={n_batches} epochs={max_epochs}", flush=True)
            step = 0
            stop = False
            import time as _time
            _t0 = _time.perf_counter()
            for _ in range(max_epochs):
                for batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    with _amp_context(device, precision):
                        loss = _loss_for_batch(model, batch, device=device)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    if step % 10 == 0 or step == n_batches:
                        _elapsed = _time.perf_counter() - _t0
                        print(f"[ewc-bank] condition={condition_names[c_idx]} phase=train "
                              f"step={step}/{n_batches} elapsed={_elapsed:.1f}s", flush=True)
                    if max_steps > 0 and step >= max_steps:
                        stop = True
                        break
                if stop:
                    break

            fisher_accum = {
                n: torch.zeros_like(p, device=device, dtype=torch.float32)
                for n, p in model.named_parameters()
                if p.requires_grad and not _matches_any_param_pattern(n, excluded_patterns)
            }
            fisher_count = 0
            model.train()
            print(f"[ewc-bank] condition={condition_names[c_idx]} phase=fisher "
                  f"batches={fisher_batches}", flush=True)
            _fisher_t0 = _time.perf_counter()
            for batch in loader:
                model.zero_grad(set_to_none=True)
                with _amp_context(device, precision):
                    loss = _loss_for_batch(model, batch, device=device)
                loss.backward()
                for n, p in model.named_parameters():
                    if (
                        not p.requires_grad
                        or p.grad is None
                        or _matches_any_param_pattern(n, excluded_patterns)
                    ):
                        continue
                    fisher_accum[n] += p.grad.detach().to(torch.float32).pow(2)
                fisher_count += 1
                if fisher_count % 10 == 0 or fisher_count == fisher_batches:
                    _elapsed = _time.perf_counter() - _fisher_t0
                    print(f"[ewc-bank] condition={condition_names[c_idx]} phase=fisher "
                          f"step={fisher_count}/{fisher_batches} elapsed={_elapsed:.1f}s", flush=True)
                if fisher_count >= fisher_batches:
                    break

            _cond_elapsed = _time.perf_counter() - _t0
            print(f"[ewc-bank] condition={condition_names[c_idx]} done "
                  f"train_steps={step} fisher_steps={fisher_count} "
                  f"elapsed={_cond_elapsed:.1f}s", flush=True)
            for n, p in model.named_parameters():
                if not p.requires_grad or _matches_any_param_pattern(n, excluded_patterns):
                    continue
                theta_buf[n].append(p.detach().cpu().to(torch.float32))
                if fisher_count > 0:
                    omega_buf[n].append((fisher_accum[n] / float(fisher_count)).detach().cpu())
                else:
                    omega_buf[n].append(torch.zeros_like(p.detach().cpu().to(torch.float32)))

            optimizer.zero_grad(set_to_none=True)
            del optimizer, fisher_accum
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        model.load_state_dict(base_state_cpu)
        model.to(device)
        model.train(was_training)
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    theta_star = {n: torch.stack(v, dim=0).to(save_dtype) for n, v in theta_buf.items()}
    omega = {n: torch.stack(v, dim=0).to(save_dtype) for n, v in omega_buf.items()}
    bank_metadata = dict(metadata or {})
    bank_metadata.setdefault("condition_scheme", condition_scheme)
    bank_metadata.setdefault("ca_ewc_bank_train_start", train_target_start)
    bank_metadata.setdefault("ca_ewc_bank_train_end", train_target_end)
    bank_metadata.setdefault("ca_ewc_bank_dtype", bank_dtype)
    bank_metadata.setdefault("bank_build_contract_version", 1)
    bank = ConditionAwareEWCBank(
        condition_scheme=condition_scheme,
        condition_names=condition_names,
        theta_star=theta_star,
        omega=omega,
        metadata=bank_metadata,
    )
    bank.save(bank_path)
    return bank


def load_condition_aware_ewc(
    *,
    model: torch.nn.Module,
    bank_path: str,
    lambda_: float,
    bank_dtype: str | None = None,
    precision: str | int = "32",
    expected_metadata: Dict[str, Any] | None = None,
    excluded_param_patterns: list[str] | tuple[str, ...] | None = None,
    importance_variant: str = "empirical",
    importance_seed: int = 0,
    device_cache_enabled: bool = False,
) -> ConditionAwareEWC:
    bank = ConditionAwareEWCBank.load(bank_path)
    if expected_metadata is not None:
        mismatches = bank.metadata_mismatches(expected_metadata)
        if mismatches:
            preview = "\n".join(f"- {m}" for m in mismatches[:12])
            if len(mismatches) > 12:
                preview += f"\n- ... {len(mismatches) - 12} more metadata mismatches"
            raise EWCMetadataMismatchError(
                f"Condition-Aware EWC bank metadata mismatch at {bank_path}:\n{preview}"
            )
    if bank_dtype is not None:
        target_dtype = _resolve_dtype(bank_dtype)
        bank.theta_star = {k: v.to(dtype=target_dtype) for k, v in bank.theta_star.items()}
        bank.omega = {k: v.to(dtype=target_dtype) for k, v in bank.omega.items()}
    regularizer = ConditionAwareEWC(
        bank=bank,
        lambda_=lambda_,
        excluded_param_patterns=excluded_param_patterns,
        precision=precision,
        importance_variant=importance_variant,
        importance_seed=importance_seed,
        device_cache_enabled=device_cache_enabled,
    )
    regularizer.validate_model(model)
    return regularizer
