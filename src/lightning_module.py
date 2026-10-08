from typing import Any, Dict

import pytorch_lightning as pl
import torch

from .adapters import build_model_adapter
from .reg.condition_aware_ewc import ConditionAwareEWC


class PM25ForecastLitModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        in_len: int,
        out_len: int,
        patch_h: int,
        patch_w: int,
        in_channels: int,
        lr: float,
        weight_decay: float,
        model_kwargs: Dict[str, Any],
        condition_ewc: ConditionAwareEWC | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["condition_ewc"])
        self.model = build_model_adapter(
            model_name=model_name,
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            in_channels=in_channels,
            model_kwargs=model_kwargs,
        )
        self.condition_ewc = condition_ewc
        self.register_buffer("y_raw_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("y_raw_offset", torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def set_y_transform_from_normalizer(self, normalizer) -> None:
        if normalizer.y_mode == "none" or normalizer.y_p1 is None or normalizer.y_p2 is None:
            scale = 1.0
            offset = 0.0
        else:
            y_p1 = float(normalizer.y_p1.reshape(-1)[0])
            y_p2 = float(normalizer.y_p2.reshape(-1)[0])
            if normalizer.y_mode in ["zscore", "minmax_01"]:
                scale = y_p2
                offset = y_p1
            elif normalizer.y_mode == "minmax_m11":
                scale = 0.5 * y_p2
                offset = y_p1 + 0.5 * y_p2
            else:
                raise ValueError(f"Unsupported y_mode={normalizer.y_mode}")
        self.y_raw_scale = torch.tensor(scale, dtype=torch.float32, device=self.y_raw_scale.device)
        self.y_raw_offset = torch.tensor(offset, dtype=torch.float32, device=self.y_raw_offset.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _to_raw_y(self, y: torch.Tensor) -> torch.Tensor:
        scale = self.y_raw_scale.to(device=y.device, dtype=y.dtype)
        offset = self.y_raw_offset.to(device=y.device, dtype=y.dtype)
        return y * scale + offset

    @staticmethod
    def _masked_l1_mse(y_hat: torch.Tensor, y: torch.Tensor, y_valid: torch.Tensor | None = None):
        # Validity must be decided in raw PM2.5 space before normalization.
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
            # A no-valid-target batch should contribute no supervised gradient.
            z = y_hat.sum() * 0.0
            l1 = z
            mse = z
        return l1, mse

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        if x.is_floating_point() and x.dtype != torch.float32 and not torch.is_autocast_enabled():
            x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(batch["y"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        y_valid = batch.get("y_valid")
        if y_valid is not None:
            y_valid = y_valid.to(dtype=torch.bool)
            valid_ratio = y_valid.float().mean()
            self.log("train_valid_ratio", valid_ratio, prog_bar=False, on_epoch=True, sync_dist=True)
        y_hat = torch.nan_to_num(self(x), nan=0.0, posinf=0.0, neginf=0.0)
        l1, mse = self._masked_l1_mse(y_hat, y, y_valid=y_valid)
        supervised_loss = l1 + mse
        loss = supervised_loss
        self.log("train_supervised_loss", supervised_loss, prog_bar=False, on_epoch=True, sync_dist=True)
        raw_l1, _ = self._masked_l1_mse(self._to_raw_y(y_hat), self._to_raw_y(y), y_valid=y_valid)
        self.log("train_mae_pm25", raw_l1, prog_bar=False, on_epoch=True, sync_dist=True)
        if self.condition_ewc is not None:
            if "condition_id" not in batch:
                raise KeyError("Batch missing 'condition_id' required by Condition-Aware EWC.")
            ewc_loss = self.condition_ewc.penalty(self.model, batch["condition_id"])
            loss = loss + ewc_loss
            ewc_ratio = ewc_loss / supervised_loss.detach().clamp_min(1e-12)
            self.log("train_ewc_loss", ewc_loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("train_ewc_to_supervised", ewc_ratio, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        if x.is_floating_point() and x.dtype != torch.float32 and not torch.is_autocast_enabled():
            x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(batch["y"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        y_valid = batch.get("y_valid")
        if y_valid is not None:
            y_valid = y_valid.to(dtype=torch.bool)
            valid_ratio = y_valid.float().mean()
            self.log("val_valid_ratio", valid_ratio, prog_bar=False, on_epoch=True, sync_dist=True)
        y_hat = torch.nan_to_num(self(x), nan=0.0, posinf=0.0, neginf=0.0)
        l1, mse = self._masked_l1_mse(y_hat, y, y_valid=y_valid)
        # Validation prediction loss mirrors the supervised training objective
        # (masked L1 + MSE) and deliberately excludes the EWC penalty. It is the
        # quantity used to select the best checkpoint within a rolling window.
        val_loss = l1 + mse
        mae = l1
        rmse = torch.sqrt(mse)
        raw_l1, raw_mse = self._masked_l1_mse(self._to_raw_y(y_hat), self._to_raw_y(y), y_valid=y_valid)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mae", mae, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_rmse", rmse, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mae_pm25", raw_l1, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_rmse_pm25", torch.sqrt(raw_mse), prog_bar=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Cosine annealing must span the real number of epochs for this rolling
        # window (manuscript: 20 epochs per cycle). Fall back to 50 when the
        # module is used outside a Trainer (e.g. bank construction) so existing
        # entry points keep working unchanged.
        t_max = 50
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            max_epochs = getattr(trainer, "max_epochs", None)
            if max_epochs is not None and int(max_epochs) > 0:
                t_max = int(max_epochs)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t_max))
        return {"optimizer": opt, "lr_scheduler": sch}
