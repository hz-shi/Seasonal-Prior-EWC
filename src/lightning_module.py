from typing import Any, Dict

import pytorch_lightning as pl
import torch

from .adapters import build_model_adapter


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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model_adapter(
            model_name=model_name,
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            in_channels=in_channels,
            model_kwargs=model_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _masked_l1_mse(y_hat: torch.Tensor, y: torch.Tensor):
        # Treat non-positive PM2.5 targets as invalid/no-data for optimization.
        valid = (y > 0) & torch.isfinite(y) & torch.isfinite(y_hat)
        if torch.any(valid):
            y_hat_v = y_hat[valid]
            y_v = y[valid]
            l1 = torch.nn.functional.l1_loss(y_hat_v, y_v)
            mse = torch.nn.functional.mse_loss(y_hat_v, y_v)
        else:
            # Fallback avoids NaN when a batch has no valid pixels.
            y_hat_safe = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
            y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            l1 = torch.nn.functional.l1_loss(y_hat_safe, y_safe)
            mse = torch.nn.functional.mse_loss(y_hat_safe, y_safe)
        return l1, mse

    def training_step(self, batch, batch_idx):
        x = torch.nan_to_num(batch["x"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(batch["y"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        y_hat = torch.nan_to_num(self(x), nan=0.0, posinf=0.0, neginf=0.0)
        l1, mse = self._masked_l1_mse(y_hat, y)
        loss = l1 + mse
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.nan_to_num(batch["x"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(batch["y"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        y_hat = torch.nan_to_num(self(x), nan=0.0, posinf=0.0, neginf=0.0)
        l1, mse = self._masked_l1_mse(y_hat, y)
        mae = l1
        rmse = torch.sqrt(mse)
        self.log("val_mae", mae, prog_bar=True, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return {"optimizer": opt, "lr_scheduler": sch}
