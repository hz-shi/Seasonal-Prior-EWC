import torch

from src.model.unet import UNet2D

from .base import PM25ModelAdapter


class UNetAdapter(PM25ModelAdapter):
    def __init__(
        self,
        in_len: int,
        out_len: int,
        patch_h: int,
        patch_w: int,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(in_len, out_len, patch_h, patch_w, in_channels)
        self.model = UNet2D(
            in_channels=in_len * in_channels,
            out_channels=out_len,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project temporal and feature channels into the 2D UNet channel axis.
        b, t, h, w, c = x.shape
        if t != self.in_len:
            raise ValueError(f"UNetAdapter expected input length {self.in_len}, got {t}.")
        if c != self.in_channels:
            raise ValueError(f"UNetAdapter expected input channels {self.in_channels}, got {c}.")
        if (h, w) != (self.patch_h, self.patch_w):
            raise ValueError(f"UNetAdapter expected patch {(self.patch_h, self.patch_w)}, got {(h, w)}.")

        x_2d = x.permute(0, 1, 4, 2, 3).contiguous().reshape(b, t * c, h, w)
        y = self.model(x_2d)
        if y.shape != (b, self.out_len, h, w):
            raise RuntimeError(f"UNet output shape mismatch, got {tuple(y.shape)}.")
        return y[:, :, :, :, None].contiguous()
