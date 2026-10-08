import torch
from torch import nn

from src.model.convlstm.model import ConvLSTMForecaster
from .base import PM25ModelAdapter


class ConvLSTMAdapter(PM25ModelAdapter):
    def __init__(
        self,
        in_len: int,
        out_len: int,
        patch_h: int,
        patch_w: int,
        in_channels: int,
        hidden_dims: list,
        kernel_size: int,
    ):
        super().__init__(in_len, out_len, patch_h, patch_w, in_channels)
        self.model = ConvLSTMForecaster(
            in_channels=in_channels,
            in_len=in_len,
            out_len=out_len,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape expected by PM25ModelAdapter: (B,T,H,W,C)
        b, t, h, w, c = x.shape
        if t != self.in_len:
            raise ValueError(f"ConvLSTMAdapter expected input length {self.in_len}, got {t}.")
        if (h, w) != (self.patch_h, self.patch_w):
            raise ValueError(f"ConvLSTMAdapter expected patch {(self.patch_h, self.patch_w)}, got {(h, w)}.")
        if c != self.in_channels:
            raise ValueError(f"ConvLSTMAdapter expected channels {self.in_channels}, got {c}.")

        # Convert to expected forecaster input shape: (B,T,C,H,W)
        btchw = x.permute(0, 1, 4, 2, 3).contiguous()

        # Forecaster returns (B,T_out,1,H,W)
        out = self.model(btchw)

        # Output shape expected by PM25ModelAdapter: (B,T_out,H,W,1)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        return out
