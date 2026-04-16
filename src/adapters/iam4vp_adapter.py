import torch
from torch import nn

from src.model.iam4vp.model import IAM4VP

from .base import PM25ModelAdapter


class IAM4VPAdapter(PM25ModelAdapter):
    def __init__(
        self,
        in_len: int,
        out_len: int,
        patch_h: int,
        patch_w: int,
        in_channels: int,
        hid_s: int = 64,
        hid_t: int = 512,
        n_s: int = 4,
        n_t: int = 6,
    ):
        super().__init__(in_len, out_len, patch_h, patch_w, in_channels)
        self.channel_proj = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.model = IAM4VP([in_len, 1, patch_h, patch_w], hid_S=hid_s, hid_T=hid_t, N_S=n_s, N_T=n_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IAM4VP expects (B,T,C,H,W) and currently has hardcoded behavior in upstream implementation.
        btchw = x.permute(0, 1, 4, 2, 3).contiguous()
        b, t, c, h, w = btchw.shape
        flat = btchw.reshape(b * t, c, h, w)
        flat = self.channel_proj(flat)
        x_1ch = flat.reshape(b, t, 1, h, w)

        pred_list = []
        outputs = []
        for step in range(self.out_len):
            t_embed = torch.full((b,), float(step * 100), device=x.device)
            y = self.model(x_1ch, y_raw=pred_list, t=t_embed)
            if y.ndim != 4:
                raise RuntimeError(f"IAM4VP output rank mismatch, got shape={tuple(y.shape)}")
            if y.shape[0] != b:
                raise RuntimeError(
                    "IAM4VP current implementation appears to force batch size 1. "
                    f"Got input batch={b}, output batch={y.shape[0]}."
                )
            pred_list.append(y)
            outputs.append(y)

        out = torch.stack(outputs, dim=1)  # (B,T_out,1,H,W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # (B,T_out,H,W,1)
        return out
