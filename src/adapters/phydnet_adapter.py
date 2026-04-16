import torch
from torch import nn

from src.model.phydnet.models import ConvLSTM, EncoderRNN, PhyCell

from .base import PM25ModelAdapter


class PhyDNetAdapter(PM25ModelAdapter):
    def __init__(
        self,
        in_len: int,
        out_len: int,
        patch_h: int,
        patch_w: int,
        in_channels: int,
    ):
        super().__init__(in_len, out_len, patch_h, patch_w, in_channels)
        self.channel_proj = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.down = nn.AdaptiveAvgPool2d((64, 64))
        self.up = nn.Upsample(size=(patch_h, patch_w), mode="bilinear", align_corners=False)

        # These settings match the canonical PhyDNet implementation shape assumptions.
        self.phycell = PhyCell(
            input_shape=(16, 16),
            input_dim=64,
            F_hidden_dims=[49],
            n_layers=1,
            kernel_size=(7, 7),
            device=torch.device("cpu"),
        )
        self.convcell = ConvLSTM(
            input_shape=(16, 16),
            input_dim=64,
            hidden_dims=[128, 128, 64],
            n_layers=3,
            kernel_size=(3, 3),
            device=torch.device("cpu"),
        )
        self.model = EncoderRNN(self.phycell, self.convcell)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        self.phycell.device = device
        self.convcell.device = device

        btchw = x.permute(0, 1, 4, 2, 3).contiguous()
        b, t, c, h, w = btchw.shape
        flat = btchw.reshape(b * t, c, h, w)
        flat = self.channel_proj(flat)
        x_1ch = flat.reshape(b, t, 1, h, w)

        cur = None
        for i in range(self.in_len):
            frame = self.down(x_1ch[:, i])
            _, _, cur, _, _ = self.model(frame, first_timestep=(i == 0), decoding=False)

        preds = []
        for _ in range(self.out_len):
            _, _, cur, _, _ = self.model(cur, first_timestep=False, decoding=True)
            preds.append(self.up(cur))

        out = torch.stack(preds, dim=1)  # (B,T_out,1,H,W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        return out
