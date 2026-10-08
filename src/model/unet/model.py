from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _norm_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_norm_groups(out_channels), out_channels),
                nn.SiLU(inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"UNet depth must be >= 1, got {depth}.")
        if base_channels < 1:
            raise ValueError(f"UNet base_channels must be >= 1, got {base_channels}.")

        channels = [int(base_channels) * (2**i) for i in range(int(depth))]
        self.stem = ConvBlock(in_channels, channels[0], dropout=dropout)
        self.down_blocks = nn.ModuleList(
            DownBlock(channels[i - 1], channels[i], dropout=dropout)
            for i in range(1, len(channels))
        )

        bottleneck_in = channels[-1]
        bottleneck_out = channels[-1] * 2
        self.bottleneck = DownBlock(bottleneck_in, bottleneck_out, dropout=dropout)

        decoder_in_channels = [bottleneck_out] + list(reversed(channels[1:]))
        decoder_skip_channels = list(reversed(channels))
        decoder_out_channels = list(reversed(channels))
        self.up_blocks = nn.ModuleList(
            UpBlock(in_ch, skip_ch, out_ch, dropout=dropout)
            for in_ch, skip_ch, out_ch in zip(decoder_in_channels, decoder_skip_channels, decoder_out_channels)
        )
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.stem(x)
        skips.append(x)
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)
        return self.head(x)
