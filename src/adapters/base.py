from abc import ABC, abstractmethod

import torch
from torch import nn


class PM25ModelAdapter(nn.Module, ABC):
    def __init__(self, in_len: int, out_len: int, patch_h: int, patch_w: int, in_channels: int):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B,T,H,W,C), Output: (B,T_out,H,W,1)."""
        raise NotImplementedError
