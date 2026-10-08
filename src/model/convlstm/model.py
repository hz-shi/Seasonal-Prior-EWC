import torch
import torch.nn as nn
from typing import List, Tuple

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTMForecaster(nn.Module):
    def __init__(self, in_channels: int, in_len: int, out_len: int, hidden_dims: List[int], kernel_size: int):
        super().__init__()

        if not hidden_dims or any(d <= 0 for d in hidden_dims):
            raise ValueError(f"hidden_dims must be a non-empty list of positive integers, got {hidden_dims}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.in_channels = in_channels
        self.in_len = in_len
        self.out_len = out_len
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        self.input_proj = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1)
        self.feedback_proj = nn.Conv2d(1, hidden_dims[0], kernel_size=1)

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = hidden_dims[0] if i == 0 else hidden_dims[i-1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dims[i],
                kernel_size=kernel_size
            ))
        self.cells = nn.ModuleList(cell_list)

    def _init_hidden(self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dims[i], height, width, device=device, dtype=dtype)
            c = torch.zeros(batch_size, self.hidden_dims[i], height, width, device=device, dtype=dtype)
            states.append((h, c))
        return states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        states = self._init_hidden(b, h, w, x.device, x.dtype)

        # Encoding
        for step in range(self.in_len):
            x_t = x[:, step]
            current_input = self.input_proj(x_t)

            for layer_idx in range(self.num_layers):
                h_next, c_next = self.cells[layer_idx](current_input, states[layer_idx])
                states[layer_idx] = (h_next, c_next)
                current_input = h_next

        # Decoding
        predictions = []
        decoder_input = self.output_proj(states[-1][0])

        for step in range(self.out_len):
            predictions.append(decoder_input)
            if step < self.out_len - 1:
                current_input = self.feedback_proj(decoder_input)

                for layer_idx in range(self.num_layers):
                    h_next, c_next = self.cells[layer_idx](current_input, states[layer_idx])
                    states[layer_idx] = (h_next, c_next)
                    current_input = h_next

                decoder_input = self.output_proj(states[-1][0])

        # Stack predictions
        out = torch.stack(predictions, dim=1) # (B, T_out, 1, H, W)
        return out
