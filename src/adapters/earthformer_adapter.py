import torch

from src.model.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

from .base import PM25ModelAdapter


class EarthformerAdapter(PM25ModelAdapter):
    def __init__(
        self,
        in_len: int,
        out_len: int,
        patch_h: int,
        patch_w: int,
        in_channels: int,
        base_units: int = 64,
        num_heads: int = 4,
        enc_depth: list[int] | None = None,
        dec_depth: list[int] | None = None,
        downsample: int = 2,
        num_global_vectors: int = 0,
        z_init_method: str = "nearest_interp",
        initial_downsample_scale: int = 2,
        initial_downsample_conv_layers: int = 2,
        final_upsample_conv_layers: int = 2,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
    ):
        super().__init__(in_len, out_len, patch_h, patch_w, in_channels)
        enc_depth = enc_depth if enc_depth is not None else [2, 2]
        dec_depth = dec_depth if dec_depth is not None else [2, 2]
        self.model = CuboidTransformerModel(
            input_shape=(in_len, patch_h, patch_w, in_channels),
            target_shape=(out_len, patch_h, patch_w, 1),
            base_units=base_units,
            num_heads=num_heads,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            downsample=downsample,
            num_global_vectors=num_global_vectors,
            z_init_method=z_init_method,
            initial_downsample_scale=initial_downsample_scale,
            initial_downsample_conv_layers=initial_downsample_conv_layers,
            final_upsample_conv_layers=final_upsample_conv_layers,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
