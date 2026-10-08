from typing import Any, Dict

from .base import PM25ModelAdapter
from .earthformer_adapter import EarthformerAdapter
from .iam4vp_adapter import IAM4VPAdapter
from .phydnet_adapter import PhyDNetAdapter
from .unet_adapter import UNetAdapter
from .convlstm_adapter import ConvLSTMAdapter


def build_model_adapter(
    model_name: str,
    in_len: int,
    out_len: int,
    patch_h: int,
    patch_w: int,
    in_channels: int,
    model_kwargs: Dict[str, Any],
) -> PM25ModelAdapter:
    name = model_name.lower()
    if name == "earthformer":
        return EarthformerAdapter(
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            in_channels=in_channels,
            base_units=int(model_kwargs.get("base_units", 64)),
            num_heads=int(model_kwargs.get("num_heads", 4)),
            enc_depth=model_kwargs.get("earthformer_enc_depth", [2, 2]),
            dec_depth=model_kwargs.get("earthformer_dec_depth", [2, 2]),
            downsample=int(model_kwargs.get("earthformer_downsample", 2)),
            num_global_vectors=int(model_kwargs.get("earthformer_num_global_vectors", 0)),
            z_init_method=str(model_kwargs.get("earthformer_z_init_method", "nearest_interp")),
            initial_downsample_scale=int(model_kwargs.get("earthformer_initial_downsample_scale", 2)),
            initial_downsample_conv_layers=int(model_kwargs.get("earthformer_initial_downsample_conv_layers", 2)),
            final_upsample_conv_layers=int(model_kwargs.get("earthformer_final_upsample_conv_layers", 2)),
            attn_drop=float(model_kwargs.get("earthformer_attn_drop", 0.0)),
            proj_drop=float(model_kwargs.get("earthformer_proj_drop", 0.0)),
            ffn_drop=float(model_kwargs.get("earthformer_ffn_drop", 0.0)),
        )
    if name == "iam4vp":
        return IAM4VPAdapter(
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            in_channels=in_channels,
            hid_s=int(model_kwargs.get("hid_s", 64)),
            hid_t=int(model_kwargs.get("hid_t", 512)),
            n_s=int(model_kwargs.get("n_s", 4)),
            n_t=int(model_kwargs.get("n_t", 6)),
        )
    if name == "phydnet":
        return PhyDNetAdapter(
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            in_channels=in_channels,
        )
    if name == "convlstm":
        return ConvLSTMAdapter(
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            in_channels=in_channels,
            hidden_dims=model_kwargs.get("convlstm_hidden_dims", [32, 32]),
            kernel_size=int(model_kwargs.get("convlstm_kernel_size", 3)),
        )
    if name == "unet":
        return UNetAdapter(
            in_len=in_len,
            out_len=out_len,
            patch_h=patch_h,
            patch_w=patch_w,
            in_channels=in_channels,
            base_channels=int(model_kwargs.get("unet_base_channels", 32)),
            depth=int(model_kwargs.get("unet_depth", 4)),
            dropout=float(model_kwargs.get("unet_dropout", 0.0)),
        )
    raise ValueError(f"Unsupported model_name={model_name}. Choose from earthformer, iam4vp, phydnet, unet, convlstm.")
