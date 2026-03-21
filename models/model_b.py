# models/model_b.py
import torch.nn as nn
from models.base import VideoGenBase
from blocks.mamba2 import Mamba2Block
from blocks.decoder import DecoderBlock


class ModelB(VideoGenBase):
    """Mamba2 + Transformer — main hypothesis"""

    def build_long_ctx(self, dim, n_layers, n_heads, dropout):
        # Mamba2 for long range temporal
        return nn.Sequential(*[
            Mamba2Block(dim=dim, state_dim=128, conv_width=4, expand=2)
            for _ in range(n_layers)
        ])

    def build_short_ctx(self, dim, n_layers, n_heads, dropout):
        # Transformer for local spatial detail
        return nn.Sequential(*[
            DecoderBlock(dim, n_heads, dim * 4, dropout)
            for _ in range(n_layers // 2)
        ])


if __name__ == "__main__":
    import torch
    model   = ModelB(dim=64, n_layers=2, n_heads=4)
    latents = torch.randn(2, 16, 4, 32, 32)
    tokens  = torch.randint(0, 32000, (2, 32))
    x_noisy = torch.randn(2, 16)
    t       = torch.rand(2, 1)
    out     = model(latents, tokens, x_noisy, t)
    print(f"out: {out.shape}")   # [2, 16]
    print("✅ Model B works!")