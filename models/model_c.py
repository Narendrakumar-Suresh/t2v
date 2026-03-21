# models/model_c.py
import torch.nn as nn
from models.base import VideoGenBase
from blocks.mamba2 import Mamba2Block


class ModelC(VideoGenBase):
    """Mamba2 + Mamba2 — pure SSM"""

    def build_long_ctx(self, dim, n_layers, n_heads, dropout):
        return nn.Sequential(*[
            Mamba2Block(dim=dim, state_dim=128, conv_width=4, expand=4)
            for _ in range(n_layers)
        ])

    def build_short_ctx(self, dim, n_layers, n_heads, dropout):
        # smaller state_dim — short context needs less memory
        return nn.Sequential(*[
            Mamba2Block(dim=dim, state_dim=64, conv_width=4, expand=4)
            for _ in range(n_layers // 2)
        ])


if __name__ == "__main__":
    import torch
    model   = ModelC(dim=64, n_layers=2, n_heads=4)
    latents = torch.randn(2, 16, 4, 32, 32)
    tokens  = torch.randint(0, 32128, (2, 32))
    x_noisy = torch.randn(2, 16)
    t       = torch.rand(2, 1)
    out     = model(latents, tokens, x_noisy, t)
    print(f"out: {out.shape}")   # [2, 16]
    print("✅ Model C works!")