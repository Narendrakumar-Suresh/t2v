import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.attention import Attention


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn  = Attention(dim, n_heads, dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn   = SwiGLU(dim, hidden_dim)

    def forward(self, x, mask=None):
        # prenorm + attention + residual
        x = x + self.attn(self.norm1(x), mask=mask)
        # prenorm + ffn + residual
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    block = DecoderBlock(dim=512, n_heads=8, hidden_dim=2048)
    x     = torch.randn(2, 1024, 512)
    out   = block(x)
    print(f"out: {out.shape}")
    assert out.shape == x.shape
    print("✅")