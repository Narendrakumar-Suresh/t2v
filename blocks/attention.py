import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.rope import RoPE, apply_rope


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.rope = RoPE(dim=self.head_dim)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        H, hd = self.n_heads, self.head_dim

        # project → [B, N, H, hd]
        q = self.wq(x).reshape(B, N, H, hd)
        k = self.wk(x).reshape(B, N, H, hd)
        v = self.wv(x).reshape(B, N, H, hd)

        # apply RoPE before attention
        cos, sin = self.rope(seq_len=N, device=x.device)
        q, k = apply_rope(q, k, cos, sin)

        # [B, N, H, hd] → [B, H, N, hd] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlashAttention on MI300X automatically
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


if __name__ == "__main__":
    x = torch.randn(2, 1024, 512)
    attn = Attention(dim=512, n_heads=8)
    out = attn(x)
    print(f"out: {out.shape}")
    assert out.shape == x.shape
    print("✅")
