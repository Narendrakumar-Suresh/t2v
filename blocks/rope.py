"""
RoPE — Rotary Position Embedding
==================================
Used in attention blocks to encode position
into Q and K before dot product.
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        # frequencies for each pair of dims
        # shape: [dim//2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # cache
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len: int, device: torch.device):
        # recompute only if seq_len changed
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cos_cached = emb.cos()[None, :, None, :]  # [1, N, 1, dim]
            self.sin_cached = emb.sin()[None, :, None, :]  # [1, N, 1, dim]

        # Ensure cached tensors are on the correct device
        return self.cos_cached.to(device), self.sin_cached.to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits x into two halves and rotates:
    [x1, x2] → [-x2, x1]
    This implements the 2D rotation matrix.
    """
    x1 = x[..., : x.shape[-1] // 2]  # first half
    x2 = x[..., x.shape[-1] // 2 :]  # second half
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to Q and K.

    Args:
        q:   [B, N, H, head_dim]
        k:   [B, N, H, head_dim]
        cos: [1, N, 1, head_dim]
        sin: [1, N, 1, head_dim]

    Returns:
        q_rot, k_rot — same shapes as input
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


if __name__ == "__main__":
    # smoke test
    B, N, H, hd = 2, 1024, 8, 64
    dim = H * hd  # 512

    rope = RoPE(dim=hd)  # per-head dim
    cos, sin = rope(seq_len=N, device=torch.device("cpu"))

    q = torch.randn(B, N, H, hd)
    k = torch.randn(B, N, H, hd)

    q_rot, k_rot = apply_rope(q, k, cos, sin)

    print(f"q:     {q.shape}")
    print(f"q_rot: {q_rot.shape}")
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("✅ RoPE works!")
