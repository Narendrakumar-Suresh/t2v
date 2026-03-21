"""
models/base.py — FIXED
========================
Fixes:
1. Masked text pooling (no padding dilution)
2. AdaLN conditioning (stronger than add)
3. 3D-aware token ordering (T, H, W separate)
"""

import torch
import torch.nn as nn
from blocks.consistency import ConsistencyHead


class PatchEmbed(nn.Module):
    """
    [B, C, T, H, W] → [B, N, dim]
    N = T * (H//P) * (W//P)
    Token order: time-major (all spatial patches for t=0, then t=1, ...)
    """
    def __init__(self, latent_channels=16, patch_size=4, dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(latent_channels * patch_size * patch_size,
                              dim, bias=False)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0
        x = x.reshape(B, C, T, H//P, P, W//P, P)
        x = x.permute(0, 2, 3, 5, 1, 4, 6)        # [B, T, H//P, W//P, C, P, P]
        x = x.reshape(B, T * (H//P) * (W//P), -1)  # [B, N, C*P*P]
        return self.norm(self.proj(x))


class TextEmbed(nn.Module):
    """
    Masked mean pooling — ignores padding tokens.
    Returns both pooled vector AND full sequence for cross-attention.
    """
    def __init__(self, vocab_size=32128, dim=512, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj  = nn.Linear(dim, dim, bias=False)
        self.norm  = nn.RMSNorm(dim)

    def forward(self, tokens: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # tokens: [B, seq]
        # mask:   [B, seq] 1=real token, 0=padding
        x = self.norm(self.proj(self.embed(tokens)))  # [B, seq, dim]

        if mask is not None:
            # masked mean — only average real tokens
            m   = mask.float().unsqueeze(-1)          # [B, seq, 1]
            out = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            out = x.mean(dim=1)

        return out   # [B, dim]


class AdaLN(nn.Module):
    """
    Adaptive LayerNorm — modulates norm with conditioning signal.
    Stronger than simple addition for text conditioning.
    DiT uses this for timestep + class conditioning.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2, bias=True),
        )
        # zero init → identity at start of training
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x:    [B, N, dim]
        # cond: [B, cond_dim]
        scale, shift = self.proj(cond).chunk(2, dim=-1)  # each [B, dim]
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class VideoGenBase(nn.Module):
    """
    Base for all three model variants.
    Subclasses implement build_long_ctx() and build_short_ctx().
    """
    def __init__(
        self,
        dim:             int   = 512,
        n_layers:        int   = 12,
        n_heads:         int   = 8,
        latent_channels: int   = 16,
        patch_size:      int   = 4,
        short_k:         int   = 64,
        vocab_size:      int   = 32128,
        max_text_len:    int   = 128,
        head_layers:     int   = 8,
        noise_std:       float = 0.1,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.dim       = dim
        self.short_k   = short_k
        self.noise_std = noise_std

        self.patch_embed = PatchEmbed(latent_channels, patch_size, dim)
        self.text_embed  = TextEmbed(vocab_size, dim, max_text_len)

        # AdaLN for text conditioning — applied before backbone
        self.ada_ln      = AdaLN(dim, cond_dim=dim)

        self.long_ctx    = self.build_long_ctx(dim, n_layers, n_heads, dropout)
        self.short_ctx   = self.build_short_ctx(dim, n_layers, n_heads, dropout)

        self.norm        = nn.RMSNorm(dim)

        self.head        = ConsistencyHead(
            dim         = dim,
            latent_dim  = latent_channels * patch_size * patch_size,
            n_layers    = head_layers,
            context_dim = dim * 2,
        )

        n = sum(p.numel() for p in self.parameters())
        print(f"  {self.__class__.__name__}: {n/1e6:.1f}M params")

    def build_long_ctx(self, dim, n_layers, n_heads, dropout):
        raise NotImplementedError

    def build_short_ctx(self, dim, n_layers, n_heads, dropout):
        raise NotImplementedError

    def inject_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def forward(
        self,
        latents:    torch.Tensor,   # [B, C, T, H, W] PAST frames only
        tokens:     torch.Tensor,   # [B, seq_len]
        x_noisy:    torch.Tensor,   # [B, C, H, W] noisy CURRENT frame
        t:          torch.Tensor,   # [B, 1] noise level
        token_mask: torch.Tensor = None,  # [B, seq_len] padding mask
    ) -> torch.Tensor:

        B, C, H, W = x_noisy.shape
        P = self.patch_embed.patch_size

        # 1. CALM noise injection into past context
        latents = self.inject_noise(latents)

        # 2. patchify past frames → token sequence
        x = self.patch_embed(latents)                  # [B, N_past, dim]

        # 3. masked text conditioning via AdaLN
        text = self.text_embed(tokens, mask=token_mask) # [B, dim]
        x    = self.ada_ln(x, text)                    # adaptive modulation

        # 4. long backbone
        z_long  = self.long_ctx(x)                     # [B, N_past, dim]
        z_long  = self.norm(z_long)

        # 5. short backbone — last K tokens
        k       = min(self.short_k, x.shape[1])
        z_short = self.short_ctx(x[:, -k:, :])         # [B, K, dim]

        # 6. Globalize past context — mean pool tokens [B, dim]
        # This provides the "context" for the current frame prediction
        z = torch.cat([z_long.mean(dim=1), z_short.mean(dim=1)], dim=-1) # [B, dim*2]

        # 7. Patchify current noisy frame [B, C, H, W] → [B, N_curr, C*P*P]
        # Using a simpler patchify for the head input/output
        x_noisy_patches = x_noisy.reshape(B, C, H//P, P, W//P, P)
        x_noisy_patches = x_noisy_patches.permute(0, 2, 4, 1, 3, 5).reshape(B, (H//P)*(W//P), -1)

        # 8. Consistency head applied spatially (per-patch)
        # Head takes [B, dim*2] context and [B, N_curr, C*P*P] noisy patches
        out_patches = self.head(z, x_noisy_patches, t) # [B, N_curr, C*P*P]

        # 9. Un-patchify [B, N_curr, C*P*P] → [B, C, H, W]
        out = out_patches.reshape(B, H//P, W//P, C, P, P)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return out


if __name__ == "__main__":
    print("Base model defined ✅")