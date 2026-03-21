import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.consistency import ConsistencyHead

class PatchEmbed(nn.Module):
    """
    Flatten video latent into a sequence of tokens.

    Cosmos latent: [B, C, T, H, W]
    After patchify: [B, N, dim]
    where N = T * (H//patch) * (W//patch)
    """
    def __init__(
        self,
        latent_channels: int = 16,   # cosmos DV4x8x8 channels
        patch_size: int = 4,         # spatial patch size
        dim: int = 512,              # model dim
    ):
        super().__init__()
        self.patch_size = patch_size
        patch_dim       = latent_channels * patch_size * patch_size

        self.proj = nn.Linear(patch_dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        P = self.patch_size

        assert H % P == 0 and W % P == 0

        # spatial patchify
        x = x.reshape(B, C, T, H//P, P, W//P, P)
        x = x.permute(0, 2, 3, 5, 1, 4, 6)       # [B, T, H//P, W//P, C, P, P]
        x = x.reshape(B, T * (H//P) * (W//P), -1) # [B, N, C*P*P]

        x = self.proj(x)   # [B, N, dim]
        x = self.norm(x)
        return x


class TextEmbed(nn.Module):
    """
    Simple text embedding using learned lookup.
    For proper conditioning use T5 encoder.
    Kept simple for now — swap later.
    """
    def __init__(self, vocab_size: int = 32000, dim: int = 512, max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj  = nn.Linear(dim, dim, bias=False)
        self.norm  = nn.RMSNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, seq_len]
        x = self.embed(tokens)    # [B, seq_len, dim]
        x = self.proj(x)
        x = self.norm(x)
        return x.mean(dim=1)      # [B, dim] — pool to single vector


class VideoGenBase(nn.Module):
    """
    Shared base for all three model variants.

    Subclasses implement:
      build_long_ctx()  → long range backbone
      build_short_ctx() → short range backbone

    Forward pass is identical for A, B, C.
    """
    def __init__(
        self,
        dim: int             = 512,
        n_layers: int        = 12,
        n_heads: int         = 8,
        latent_channels: int = 16,
        patch_size: int      = 4,
        short_k: int         = 64,    # short context window size
        vocab_size: int      = 32000,
        max_text_len: int    = 128,
        head_layers: int     = 8,
        noise_std: float     = 0.1,   # noise injection into past latents
        dropout: float       = 0.0,
    ):
        super().__init__()
        self.dim       = dim
        self.short_k   = short_k
        self.noise_std = noise_std

        # shared components — identical across A, B, C
        self.patch_embed = PatchEmbed(latent_channels, patch_size, dim)
        self.text_embed  = TextEmbed(vocab_size, dim, max_text_len)

        # backbones — defined by subclass
        self.long_ctx  = self.build_long_ctx(dim, n_layers, n_heads, dropout)
        self.short_ctx = self.build_short_ctx(dim, n_layers, n_heads, dropout)

        # final norm before head
        self.norm = nn.RMSNorm(dim)

        # consistency head — identical across A, B, C
        self.head = ConsistencyHead(
            dim        = dim * 2,    # long + short concatenated
            latent_dim = latent_channels,
            n_layers   = head_layers,
        )

    def build_long_ctx(self, dim, n_layers, n_heads, dropout):
        raise NotImplementedError

    def build_short_ctx(self, dim, n_layers, n_heads, dropout):
        raise NotImplementedError

    def inject_noise(self, x: torch.Tensor) -> torch.Tensor:
        """CALM noise injection — makes model robust to imperfect past latents."""
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

    def forward(
        self,
        latents: torch.Tensor,   # [B, C, T, H, W] past latents
        tokens:  torch.Tensor,   # [B, seq_len]    text tokens
        x_noisy: torch.Tensor,   # [B, C]          noisy current latent (flattened)
        t:       torch.Tensor,   # [B, 1]          noise level
    ) -> torch.Tensor:

        # 1. noise injection into past latents (CALM trick)
        latents = self.inject_noise(latents)

        # 2. patchify → tokens
        x = self.patch_embed(latents)    # [B, N, dim]

        # 3. text conditioning — add to every token
        text = self.text_embed(tokens)   # [B, dim]
        x    = x + text.unsqueeze(1)     # [B, N, dim]

        # 4. long range context
        z_long  = self.long_ctx(x)       # [B, N, dim]
        z_long  = self.norm(z_long)

        # 5. short range context (last K tokens only)
        z_short = self.short_ctx(x[:, -self.short_k:, :])  # [B, K, dim]
        z_short = z_short.mean(dim=1)    # [B, dim] pool

        # 6. pool long context
        z_long  = z_long.mean(dim=1)     # [B, dim]

        # 7. combine long + short
        z = torch.cat([z_long, z_short], dim=-1)  # [B, dim*2]

        # 8. consistency head → predict x0
        x0 = self.head(z, x_noisy, t)   # [B, latent_channels]
        return x0


if __name__ == "__main__":
    # can't run base directly — needs subclass
    print("Base model defined ✅")
    print("Implement model_a.py, model_b.py, model_c.py next")