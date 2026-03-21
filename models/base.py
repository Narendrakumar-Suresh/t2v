import torch
import torch.nn as nn
from blocks.consistency import ConsistencyHead


class PatchEmbed(nn.Module):
    """
    Flatten video latent into sequence of tokens.
    Cosmos latent: [B, C, T, H, W]
    After patchify: [B, N, dim]
    where N = T * (H//patch) * (W//patch)
    """
    def __init__(
        self,
        latent_channels: int = 16,
        patch_size: int = 4,
        dim: int = 896,
    ):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = latent_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, \
            f"H={H}, W={W} must be divisible by patch_size={P}"
        x = x.reshape(B, C, T, H//P, P, W//P, P)
        x = x.permute(0, 2, 3, 5, 1, 4, 6)        # [B, T, H//P, W//P, C, P, P]
        x = x.reshape(B, T * (H//P) * (W//P), -1)  # [B, N, C*P*P]
        x = self.proj(x)
        x = self.norm(x)
        return x


class TextEmbed(nn.Module):
    """
    Text embedding via learned lookup.
    Uses flan-t5-base vocab size (32128).
    """
    def __init__(
        self,
        vocab_size: int = 32128,   # ← flan-t5-base exact vocab size
        dim: int = 896,
        max_len: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj  = nn.Linear(dim, dim, bias=False)
        self.norm  = nn.RMSNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)    # [B, seq_len, dim]
        x = self.proj(x)
        x = self.norm(x)
        return x.mean(dim=1)      # [B, dim] — mean pool


class VideoGenBase(nn.Module):
    """
    Shared base for all three model variants.
    Subclasses implement build_long_ctx() and build_short_ctx().
    """
    def __init__(
        self,
        dim: int             = 512,    # ← 896 gives ~500M params
        n_layers: int        = 24,
        n_heads: int         = 16,
        latent_channels: int = 16,
        patch_size: int      = 4,
        short_k: int         = 64,
        vocab_size: int      = 32128,  # ← flan-t5-base
        max_text_len: int    = 128,
        head_layers: int     = 8,
        noise_std: float     = 0.1,
        dropout: float       = 0.0,
    ):
        super().__init__()
        self.dim       = dim
        self.short_k   = short_k
        self.noise_std = noise_std

        self.patch_embed = PatchEmbed(latent_channels, patch_size, dim)
        self.text_embed  = TextEmbed(vocab_size, dim, max_text_len)

        self.long_ctx  = self.build_long_ctx(dim, n_layers, n_heads, dropout)
        self.short_ctx = self.build_short_ctx(dim, n_layers, n_heads, dropout)

        self.norm = nn.RMSNorm(dim)

        self.head = ConsistencyHead(
            dim        = dim * 2,
            latent_dim = latent_channels,
            n_layers   = head_layers,
        )

        # print param count on init
        n = sum(p.numel() for p in self.parameters())
        print(f"  {self.__class__.__name__}: {n/1e6:.1f}M params")

    def build_long_ctx(self, dim, n_layers, n_heads, dropout):
        raise NotImplementedError

    def build_short_ctx(self, dim, n_layers, n_heads, dropout):
        raise NotImplementedError

    def inject_noise(self, x: torch.Tensor) -> torch.Tensor:
        """CALM noise injection into past latents."""
        if self.training and self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def forward(
        self,
        latents: torch.Tensor,   # [B, C, T, H, W] past latents
        tokens:  torch.Tensor,   # [B, seq_len]
        x_noisy: torch.Tensor,   # [B, C] noisy current latent
        t:       torch.Tensor,   # [B, 1] noise level
    ) -> torch.Tensor:

        # 1. noise injection (CALM trick)
        latents = self.inject_noise(latents)

        # 2. patchify → [B, N, dim]
        x = self.patch_embed(latents)

        # 3. text conditioning
        text = self.text_embed(tokens)      # [B, dim]
        x    = x + text.unsqueeze(1)        # broadcast to [B, N, dim]

        # 4. long backbone
        z_long = self.long_ctx(x)           # [B, N, dim]
        z_long = self.norm(z_long)

        # 5. short backbone — guard against short sequences
        k      = min(self.short_k, x.shape[1])
        z_short = self.short_ctx(x[:, -k:, :])  # [B, K, dim]
        z_short = z_short.mean(dim=1)            # [B, dim]

        # 6. pool long
        z_long = z_long.mean(dim=1)         # [B, dim]

        # 7. combine
        z = torch.cat([z_long, z_short], dim=-1)  # [B, dim*2]

        # 8. consistency head
        return self.head(z, x_noisy, t)     # [B, latent_channels]


if __name__ == "__main__":
    print("Base model defined ✅")