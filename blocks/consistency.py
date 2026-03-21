"""
blocks/consistency.py — FIXED
================================
Fixes:
1. Boundary condition: f(x, 0) = x via c_skip + c_out
2. Zero-init collapse prevention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyHead(nn.Module):
    """
    Predicts clean latent x0 from noisy input + backbone context.

    Uses proper consistency model parameterization:
      f(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)

    Boundary condition guaranteed:
      as t → 0: c_skip → 1, c_out → 0
      so f(x, 0) = x  ✅
    """

    def __init__(
        self,
        dim:        int = 512,
        latent_dim: int = 16,
        n_layers:   int = 8,
        sigma_data: float = 0.5,   # data std estimate
        context_dim: int = None,    # dimension of backbone context z
    ):
        super().__init__()
        self.dim         = dim
        self.latent_dim  = latent_dim
        self.sigma_data  = sigma_data
        self.context_dim = context_dim if context_dim is not None else dim

        # noise level embedding (sinusoidal → MLP)
        self.t_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # project noisy latent → dim
        self.x_proj  = nn.Linear(latent_dim, dim, bias=False)

        # combine z + x_emb + t_emb
        self.in_proj = nn.Linear(self.context_dim + dim + dim, dim, bias=False)

        # residual MLP blocks
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, dim * 4, bias=False),
                nn.SiLU(),
                nn.Linear(dim * 4, dim, bias=False),
            )
            for _ in range(n_layers)
        ])

        self.out_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, latent_dim, bias=False)

        # small random init (not zero) to avoid collapse
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def c_skip(self, t: torch.Tensor) -> torch.Tensor:
        """Skip weight → 1 as t → 0 (boundary condition)"""
        return self.sigma_data ** 2 / (t ** 2 + self.sigma_data ** 2)

    def c_out(self, t: torch.Tensor) -> torch.Tensor:
        """Output weight → 0 as t → 0 (boundary condition)"""
        return t * self.sigma_data / (t ** 2 + self.sigma_data ** 2).sqrt()

    def forward(
        self,
        z:       torch.Tensor,   # [B, dim] or [B, N, dim]
        x_noisy: torch.Tensor,   # [B, latent_dim] or [B, N, latent_dim]
        t:       torch.Tensor,   # [B, 1]
    ) -> torch.Tensor:

        # compute skip and output weights
        cs = self.c_skip(t)    # [B, 1]
        co = self.c_out(t)     # [B, 1]

        # embed noise level
        t_emb = self.t_embed(t)          # [B, dim]

        # project noisy latent
        x_emb = self.x_proj(x_noisy)     # [B, dim] or [B, N, dim]

        # handles spatial broadcasting if x_noisy is [B, N, C]
        if x_noisy.ndim == 3:
            B, N, _ = x_emb.shape
            z     = z.unsqueeze(1) if z.ndim == 2 else z
            t_emb = t_emb.unsqueeze(1)
            # expand spatial but keep feature dim
            z     = z.expand(-1, N, -1)
            t_emb = t_emb.expand(-1, N, -1)
            cs    = cs.view(B, 1, 1)
            co    = co.view(B, 1, 1)

        # combine
        h = torch.cat([z, x_emb, t_emb], dim=-1)
        h = self.in_proj(h)

        # residual MLP
        for layer in self.layers:
            h = h + layer(h)

        # raw network output
        h  = self.out_norm(h)
        Fx = self.out_proj(h)             # [B, latent_dim]

        # boundary-preserving parameterization:
        # f(x, t) = c_skip(t) * x + c_out(t) * F(x, t)
        return cs * x_noisy + co * Fx


if __name__ == "__main__":
    B, dim, latent_dim = 2, 512, 16
    head    = ConsistencyHead(dim=dim, latent_dim=latent_dim)
    z       = torch.randn(B, dim)
    x_noisy = torch.randn(B, latent_dim)

    # test boundary: at t=0, output should equal x_noisy
    t_zero  = torch.zeros(B, 1)
    out_zero = head(z, x_noisy, t_zero)
    err = (out_zero - x_noisy).abs().max().item()
    print(f"Boundary condition error at t=0: {err:.6f} (should be ~0)")
    assert err < 1e-5, "Boundary condition violated!"

    # test normal forward
    t = torch.rand(B, 1)
    out = head(z, x_noisy, t)
    print(f"x0: {out.shape}")
    assert out.shape == (B, latent_dim)
    print("✅ ConsistencyHead works with boundary conditions!")