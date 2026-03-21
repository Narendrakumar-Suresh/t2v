import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyHead(nn.Module):
    """
    MLP head that predicts clean latent x0
    from noisy latent + backbone context.

    Input:  z (backbone context) + x_noisy + noise level t
    Output: predicted clean x0

    This is the core of consistency training —
    model learns to denoise in 1-4 steps.
    """

    def __init__(
        self,
        dim: int,  # backbone context dim (e.g. 512)
        latent_dim: int,  # cosmos latent channels (16)
        n_layers: int = 8,  # MLP depth
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim

        # noise level embedding
        # t is a scalar → embed into dim
        self.t_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # project noisy latent → dim
        self.x_proj = nn.Linear(latent_dim, dim, bias=False)

        # combine z + x_noisy + t → dim
        self.in_proj = nn.Linear(dim * 3, dim, bias=False)

        # residual MLP blocks
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.RMSNorm(dim),
                    nn.Linear(dim, dim * 4, bias=False),
                    nn.SiLU(),
                    nn.Linear(dim * 4, dim, bias=False),
                )
                for _ in range(n_layers)
            ]
        )

        # output → predict clean latent
        self.out_norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, latent_dim, bias=False)

        # zero init output → stable training start
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        z: torch.Tensor,  # [B, dim]        backbone context
        x_noisy: torch.Tensor,  # [B, latent_dim] noisy latent
        t: torch.Tensor,  # [B, 1]          noise level
    ) -> torch.Tensor:

        # embed noise level
        t_emb = self.t_embed(t)  # [B, dim]

        # project noisy latent
        x_emb = self.x_proj(x_noisy)  # [B, dim]

        # combine all three signals
        h = torch.cat([z, x_emb, t_emb], dim=-1)  # [B, dim*3]
        h = self.in_proj(h)  # [B, dim]

        # residual MLP
        for layer in self.layers:
            h = h + layer(h)

        # predict x0
        h = self.out_norm(h)
        x0 = self.out_proj(h)  # [B, latent_dim]
        return x0


if __name__ == "__main__":
    B = 2
    dim = 512
    latent_dim = 16

    head = ConsistencyHead(dim=dim, latent_dim=latent_dim)
    z = torch.randn(B, dim)  # backbone output
    x_noisy = torch.randn(B, latent_dim)  # noisy latent
    t = torch.rand(B, 1)  # noise level 0-1

    x0 = head(z, x_noisy, t)
    print(f"x0: {x0.shape}")  # [2, 16]
    assert x0.shape == (B, latent_dim)
    print("✅ ConsistencyHead works!")
