import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2Block(nn.Module):
    """
    Simplified Mamba2 (State Space Model) block.

    Core idea:
      x → expand → conv → SSM scan → gate → contract → output

    SSM:
      h_t = A * h_{t-1} + B * x_t   ← state update
      y_t = C * h_t                  ← output
    """

    def __init__(
        self,
        dim: int,  # model dim (e.g. 512)
        state_dim: int = 64,  # SSM state size (N)
        conv_width: int = 4,  # local conv kernel
        expand: int = 2,  # inner expansion factor
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.inner_dim = dim * expand  # expanded dim

        # input projection → x and z (gate)
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)

        # local conv (short-range context before SSM)
        self.conv = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=conv_width,
            padding=conv_width - 1,  # causal padding
            groups=self.inner_dim,  # depthwise
            bias=True,
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.inner_dim, state_dim))
        self.B = nn.Linear(self.inner_dim, state_dim, bias=False)
        self.C = nn.Linear(self.inner_dim, state_dim, bias=False)
        self.D = nn.Parameter(torch.ones(self.inner_dim))  # skip connection

        # dt (discretization step) projection
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        # stable dt init: small values initially
        nn.init.constant_(self.dt_proj.bias, -3.0)
        nn.init.zeros_(self.dt_proj.weight)

        # output projection
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim)

    def ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequential SSM scan.
        x: [B, L, inner_dim]

        h_t = A * h_{t-1} + B * x_t
        y_t = C * h_t + D * x_t
        """
        B, L, D = x.shape
        N = self.state_dim

        # discretize A
        A = -torch.exp(self.A_log.float())  # [inner_dim, N] negative = stable

        # compute B, C per token
        Bx = self.B(x)  # [B, L, N]
        Cx = self.C(x)  # [B, L, N]

        # dt (step size)
        dt = F.softplus(self.dt_proj(x))  # [B, L, inner_dim]

        # scan
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            # discretized A: [B, inner_dim, N]
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))

            # discretized B: [B, inner_dim, N]
            dB = dt[:, t, :].unsqueeze(-1) * Bx[:, t, :].unsqueeze(1)

            # state update: h = A*h + B*x
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)

            # output: y = C*h + D*x
            y = (Cx[:, t, :].unsqueeze(1) * h).sum(-1)  # [B, inner_dim]
            y = y + self.D * x[:, t, :]
            ys.append(y)

        return torch.stack(ys, dim=1)  # [B, L, inner_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        x = self.norm(x)

        # project → split into x and gate z
        xz = self.in_proj(x)  # [B, L, inner_dim*2]
        x, z = xz.chunk(2, dim=-1)  # each [B, L, inner_dim]

        # causal conv (local context)
        x = x.transpose(1, 2)  # [B, inner_dim, L]
        x = self.conv(x)[:, :, :L]  # trim causal padding
        x = x.transpose(1, 2)  # [B, L, inner_dim]
        x = F.silu(x)

        # SSM scan
        y = self.ssm_scan(x)  # [B, L, inner_dim]

        # gate
        y = y * F.silu(z)

        # output projection
        y = self.out_proj(y)  # [B, L, dim]

        return y + residual  # residual


if __name__ == "__main__":
    block = Mamba2Block(dim=512, state_dim=64, conv_width=4, expand=2)
    x = torch.randn(2, 1024, 512)
    out = block(x)
    print(f"out: {out.shape}")
    assert out.shape == x.shape
    print("✅ Mamba2 works!")
