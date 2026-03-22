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
        # Improved A initialization: spread decay rates for better long-range dependency
        # shape: [inner_dim, state_dim]
        indices = torch.arange(1, state_dim + 1).float()
        A_init = torch.log(indices).repeat(self.inner_dim, 1)
        self.A_log = nn.Parameter(A_init)
        
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
        """
        B, L, D = x.shape
        N = self.state_dim

        # discretize A (always in float32 for stability)
        # A is [D, N], always negative
        A = -torch.exp(self.A_log.float())  # [D, N]

        # compute B, C, dt per token
        Bx = self.B(x).float()   # [B, L, N]
        Cx = self.C(x).float()   # [B, L, N]
        dt = F.softplus(self.dt_proj(x)).float() + 1e-4 # [B, L, D] (add eps for stability)

        # Pre-expand A to [1, D, N] for broadcasting
        A_expanded = A.unsqueeze(0)

        # scan in float32
        h = torch.zeros(B, D, N, device=x.device, dtype=torch.float32)
        ys = []

        # Optimization: Pull constants out of the loop
        # x_float is used repeatedly
        x_float = x.float()
        D_param = self.D.float()

        for t in range(L):
            # dt_t: [B, D, 1]
            dt_t = dt[:, t, :].unsqueeze(-1)

            # discretized A: [B, D, N]
            dA = torch.exp(dt_t * A_expanded)

            # discretized B: [B, D, N]
            # Use ZOH approximation: dB = dt * B
            dB = dt_t * Bx[:, t, :].unsqueeze(1)

            # state update: h = A*h + B*x
            # x_t: [B, D, 1]
            x_t = x_float[:, t, :].unsqueeze(-1)
            h = dA * h + dB * x_t

            # output: y = C*h + D*x
            # C_t: [B, 1, N]
            C_t = Cx[:, t, :].unsqueeze(1)
            y = (C_t * h).sum(-1) + D_param * x_float[:, t, :]
            ys.append(y)

        return torch.stack(ys, dim=1).to(x.dtype)  # [B, L, D]


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
