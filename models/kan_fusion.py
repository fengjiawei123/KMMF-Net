from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F


class RBFKANLinear(nn.Module):
    """Small GPU-friendly KAN fallback used only when pykan is unavailable."""

    def __init__(self, in_features: int, out_features: int, grid: int = 5) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        centers = torch.linspace(-1.0, 1.0, grid)
        self.register_buffer("centers", centers)
        self.log_width = nn.Parameter(torch.tensor(-0.2))
        self.edge_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid)
        )
        nn.init.normal_(self.edge_weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        width = self.log_width.exp().clamp_min(1e-3)
        basis = torch.exp(-((x[..., None] - self.centers) / width).square())
        spline = torch.einsum("nig,oig->no", basis, self.edge_weight)
        return self.base(F.silu(x)) + spline


class RBFKAN(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, grid: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            RBFKANLinear(in_features, hidden, grid),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            RBFKANLinear(hidden, out_features, grid),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class KANImplicitFusion(nn.Module):
    """Map low-frequency features to a compact implicit grid and fuse with KAN."""

    def __init__(
        self,
        channels: int,
        backend: str = "pykan",
        grid: int = 3,
        order: int = 3,
        pool_size: int = 12,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.pool_size = int(pool_size)
        self.backend = backend
        self.map_a = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU())
        self.map_b = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU())
        self.token_norm = nn.LayerNorm(channels * 2)

        if backend == "pykan":
            try:
                from kan import KAN

                self.kan = KAN(
                    width=[channels * 2, channels, channels],
                    grid=grid,
                    k=order,
                    symbolic_enabled=False,
                    save_act=False,
                    auto_save=False,
                    device="cpu",
                ).speed()
            except Exception as error:
                raise RuntimeError(
                    "pykan backend requested but could not be initialized. "
                    "Install pykan==0.2.8 or set model.kan_backend=rbf."
                ) from error
        elif backend == "rbf":
            warnings.warn("Using the RBF-KAN fallback; use pykan for formal training.")
            self.kan = RBFKAN(channels * 2, channels, channels, grid)
        else:
            raise ValueError(f"Unknown KAN backend: {backend}")

        self.inverse_map = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, low_a: torch.Tensor, low_b: torch.Tensor) -> torch.Tensor:
        height, width = low_a.shape[-2:]
        mapped_a = self.map_a(low_a)
        mapped_b = self.map_b(low_b)
        implicit = torch.cat([mapped_a, mapped_b], dim=1)
        implicit = F.adaptive_avg_pool2d(
            implicit, (self.pool_size, self.pool_size)
        )
        batch = implicit.shape[0]
        tokens = implicit.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens).reshape(-1, self.channels * 2)
        token_dtype = tokens.dtype
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            fused_tokens = self.kan(tokens.float())
        fused_tokens = fused_tokens.to(token_dtype)
        fused = fused_tokens.view(batch, self.pool_size**2, self.channels)
        fused = fused.transpose(1, 2).reshape(
            batch, self.channels, self.pool_size, self.pool_size
        )
        fused = F.interpolate(fused, size=(height, width), mode="bilinear", align_corners=False)
        fused = self.inverse_map(fused)
        base = 0.5 * (low_a + low_b)
        return base + torch.tanh(self.residual_scale) * fused
