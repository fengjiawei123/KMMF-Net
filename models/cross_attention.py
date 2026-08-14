from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _window_partition(x: torch.Tensor, window: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    batch, channels, height, width = x.shape
    pad_h = (window - height % window) % window
    pad_w = (window - width % window) % window
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    hp, wp = height + pad_h, width + pad_w
    tokens = (
        x.view(batch, channels, hp // window, window, wp // window, window)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, window * window, channels)
    )
    return tokens, (batch, channels, height, width, hp, wp)


def _window_reverse(tokens: torch.Tensor, shape: tuple[int, ...], window: int) -> torch.Tensor:
    batch, channels, height, width, hp, wp = shape
    x = (
        tokens.view(batch, hp // window, wp // window, window, window, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, hp, wp)
    )
    return x[:, :, :height, :width]


class HighFrequencyCrossAttention(nn.Module):
    """Windowed bidirectional cross-attention followed by local gated fusion."""

    def __init__(self, channels: int, heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.window_size = int(window_size)
        self.norm_a = nn.LayerNorm(channels)
        self.norm_b = nn.LayerNorm(channels)
        self.cross_a = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.cross_b = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.local_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        tokens_a, shape = _window_partition(feature_a, self.window_size)
        tokens_b, shape_b = _window_partition(feature_b, self.window_size)
        if shape != shape_b:
            raise ValueError("High-frequency feature shapes must match")

        norm_a = self.norm_a(tokens_a)
        norm_b = self.norm_b(tokens_b)
        enhanced_a = tokens_a + self.cross_a(norm_a, norm_b, norm_b, need_weights=False)[0]
        enhanced_b = tokens_b + self.cross_b(norm_b, norm_a, norm_a, need_weights=False)[0]
        enhanced_a = _window_reverse(enhanced_a, shape, self.window_size)
        enhanced_b = _window_reverse(enhanced_b, shape, self.window_size)

        gate = self.gate(torch.cat([enhanced_a, enhanced_b], dim=1))
        fused = gate * enhanced_a + (1.0 - gate) * enhanced_b
        return fused + self.local_fusion(fused)

