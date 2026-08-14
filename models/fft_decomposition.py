from __future__ import annotations

import torch
from torch import nn


class FFTFeatureDecomposition(nn.Module):
    """Differentiable Gaussian low/high-frequency feature decomposition."""

    def __init__(self, cutoff: float = 0.12) -> None:
        super().__init__()
        if not 0.0 < cutoff < 0.5:
            raise ValueError("FFT cutoff must be in (0, 0.5).")
        self.cutoff = float(cutoff)

    def _low_pass_mask(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(width, device=device, dtype=dtype)
        radius_sq = fy[:, None].square() + fx[None, :].square()
        mask = torch.exp(-radius_sq / (2.0 * self.cutoff**2))
        return mask[None, None]

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(feature.shape)}")
        _, _, height, width = feature.shape
        spectrum = torch.fft.fft2(feature.float(), norm="ortho")
        low_mask = self._low_pass_mask(
            height, width, spectrum.device, spectrum.real.dtype
        )
        low = torch.fft.ifft2(spectrum * low_mask, norm="ortho").real
        low = low.to(feature.dtype)
        high = feature - low
        return low, high

