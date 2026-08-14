from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .cross_attention import HighFrequencyCrossAttention
from .fft_decomposition import FFTFeatureDecomposition
from .kan_fusion import KANImplicitFusion
from .vmamba_adapter import VSSStage


class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        depths: tuple[int, int],
        backend: str,
        d_state: int,
        ssm_ratio: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.stage1 = VSSStage(
            base_channels, depths[0], backend, d_state, ssm_ratio
        )
        self.down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )
        self.stage2 = VSSStage(
            base_channels * 2, depths[1], backend, d_state, ssm_ratio
        )
        self.backend = self.stage2.backend

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shallow = self.stage1(self.stem(image))
        deep = self.stage2(self.down(shallow))
        return shallow, deep


class MambaDecoder(nn.Module):
    def __init__(
        self,
        deep_channels: int,
        shallow_channels: int,
        depth: int,
        backend: str,
        d_state: int,
        ssm_ratio: float,
    ) -> None:
        super().__init__()
        self.deep_refine = VSSStage(
            deep_channels, depth, backend, d_state, ssm_ratio
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                deep_channels, shallow_channels, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(shallow_channels),
            nn.GELU(),
        )
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(shallow_channels * 2, shallow_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(shallow_channels),
            nn.GELU(),
        )
        self.shallow_refine = VSSStage(
            shallow_channels, 1, backend, d_state, ssm_ratio
        )
        self.head = nn.Conv2d(shallow_channels, 1, 3, padding=1)

    def forward(self, deep: torch.Tensor, shallow: torch.Tensor) -> torch.Tensor:
        x = self.up(self.deep_refine(deep))
        if x.shape[-2:] != shallow.shape[-2:]:
            x = F.interpolate(x, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        x = self.skip_fusion(torch.cat([x, shallow], dim=1))
        return self.head(self.shallow_refine(x))


class KMMFNet(nn.Module):
    """KAN-guided Mamba multimodal image fusion network.

    Input and output tensors use luminance in [0, 1]:
      source_a/source_b: [B, 1, H, W]
      fused:             [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        encoder_depths: tuple[int, int] = (1, 2),
        decoder_depth: int = 2,
        mamba_backend: str = "auto",
        mamba_d_state: int = 16,
        mamba_ratio: float = 2.0,
        fft_cutoff: float = 0.12,
        kan_backend: str = "pykan",
        kan_grid: int = 3,
        kan_order: int = 3,
        kan_pool_size: int = 12,
        attention_heads: int = 4,
        attention_window: int = 8,
        use_output_residual: bool = False,
        output_residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        encoder_args = dict(
            in_channels=in_channels,
            base_channels=base_channels,
            depths=tuple(encoder_depths),
            backend=mamba_backend,
            d_state=mamba_d_state,
            ssm_ratio=mamba_ratio,
        )
        self.encoder_a = MambaEncoder(**encoder_args)
        self.encoder_b = MambaEncoder(**deepcopy(encoder_args))
        feature_channels = base_channels * 2
        self.frequency = FFTFeatureDecomposition(fft_cutoff)
        self.low_fusion = KANImplicitFusion(
            feature_channels,
            backend=kan_backend,
            grid=kan_grid,
            order=kan_order,
            pool_size=kan_pool_size,
        )
        self.high_fusion = HighFrequencyCrossAttention(
            feature_channels, attention_heads, attention_window
        )
        self.frequency_fusion = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
        )
        self.deep_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.decoder = MambaDecoder(
            feature_channels,
            base_channels,
            decoder_depth,
            mamba_backend,
            mamba_d_state,
            mamba_ratio,
        )
        self.use_output_residual = bool(use_output_residual)
        self.output_residual_scale = float(output_residual_scale)
        self.mamba_backend = self.encoder_a.backend
        self.kan_backend = kan_backend

    @staticmethod
    def _validate_inputs(source_a: torch.Tensor, source_b: torch.Tensor) -> None:
        if source_a.shape != source_b.shape:
            raise ValueError(
                f"Input shapes must match, got {tuple(source_a.shape)} and {tuple(source_b.shape)}"
            )
        if source_a.ndim != 4 or source_a.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W] luminance tensors, got {tuple(source_a.shape)}")

    def forward(self, source_a: torch.Tensor, source_b: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_inputs(source_a, source_b)
        original_size = source_a.shape[-2:]
        pad_h = original_size[0] % 2
        pad_w = original_size[1] % 2
        if pad_h or pad_w:
            source_a = F.pad(source_a, (0, pad_w, 0, pad_h), mode="reflect")
            source_b = F.pad(source_b, (0, pad_w, 0, pad_h), mode="reflect")

        shallow_a, deep_a = self.encoder_a(source_a)
        shallow_b, deep_b = self.encoder_b(source_b)
        low_a, high_a = self.frequency(deep_a)
        low_b, high_b = self.frequency(deep_b)
        low_fused = self.low_fusion(low_a, low_b)
        high_fused = self.high_fusion(high_a, high_b)
        frequency_fused = self.frequency_fusion(
            torch.cat([low_fused, high_fused], dim=1)
        )
        deep_base = 0.5 * (deep_a + deep_b)
        deep_fused = deep_base + torch.tanh(self.deep_residual_scale) * frequency_fused
        shallow_fused = 0.5 * (shallow_a + shallow_b)
        logits = self.decoder(deep_fused, shallow_fused)
        if self.use_output_residual:
            base = torch.maximum(source_a, source_b).clamp(1e-4, 1.0 - 1e-4)
            fused = torch.sigmoid(
                torch.logit(base) + self.output_residual_scale * logits
            )
        else:
            fused = torch.sigmoid(logits)
        fused = fused[..., : original_size[0], : original_size[1]]
        return {
            "fused": fused,
            "decoder_logits": logits[..., : original_size[0], : original_size[1]],
            "low_fused": low_fused,
            "high_fused": high_fused,
        }


def build_model(config: dict) -> KMMFNet:
    return KMMFNet(**config)
