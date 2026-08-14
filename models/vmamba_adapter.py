from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LiteVSSBlock(nn.Module):
    """Portable shape-compatible fallback for Windows and lightweight runs.

    Formal training should use the official VMamba backend. This block keeps
    four directional spatial paths and the same residual/MLP organization, but
    it does not claim to replace SS2D selective scan.
    """

    def __init__(self, channels: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = max(channels, int(channels * mlp_ratio))
        self.norm1 = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.horizontal = nn.Conv2d(
            channels, channels, (1, 7), padding=(0, 3), groups=channels
        )
        self.vertical = nn.Conv2d(
            channels, channels, (7, 1), padding=(3, 0), groups=channels
        )
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, gate = self.in_proj(self.norm1(x)).chunk(2, dim=1)
        directional = self.horizontal(content) + self.vertical(content)
        x = x + self.out_proj(directional * torch.sigmoid(gate))
        return x + self.mlp(self.norm2(x))


def _install_triton_import_stub() -> bool:
    """Let VMamba import on Windows when its Triton path will not be used."""
    if importlib.util.find_spec("triton") is not None:
        return False
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    triton.jit = lambda function: function
    language.tensor = torch.Tensor
    language.constexpr = int
    triton.language = language
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language
    return True


def _load_official_vssblock(allow_torch_fallback: bool = False) -> type[nn.Module]:
    root = Path(__file__).resolve().parents[1]
    vmamba_root = root / "third_party" / "VMamba"
    if not (vmamba_root / "vmamba.py").is_file():
        raise FileNotFoundError(f"Missing official VMamba file: {vmamba_root / 'vmamba.py'}")
    if str(vmamba_root) not in sys.path:
        sys.path.insert(0, str(vmamba_root))
    # Import timm before installing the narrow VMamba-only Triton stub. PyTorch
    # Dynamo probes a full Triton installation during timm/torchvision import.
    importlib.import_module("timm")
    using_triton_stub = _install_triton_import_stub()
    module = importlib.import_module("vmamba")
    if using_triton_stub:
        module.WITH_TRITON = False
        # VMamba keeps direct references to the stub; removing it from the
        # import registry prevents PyTorch CUDA initialization from treating
        # the placeholder as an installed Triton package.
        sys.modules.pop("triton.language", None)
        sys.modules.pop("triton", None)
    backends = (
        bool(getattr(module, "WITH_SELECTIVESCAN_MAMBA", False)),
        bool(getattr(module, "WITH_SELECTIVESCAN_CORE", False)),
        bool(getattr(module, "WITH_SELECTIVESCAN_OFLEX", False)),
    )
    if not any(backends) and not allow_torch_fallback:
        raise RuntimeError(
            "VMamba imported, but no selective-scan backend is available. "
            "Install mamba-ssm or compile VMamba kernels."
        )
    return module.VSSBlock


def make_vss_block(
    channels: int,
    backend: str,
    d_state: int = 16,
    ssm_ratio: float = 2.0,
) -> tuple[nn.Module, str]:
    if backend not in {"auto", "official", "official_torch", "lite"}:
        raise ValueError(f"Unknown Mamba backend: {backend}")
    if backend in {"auto", "official", "official_torch"}:
        try:
            block_type = _load_official_vssblock(
                allow_torch_fallback=backend == "official_torch"
            )
            block = block_type(
                hidden_dim=channels,
                channel_first=True,
                ssm_d_state=d_state,
                ssm_ratio=ssm_ratio,
                mlp_ratio=2.0,
                forward_type="v04" if backend == "official_torch" else "v2",
            )
            used_backend = "official_torch" if backend == "official_torch" else "official"
            return block, used_backend
        except Exception as error:
            if backend in {"official", "official_torch"}:
                raise RuntimeError("Official VMamba backend could not be initialized") from error
            warnings.warn(f"Official VMamba unavailable ({error}); using lite fallback.")
    return LiteVSSBlock(channels, mlp_ratio=2.0), "lite"


class VSSStage(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        backend: str,
        d_state: int,
        ssm_ratio: float,
    ) -> None:
        super().__init__()
        blocks = []
        used_backends = []
        for _ in range(depth):
            block, used = make_vss_block(channels, backend, d_state, ssm_ratio)
            blocks.append(block)
            used_backends.append(used)
        self.blocks = nn.Sequential(*blocks)
        self.backend = used_backends[0] if used_backends else backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
