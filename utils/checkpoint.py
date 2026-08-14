from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, state: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(output)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Invalid KMMF-Net checkpoint: {checkpoint_path}")
    return checkpoint

