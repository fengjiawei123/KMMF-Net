from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_luma(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1).squeeze().cpu().numpy()
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L")


def tensor_to_rgb(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1).cpu()
    if array.ndim == 4:
        array = array[0]
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")


def colorize_luma(fused: torch.Tensor, color_source: torch.Tensor) -> Image.Image:
    fused_y = tensor_to_luma(fused)
    color = tensor_to_rgb(color_source).convert("YCbCr")
    _, cb, cr = color.split()
    if cb.size != fused_y.size:
        cb = cb.resize(fused_y.size, Image.Resampling.BICUBIC)
        cr = cr.resize(fused_y.size, Image.Resampling.BICUBIC)
    return Image.merge("YCbCr", (fused_y, cb, cr)).convert("RGB")


def save_validation_panel(
    source_a: torch.Tensor,
    source_b: torch.Tensor,
    fused: torch.Tensor,
    color_source: torch.Tensor,
    path: str | Path,
) -> None:
    image_a = tensor_to_luma(source_a).convert("RGB")
    image_b = tensor_to_luma(source_b).convert("RGB")
    image_fused = colorize_luma(fused, color_source)
    canvas = Image.new("RGB", (image_a.width * 3, image_a.height))
    canvas.paste(image_a, (0, 0))
    canvas.paste(image_b, (image_a.width, 0))
    canvas.paste(image_fused, (image_a.width * 2, 0))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

