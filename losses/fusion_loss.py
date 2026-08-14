"""Weighted intensity and signed-gradient fusion losses.

This module follows the Fusionloss definition supplied with the project. The
source weights are fixed by configuration because the current MSRS training
split does not provide downstream task labels for learning a weight predictor.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SignedSobel(nn.Module):
    """Return signed horizontal and vertical Sobel responses for [B, 1, H, W]."""

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("weight_x", kernel_x, persistent=False)
        self.register_buffer("weight_y", kernel_y, persistent=False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"SignedSobel expects [B,1,H,W], got {tuple(image.shape)}")
        grad_x = F.conv2d(image, self.weight_x.to(image), padding=1)
        grad_y = F.conv2d(image, self.weight_y.to(image), padding=1)
        return grad_x, grad_y


def Fusionloss_int(
    img_f: torch.Tensor,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    weight_a: float | torch.Tensor,
    weight_b: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted dual-source MSE from the supplied Fusionloss implementation."""
    zeros = torch.zeros_like(img_f)
    loss_a = F.mse_loss(weight_a * (img_a - img_f), zeros)
    loss_b = F.mse_loss(weight_b * (img_b - img_f), zeros)
    return loss_a + loss_b, loss_a, loss_b


def Fusionloss_grad(
    img_f: torch.Tensor,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    sobel: SignedSobel | None = None,
) -> torch.Tensor:
    """Match the signed source gradient having the larger absolute response."""
    sobel = SignedSobel().to(img_f.device) if sobel is None else sobel
    fused_x, fused_y = sobel(img_f)
    source_a_x, source_a_y = sobel(img_a)
    source_b_x, source_b_y = sobel(img_b)
    target_x = torch.where(
        source_a_x.abs() >= source_b_x.abs(), source_a_x, source_b_x
    )
    target_y = torch.where(
        source_a_y.abs() >= source_b_y.abs(), source_a_y, source_b_y
    )
    return F.l1_loss(fused_x, target_x) + F.l1_loss(fused_y, target_y)


class Fusionloss(nn.Module):
    """Weighted source-intensity MSE plus signed maximum-gradient L1 loss."""

    def __init__(
        self,
        coeff_int: float = 1.0,
        coeff_grad: float = 1.0,
        weight_vis: float = 0.5,
        weight_ir: float = 0.5,
    ) -> None:
        super().__init__()
        if weight_vis < 0.0 or weight_ir < 0.0:
            raise ValueError("Fusion source weights must be non-negative")
        if weight_vis + weight_ir <= 0.0:
            raise ValueError("At least one fusion source weight must be positive")
        self.sobel = SignedSobel()
        self.coeff_int = float(coeff_int)
        self.coeff_grad = float(coeff_grad)
        self.weight_vis = float(weight_vis)
        self.weight_ir = float(weight_ir)

    def forward(
        self,
        image_vis: torch.Tensor,
        image_ir: torch.Tensor,
        generate_img: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Training is performed in luminance space: all tensors are [B,1,H,W].
        image_vis = image_vis[:, :1]
        image_ir = image_ir[:, :1]
        if generate_img.shape != image_vis.shape or image_ir.shape != image_vis.shape:
            raise ValueError(
                "Fusion inputs must have matching luminance shapes, got "
                f"vis={tuple(image_vis.shape)}, ir={tuple(image_ir.shape)}, "
                f"fused={tuple(generate_img.shape)}"
            )

        loss_intensity, loss_int_vis, loss_int_ir = Fusionloss_int(
            generate_img,
            image_vis,
            image_ir,
            self.weight_vis,
            self.weight_ir,
        )
        loss_gradient = Fusionloss_grad(generate_img, image_vis, image_ir, self.sobel)

        loss_total = self.coeff_int * loss_intensity + self.coeff_grad * loss_gradient
        return loss_total, loss_intensity, loss_gradient, loss_int_vis, loss_int_ir
