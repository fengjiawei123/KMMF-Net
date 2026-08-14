from __future__ import annotations

import torch

from losses import Fusionloss
from models import build_model
from utils.config import load_config


def main() -> None:
    config = load_config("configs/smoke.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device).train()
    criterion = Fusionloss(**config["loss"]).to(device)
    source_a = torch.rand(1, 1, 64, 64, device=device)
    source_b = torch.rand(1, 1, 64, 64, device=device)
    output = model(source_a, source_b)
    loss, loss_int, loss_grad, loss_int_vis, loss_int_ir = criterion(
        source_a, source_b, output["fused"]
    )
    loss.backward()
    gradient_parameters = sum(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"device={device} mamba={model.mamba_backend} kan={model.kan_backend} "
        f"input={tuple(source_a.shape)} fused={tuple(output['fused'].shape)} "
        f"range=({output['fused'].min().item():.4f},{output['fused'].max().item():.4f})"
    )
    print(
        f"loss={loss.item():.6f} intensity={loss_int.item():.6f} "
        f"gradient={loss_grad.item():.6f} vis={loss_int_vis.item():.6f} "
        f"ir={loss_int_ir.item():.6f} finite_gradient_parameters={gradient_parameters}"
    )
    if gradient_parameters == 0:
        raise RuntimeError("No finite parameter gradients were produced")


if __name__ == "__main__":
    main()
