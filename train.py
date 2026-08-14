from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_datasets
from losses import Fusionloss
from models import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.image import save_validation_panel
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KMMF-Net")
    parser.add_argument("--config", default="configs/msrs.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Load model weights only and start a new optimizer/schedule",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    criterion: Fusionloss,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    visual_path: Path,
    max_batches: int = 32,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "intensity": 0.0,
        "gradient": 0.0,
        "intensity_vis": 0.0,
        "intensity_ir": 0.0,
    }
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        source_a = batch["source_a"].to(device, non_blocking=True)
        source_b = batch["source_b"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(source_a, source_b)["fused"]
            (
                loss,
                loss_int,
                loss_grad,
                loss_int_vis,
                loss_int_ir,
            ) = criterion(source_a, source_b, output)
        totals["loss"] += float(loss)
        totals["intensity"] += float(loss_int)
        totals["gradient"] += float(loss_grad)
        totals["intensity_vis"] += float(loss_int_vis)
        totals["intensity_ir"] += float(loss_int_ir)
        batches += 1
        if batch_index == 0:
            save_validation_panel(
                source_a[0],
                source_b[0],
                output[0],
                batch["color"][0],
                visual_path,
            )
    if batches == 0:
        raise RuntimeError("Validation loader yielded no batches")
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.resume and args.pretrained:
        raise ValueError("Use either --resume or --pretrained, not both")
    config = load_config(args.config)
    if args.data_root:
        config["dataset"]["root"] = args.data_root
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    train_config = config["train"]
    requested_device = args.device or train_config.get("device", "cuda")
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp = bool(train_config.get("amp", True) and device.type == "cuda")

    train_dataset, val_dataset = build_datasets(config["dataset"], seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=int(train_config.get("num_workers", 0)) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=min(1, int(train_config.get("num_workers", 0))),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config["model"]).to(device)
    criterion = Fusionloss(**config["loss"]).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    epochs = int(train_config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    accumulation = int(train_config.get("gradient_accumulation", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be >= 1")

    output_dir = Path(train_config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    visual_dir = output_dir / "validation"
    log_path = output_dir / "train.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    start_epoch, global_step, best_loss = 0, 0, math.inf
    if args.pretrained:
        checkpoint = load_checkpoint(args.pretrained, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"loaded_pretrained={Path(args.pretrained).expanduser().resolve()}")
    resume_path = args.resume
    if resume_path:
        checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", math.inf))
        if bool(train_config.get("reset_best_on_resume", False)):
            best_loss = math.inf

    total_parameters, trainable_parameters = count_parameters(model)
    effective_batch = int(train_config["batch_size"]) * accumulation
    print(
        f"device={device} amp={amp} mamba={model.mamba_backend} kan={model.kan_backend} "
        f"train_pairs={len(train_dataset)} val_pairs={len(val_dataset)}"
    )
    print(
        f"parameters total={total_parameters:,} trainable={trainable_parameters:,} "
        f"batch={train_config['batch_size']} accumulation={accumulation} "
        f"effective_batch={effective_batch}"
    )

    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(start_epoch, epochs):
        model.train()
        running = {
            "loss": 0.0,
            "intensity": 0.0,
            "gradient": 0.0,
            "intensity_vis": 0.0,
            "intensity_ir": 0.0,
        }
        micro_steps = 0
        epoch_started = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_index, batch in enumerate(progress):
            if args.limit_train_batches and batch_index >= args.limit_train_batches:
                break
            source_a = batch["source_a"].to(device, non_blocking=True)
            source_b = batch["source_b"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(source_a, source_b)["fused"]
                (
                    loss,
                    loss_int,
                    loss_grad,
                    loss_int_vis,
                    loss_int_ir,
                ) = criterion(source_a, source_b, output)
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch} batch={batch_index}")
            scaler.scale(scaled_loss).backward()
            micro_steps += 1

            should_step = micro_steps % accumulation == 0
            last_batch = batch_index + 1 == len(train_loader)
            last_limited_batch = bool(
                args.limit_train_batches
                and batch_index + 1 == args.limit_train_batches
            )
            if should_step or last_batch or last_limited_batch:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), float(train_config.get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running["loss"] += float(loss.detach())
            running["intensity"] += float(loss_int.detach())
            running["gradient"] += float(loss_grad.detach())
            running["intensity_vis"] += float(loss_int_vis.detach())
            running["intensity_ir"] += float(loss_int_ir.detach())
            if batch_index % int(train_config.get("log_interval", 20)) == 0:
                progress.set_postfix(
                    loss=f"{float(loss):.4f}",
                    int=f"{float(loss_int):.4f}",
                    grad=f"{float(loss_grad):.4f}",
                    step=global_step,
                )
            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                break

        scheduler.step()
        train_batches = max(1, min(len(train_loader), args.limit_train_batches or len(train_loader)))
        validation = validate(
            model,
            criterion,
            val_loader,
            device,
            amp,
            visual_dir / f"epoch_{epoch + 1:03d}.png",
            max_batches=int(train_config.get("max_val_batches", 32)),
        )
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running["loss"] / train_batches,
            "train_intensity": running["intensity"] / train_batches,
            "train_gradient": running["gradient"] / train_batches,
            "train_intensity_vis": running["intensity_vis"] / train_batches,
            "train_intensity_ir": running["intensity_ir"] / train_batches,
            "val_loss": validation["loss"],
            "val_intensity": validation["intensity"],
            "val_gradient": validation["gradient"],
            "val_intensity_vis": validation["intensity_vis"],
            "val_intensity_ir": validation["intensity_ir"],
            "seconds": time.time() - epoch_started,
        }
        print(json.dumps(record, ensure_ascii=False))
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": min(best_loss, validation["loss"]),
            "config": config,
        }
        save_checkpoint(checkpoint_dir / "latest.pt", state)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            state["best_loss"] = best_loss
            save_checkpoint(checkpoint_dir / "best.pt", state)
        if stop_training:
            break

    print(f"training_finished global_step={global_step} best_val_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
