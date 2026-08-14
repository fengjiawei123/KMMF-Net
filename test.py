from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.image import colorize_luma


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse paired images with KMMF-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-a", required=True, help="Image or first modality directory")
    parser.add_argument("--source-b", required=True, help="Image or second modality directory")
    parser.add_argument("--output", default="outputs/test")
    parser.add_argument("--color-from", choices=["a", "b", "gray"], default="a")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def image_files(path: Path) -> dict[str, Path]:
    if path.is_file():
        return {path.name: path}
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {
        item.name: item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in EXTENSIONS
    }


def luma_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("YCbCr").getchannel("Y"), dtype=np.float32).copy()
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0) / 255.0


def rgb_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = build_model(config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"checkpoint={Path(args.checkpoint).expanduser().resolve()} "
        f"device={device} mamba={model.mamba_backend} kan={model.kan_backend} "
        f"parameters={parameters:,}"
    )

    files_a = image_files(Path(args.source_a))
    files_b = image_files(Path(args.source_b))
    common = sorted(set(files_a) & set(files_b))
    if not common and len(files_a) == len(files_b) == 1:
        common = [next(iter(files_a))]
        files_b[common[0]] = next(iter(files_b.values()))
    if not common:
        raise RuntimeError("No same-name image pairs were found")
    if args.limit:
        common = common[: args.limit]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name in tqdm(common, desc="fusion"):
            image_a = Image.open(files_a[name]).convert("RGB")
            image_b = Image.open(files_b[name]).convert("RGB")
            if image_a.size != image_b.size:
                raise ValueError(f"Size mismatch for {name}: {image_a.size} vs {image_b.size}")
            source_a = luma_tensor(image_a).to(device)
            source_b = luma_tensor(image_b).to(device)
            fused = model(source_a, source_b)["fused"][0]
            if args.color_from == "a":
                color = rgb_tensor(image_a)
                result = colorize_luma(fused, color)
            elif args.color_from == "b":
                color = rgb_tensor(image_b)
                result = colorize_luma(fused, color)
            else:
                result = Image.fromarray(
                    np.round(fused.squeeze().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                    mode="L",
                )
            result.save(output_dir / Path(name).with_suffix(".png"))
    print(f"saved={len(common)} output={output_dir.resolve()}")


if __name__ == "__main__":
    main()
