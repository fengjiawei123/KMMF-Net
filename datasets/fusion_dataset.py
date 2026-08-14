from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairRecord:
    source_a: Path
    source_b: Path
    color_from: str
    group: str

    @property
    def name(self) -> str:
        return self.source_a.stem


def _image_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not files:
        raise RuntimeError(f"No images found in {directory}")
    return files


def _match_directory_pair(
    directory_a: Path, directory_b: Path, color_from: str, group: str
) -> list[PairRecord]:
    images_a = _image_map(directory_a)
    images_b = _image_map(directory_b)
    names_a, names_b = set(images_a), set(images_b)
    missing_b = sorted(names_a - names_b)
    missing_a = sorted(names_b - names_a)
    if missing_a or missing_b:
        raise RuntimeError(
            f"Pair mismatch for {group}: missing_from_a={len(missing_a)}, "
            f"missing_from_b={len(missing_b)}. Example: {(missing_a + missing_b)[:5]}"
        )
    return [
        PairRecord(images_a[name], images_b[name], color_from, group)
        for name in sorted(names_a)
    ]


def discover_pairs(dataset_config: dict) -> list[PairRecord]:
    root = Path(dataset_config["root"]).expanduser()
    dataset_type = dataset_config["type"].lower()
    if dataset_type == "msrs":
        return _match_directory_pair(
            root / dataset_config.get("source_a", "vi"),
            root / dataset_config.get("source_b", "ir"),
            dataset_config.get("color_from", "a"),
            "msrs",
        )
    if dataset_type == "medical":
        records: list[PairRecord] = []
        for item in dataset_config.get("medical_sets", []):
            records.extend(
                _match_directory_pair(
                    root / item["source_a"],
                    root / item["source_b"],
                    item.get("color_from", "b"),
                    item["name"],
                )
            )
        if not records:
            raise RuntimeError("No medical pair definitions were configured")
        return records
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _to_luma_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("YCbCr").getchannel("Y"), dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0) / 255.0


def _to_rgb_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _paired_crop(
    images: list[Image.Image], crop_size: int, training: bool
) -> list[Image.Image]:
    width, height = images[0].size
    if any(image.size != (width, height) for image in images[1:]):
        raise ValueError(f"Paired images must have equal sizes: {[image.size for image in images]}")
    if width < crop_size or height < crop_size:
        scale = max(crop_size / width, crop_size / height)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        images = [image.resize(new_size, Image.Resampling.BICUBIC) for image in images]
        width, height = new_size
    if training:
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
    else:
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
    box = (left, top, left + crop_size, top + crop_size)
    return [image.crop(box) for image in images]


class FusionPairDataset(Dataset):
    def __init__(
        self,
        records: Iterable[PairRecord],
        crop_size: int,
        training: bool,
        augment: bool = True,
    ) -> None:
        self.records = list(records)
        self.crop_size = int(crop_size)
        self.training = bool(training)
        self.augment = bool(augment and training)
        if not self.records:
            raise ValueError("FusionPairDataset received no records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image_a = Image.open(record.source_a).convert("RGB")
        image_b = Image.open(record.source_b).convert("RGB")
        color = image_a.copy() if record.color_from == "a" else image_b.copy()
        image_a, image_b, color = _paired_crop(
            [image_a, image_b, color], self.crop_size, self.training
        )
        if self.augment and random.random() < 0.5:
            image_a = image_a.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image_b = image_b.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            color = color.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augment and random.random() < 0.5:
            image_a = image_a.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image_b = image_b.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            color = color.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        return {
            "source_a": _to_luma_tensor(image_a),
            "source_b": _to_luma_tensor(image_b),
            "color": _to_rgb_tensor(color),
            "name": record.name,
            "group": record.group,
        }


def build_datasets(dataset_config: dict, seed: int) -> tuple[FusionPairDataset, FusionPairDataset]:
    records = discover_pairs(dataset_config)
    generator = random.Random(seed)
    generator.shuffle(records)
    val_count = max(1, int(round(len(records) * float(dataset_config.get("val_ratio", 0.1)))))
    val_records = records[:val_count]
    train_records = records[val_count:]
    crop_size = int(dataset_config.get("crop_size", 256))
    return (
        FusionPairDataset(
            train_records,
            crop_size=crop_size,
            training=True,
            augment=bool(dataset_config.get("augment", True)),
        ),
        FusionPairDataset(
            val_records,
            crop_size=crop_size,
            training=False,
            augment=False,
        ),
    )

