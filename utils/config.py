from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config

