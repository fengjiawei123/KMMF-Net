from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import discover_pairs
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["dataset"]["root"] = args.data_root
    records = discover_pairs(config["dataset"])
    groups = Counter(record.group for record in records)
    print(f"pairs={len(records)} groups={dict(groups)}")
    for record in records[:5]:
        print(
            f"{record.group}: {record.source_a.name} <-> {record.source_b.name} "
            f"color_from={record.color_from}"
        )


if __name__ == "__main__":
    main()
