#!/usr/bin/env python3
"""
Batch image generator for experiment dataset.
Integer ticks only; all black curves/axes/grid; origin labeled once; digital degradation profiles.
"""

import argparse
import random
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from generator.core import generate_one
from generator.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate full-page math graph images and ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  --count N       number of images to generate (default: 10)
  --config PATH   path to generator config YAML
  --seed SEED     random seed for reproducibility
  --out-dir DIR   base output dir; images/ and ground_truth/ under it
        """,
    )
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=None, help="Base output dir")
    args = parser.parse_args()

    config = load_config(args.config)
    out_cfg = config.get("output", {})
    if args.out_dir is not None:
        base = Path(args.out_dir)
        image_dir = base / "images"
        gt_dir = base / "ground_truth"
    else:
        image_dir = Path(out_cfg.get("image_dir", "dataset/images"))
        gt_dir = Path(out_cfg.get("gt_dir", "dataset/ground_truth"))

    image_dir = image_dir if image_dir.is_absolute() else _root / image_dir
    gt_dir = gt_dir if gt_dir.is_absolute() else _root / gt_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    fmt = out_cfg.get("image_format", "png")
    n = args.count
    if args.seed is not None:
        random.seed(args.seed)

    for i in range(n):
        stem = f"img_{i:05d}"
        img_path = image_dir / f"{stem}.{fmt}"
        gt_path = gt_dir / f"{stem}.json"
        seed = (args.seed + i) if args.seed is not None else None
        generate_one(
            config,
            output_image_path=img_path,
            output_gt_path=gt_path,
            layout_override=None,
            seed=seed,
        )
        print(f"  wrote {img_path.name} + {gt_path.name}")

    print(f"Done. {n} images -> {image_dir}")
    print(f"       {n} gt files -> {gt_dir}")


if __name__ == "__main__":
    main()
