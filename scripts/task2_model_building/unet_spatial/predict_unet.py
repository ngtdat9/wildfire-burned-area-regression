# -*- coding: utf-8 -*-
"""
Save prediction visualizations from normalized fire-filtered U-Net checkpoint.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    CACHE_ROOT,
    STATS_PATH,
    SELECTED_THRESHOLD,
    CROP_SIZE,
    BASE_CHANNELS,
    WINDOW_SIZE,
)

from unet_dataset import WildfireUNetNormalizedFireFilteredDataset
from unet_model import UNetFire
from utils.visualization import save_fire_prediction_figure


def parse_indices(raw: str):
    if not raw:
        return None

    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Save U-Net prediction visualizations.")

    parser.add_argument("--cache-root", type=str, default=str(CACHE_ROOT))
    parser.add_argument("--normalization-stats", type=str, default=str(STATS_PATH))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=SELECTED_THRESHOLD)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--indices", type=str, default="")

    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--min-fire-pixels", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Normalized U-Net Prediction Visualization")
    print("=" * 80)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", args.checkpoint)
    print("Split:", args.split)
    print("Threshold:", args.threshold)
    print("Output dir:", output_dir)
    print("=" * 80)

    dataset = WildfireUNetNormalizedFireFilteredDataset(
        cache_root=args.cache_root,
        split=args.split,
        normalization_stats_path=args.normalization_stats,
        window_size=args.window_size,
        crop_size=args.crop_size,
        crop_mode="fire_aware",
        positive_crop_prob=1.0,
        require_fire=True,
        min_fire_pixels=args.min_fire_pixels,
        limit_samples=None,
    )

    model = UNetFire(
        in_channels=40,
        base_channels=args.base_channels,
        out_channels=1,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint epoch:", checkpoint.get("epoch"))
    print("Dataset samples:", len(dataset))

    selected_indices = parse_indices(args.indices)

    if selected_indices is None:
        selected_indices = list(range(min(args.num_examples, len(dataset))))

    rows = []

    with torch.no_grad():
        for order_idx, sample_idx in enumerate(selected_indices):
            sample = dataset[sample_idx]

            x = sample["x"].unsqueeze(0).to(device)
            y_fire = sample["y_fire"].unsqueeze(0).to(device)

            logits = model(x)

            fire_id = sample["fire_id"]
            start_day = int(sample["start_day"])

            output_path = output_dir / (
                f"{args.split}_{sample_idx:03d}_fire_{fire_id}_day_{start_day}_th{args.threshold}.png"
            )

            stats = save_fire_prediction_figure(
                x=x.cpu(),
                y_fire=y_fire.cpu(),
                logits=logits.cpu(),
                output_path=output_path,
                threshold=args.threshold,
                title=(
                    f"{args.split.upper()} | idx={sample_idx}, "
                    f"fire={fire_id}, start_day={start_day}, threshold={args.threshold}"
                ),
            )

            stats["order_idx"] = order_idx
            stats["sample_idx"] = sample_idx
            stats["fire_id"] = fire_id
            stats["start_day"] = start_day
            rows.append(stats)

            print("Saved:", output_path)

    csv_path = output_dir / "prediction_stats.csv"

    if rows:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print("Saved stats:", csv_path)


if __name__ == "__main__":
    main()