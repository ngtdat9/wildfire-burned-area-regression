# -*- coding: utf-8 -*-
"""
Evaluate normalized fire-filtered U-Net checkpoint.

Supports:
    1. validation threshold sweep
    2. fixed-threshold validation/test evaluation

Important:
    Use validation split for threshold selection.
    Use test split only once with selected threshold.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from config import (
    CACHE_ROOT,
    STATS_PATH,
    SELECTED_THRESHOLD,
    CROP_SIZE,
    BASE_CHANNELS,
    WINDOW_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
)

from unet_dataset import WildfireUNetNormalizedFireFilteredDataset
from unet_model import UNetFire
from utils.metrics import (
    binary_segmentation_counts,
    binary_segmentation_metrics_from_counts,
    merge_counts,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device, base_channels: int = 16):
    model = UNetFire(
        in_channels=40,
        base_channels=base_channels,
        out_channels=1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate_threshold(
    model,
    loader,
    device,
    threshold: float,
    save_sample_rows: bool = False,
) -> Dict:
    total_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    sample_rows = []

    for idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        y_fire = batch["y_fire"].to(device)

        logits = model(x)

        counts = binary_segmentation_counts(
            logits=logits.detach(),
            targets=y_fire.detach(),
            threshold=threshold,
        )

        total_counts = merge_counts(total_counts, counts)

        if save_sample_rows:
            metrics = binary_segmentation_metrics_from_counts(counts)

            fire_id = batch["fire_id"][0]
            start_day = int(batch["start_day"][0])

            pred_pixels = int((torch.sigmoid(logits) >= threshold).sum().item())
            target_pixels = int(y_fire.sum().item())

            sample_rows.append({
                "idx": idx,
                "fire_id": fire_id,
                "start_day": start_day,
                "threshold": threshold,
                **counts,
                **metrics,
                "target_pixels": target_pixels,
                "pred_pixels": pred_pixels,
            })

    metrics = binary_segmentation_metrics_from_counts(total_counts)
    metrics.update(total_counts)
    metrics["threshold"] = threshold

    return {
        "summary": metrics,
        "sample_rows": sample_rows,
    }


def save_results_csv(results: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def save_json(data: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_thresholds(raw: str):
    if raw:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    return [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90,
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate normalized U-Net checkpoint.")

    parser.add_argument("--cache-root", type=str, default=str(CACHE_ROOT))
    parser.add_argument("--normalization-stats", type=str, default=str(STATS_PATH))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--mode", type=str, default="sweep", choices=["sweep", "fixed"])

    parser.add_argument("--threshold", type=float, default=SELECTED_THRESHOLD)
    parser.add_argument("--thresholds", type=str, default="")

    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--min-fire-pixels", type=int, default=1)
    parser.add_argument("--save-sample-metrics", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Normalized U-Net Evaluation")
    print("=" * 80)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", args.checkpoint)
    print("Split:", args.split)
    print("Mode:", args.mode)
    print("Output dir:", output_dir)
    print("=" * 80)

    ds = WildfireUNetNormalizedFireFilteredDataset(
        cache_root=args.cache_root,
        split=args.split,
        normalization_stats_path=args.normalization_stats,
        window_size=args.window_size,
        crop_size=args.crop_size,
        crop_mode="fire_aware",
        positive_crop_prob=1.0,
        require_fire=True,
        min_fire_pixels=args.min_fire_pixels,
        limit_samples=args.limit_samples,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model, checkpoint = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        base_channels=args.base_channels,
    )

    print("Loaded checkpoint epoch:", checkpoint.get("epoch"))
    print("Samples:", len(ds))

    if args.mode == "sweep":
        thresholds = parse_thresholds(args.thresholds)
        results = []

        for th in thresholds:
            output = evaluate_threshold(
                model=model,
                loader=loader,
                device=device,
                threshold=th,
                save_sample_rows=False,
            )

            metrics = output["summary"]
            results.append(metrics)

            print(
                f"Threshold {th:.2f} | "
                f"IoU: {metrics['iou']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f}"
            )

        best_by_f1 = max(results, key=lambda x: x["f1"])
        best_by_iou = max(results, key=lambda x: x["iou"])

        print("\nBest by F1:")
        print(best_by_f1)

        print("\nBest by IoU:")
        print(best_by_iou)

        sweep_csv = output_dir / f"threshold_sweep_{args.split}.csv"
        save_results_csv(results, sweep_csv)

        summary_json = output_dir / f"threshold_sweep_{args.split}_summary.json"
        save_json(
            {
                "split": args.split,
                "checkpoint": args.checkpoint,
                "samples": len(ds),
                "best_by_f1": best_by_f1,
                "best_by_iou": best_by_iou,
                "note": "Threshold selected using validation split only.",
            },
            summary_json,
        )

        print("Saved sweep CSV:", sweep_csv)
        print("Saved summary JSON:", summary_json)

    else:
        output = evaluate_threshold(
            model=model,
            loader=loader,
            device=device,
            threshold=args.threshold,
            save_sample_rows=args.save_sample_metrics,
        )

        metrics = output["summary"]

        print(f"\nFixed-threshold metrics at {args.threshold}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        summary_json = output_dir / f"{args.split}_summary_threshold{args.threshold}.json"
        save_json(
            {
                "split": args.split,
                "checkpoint": args.checkpoint,
                "samples": len(ds),
                "threshold": args.threshold,
                "metrics": metrics,
                "note": "Filtered fire-aware patch-level evaluation.",
            },
            summary_json,
        )

        print("Saved summary JSON:", summary_json)

        if args.save_sample_metrics:
            sample_csv = output_dir / f"{args.split}_sample_metrics_threshold{args.threshold}.csv"
            save_results_csv(output["sample_rows"], sample_csv)
            print("Saved sample metrics CSV:", sample_csv)


if __name__ == "__main__":
    main()