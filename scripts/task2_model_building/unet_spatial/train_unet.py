# -*- coding: utf-8 -*-
"""
Train normalized fire-filtered U-Net spatial baseline.

This script matches the final Colab pipeline:

Dataset:
    WildfireUNetNormalizedFireFilteredDataset

Input:
    x:      [B, 40, 512, 512]

Target:
    y_fire: [B, 1, 512, 512]

Evaluation:
    filtered fire-aware patch-level validation

Checkpoint saving:
    latest.pt overwritten every epoch
    best.pt saved only when validation loss improves
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from config import (
    CACHE_ROOT,
    STATS_PATH,
    UNET_WORKSPACE,
    FULL_RUN_NAME,
    EPOCHS,
    BATCH_SIZE,
    CROP_SIZE,
    BASE_CHANNELS,
    WINDOW_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BCE_WEIGHT,
    DICE_WEIGHT,
    SELECTED_THRESHOLD,
    POSITIVE_CROP_PROB,
    MIN_FIRE_PIXELS,
    NUM_WORKERS,
)

from unet_dataset import WildfireUNetNormalizedFireFilteredDataset
from unet_model import UNetFire, count_parameters
from utils.losses import BCEDiceLoss
from utils.metrics import (
    binary_segmentation_counts,
    binary_segmentation_metrics_from_counts,
    merge_counts,
)


def get_device(require_cuda: bool = False) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA GPU is required but not available.")

    return device


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    threshold: float,
    epoch: int,
    print_every: int = 100,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_batches = 0
    total_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y_fire = batch["y_fire"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        fire_logits = model(x)
        loss = criterion(fire_logits, y_fire)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        counts = binary_segmentation_counts(
            logits=fire_logits.detach(),
            targets=y_fire.detach(),
            threshold=threshold,
        )
        total_counts = merge_counts(total_counts, counts)

        if batch_idx % print_every == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / max(total_batches, 1)
    metrics = binary_segmentation_metrics_from_counts(total_counts)

    return {
        "loss": avg_loss,
        **metrics,
        **total_counts,
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_batches = 0
    total_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y_fire = batch["y_fire"].to(device, non_blocking=True)

        fire_logits = model(x)
        loss = criterion(fire_logits, y_fire)

        total_loss += loss.item()
        total_batches += 1

        counts = binary_segmentation_counts(
            logits=fire_logits.detach(),
            targets=y_fire.detach(),
            threshold=threshold,
        )
        total_counts = merge_counts(total_counts, counts)

    avg_loss = total_loss / max(total_batches, 1)
    metrics = binary_segmentation_metrics_from_counts(total_counts)

    return {
        "loss": avg_loss,
        **metrics,
        **total_counts,
    }


def save_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    args,
    config_dict: Dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "run_name": args.run_name,
        "args": vars(args),
        "config": config_dict,
        "dataset_note": "Full normalized fire-filtered fire-aware patch dataset.",
    }

    torch.save(checkpoint, checkpoint_path)


def append_or_write_log(log_path: Path, rows):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train normalized fire-filtered U-Net wildfire fire-mask model."
    )

    parser.add_argument("--cache-root", type=str, default=str(CACHE_ROOT))
    parser.add_argument("--normalization-stats", type=str, default=str(STATS_PATH))
    parser.add_argument("--output-dir", type=str, default=str(UNET_WORKSPACE))

    parser.add_argument("--run-name", type=str, default=FULL_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)

    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)

    parser.add_argument("--bce-weight", type=float, default=BCE_WEIGHT)
    parser.add_argument("--dice-weight", type=float, default=DICE_WEIGHT)

    parser.add_argument("--threshold", type=float, default=SELECTED_THRESHOLD)
    parser.add_argument("--positive-crop-prob", type=float, default=POSITIVE_CROP_PROB)
    parser.add_argument("--min-fire-pixels", type=int, default=MIN_FIRE_PIXELS)

    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)

    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume-latest", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(require_cuda=args.require_cuda)

    output_dir = Path(args.output_dir)
    run_dir = output_dir / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    report_dir = run_dir / "reports"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = checkpoint_dir / f"{args.run_name}_latest.pt"
    best_ckpt = checkpoint_dir / f"{args.run_name}_best.pt"
    log_path = log_dir / "full_training_log.csv"
    config_path = run_dir / "training_config.json"

    config_dict = {
        "run_name": args.run_name,
        "cache_root": args.cache_root,
        "normalization_stats": args.normalization_stats,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "base_channels": args.base_channels,
        "window_size": args.window_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "bce_weight": args.bce_weight,
        "dice_weight": args.dice_weight,
        "threshold": args.threshold,
        "positive_crop_prob": args.positive_crop_prob,
        "min_fire_pixels": args.min_fire_pixels,
        "evaluation_type": "filtered fire-aware patch-level evaluation",
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 80)
    print("Normalized Fire-Filtered U-Net Training")
    print("=" * 80)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Cache root:", args.cache_root)
    print("Normalization stats:", args.normalization_stats)
    print("Run dir:", run_dir)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Threshold for epoch metrics:", args.threshold)
    print("=" * 80)

    train_ds = WildfireUNetNormalizedFireFilteredDataset(
        cache_root=args.cache_root,
        split="train",
        normalization_stats_path=args.normalization_stats,
        window_size=args.window_size,
        crop_size=args.crop_size,
        crop_mode="fire_aware",
        positive_crop_prob=args.positive_crop_prob,
        require_fire=True,
        min_fire_pixels=args.min_fire_pixels,
        limit_samples=args.limit_train_samples,
    )

    val_ds = WildfireUNetNormalizedFireFilteredDataset(
        cache_root=args.cache_root,
        split="val",
        normalization_stats_path=args.normalization_stats,
        window_size=args.window_size,
        crop_size=args.crop_size,
        crop_mode="fire_aware",
        positive_crop_prob=1.0,
        require_fire=True,
        min_fire_pixels=args.min_fire_pixels,
        limit_samples=args.limit_val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    model = UNetFire(
        in_channels=40,
        base_channels=args.base_channels,
        out_channels=1,
    ).to(device)

    print("Trainable parameters:", count_parameters(model))

    criterion = BCEDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = -1
    log_rows = []

    if args.resume_latest and latest_ckpt.exists():
        print("Resuming from latest checkpoint:", latest_ckpt)
        checkpoint = torch.load(latest_ckpt, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["val_metrics"]["loss"])
        print("Resume start epoch:", start_epoch)

    start_time_all = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=args.threshold,
            epoch=epoch,
            print_every=100,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        scheduler.step()

        epoch_time_min = (time.time() - epoch_start) / 60.0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        print(f"Learning rate: {current_lr:.8f}")
        print(f"Epoch time: {epoch_time_min:.2f} minutes")

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_accuracy": train_metrics["accuracy"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": current_lr,
            "epoch_time_min": epoch_time_min,
        }

        log_rows.append(row)

        save_checkpoint(
            checkpoint_path=latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
            config_dict=config_dict,
        )
        print("Saved latest checkpoint:", latest_ckpt)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            save_checkpoint(
                checkpoint_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
                config_dict=config_dict,
            )
            print("Saved BEST checkpoint:", best_ckpt)

        append_or_write_log(log_path, log_rows)
        print("Updated training log:", log_path)

    total_time_min = (time.time() - start_time_all) / 60.0

    print("\n" + "=" * 80)
    print("Training complete.")
    print("=" * 80)
    print("Best epoch:", best_epoch)
    print("Best val loss:", best_val_loss)
    print("Total training time:", total_time_min, "minutes")
    print("Latest checkpoint:", latest_ckpt)
    print("Best checkpoint:", best_ckpt)
    print("Training log:", log_path)


if __name__ == "__main__":
    main()