# -*- coding: utf-8 -*-
"""
Visualization utilities for U-Net wildfire predictions.

Overlay color meaning:
    black  = correct background
    yellow = correct overlap
    red    = false positive
    green  = missed fire
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def make_prediction_overlay(gt_fire: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*gt_fire.shape, 3), dtype=np.float32)

    gt = gt_fire > 0.5
    pred = pred_mask > 0.5

    overlay[gt & ~pred] = [0, 1, 0]      # missed fire
    overlay[pred & ~gt] = [1, 0, 0]      # false positive
    overlay[gt & pred] = [1, 1, 0]       # correct overlap

    return overlay


def compute_binary_metrics(gt_fire: np.ndarray, pred_mask: np.ndarray):
    gt = gt_fire > 0.5
    pred = pred_mask > 0.5

    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def save_fire_prediction_figure(
    x: torch.Tensor,
    y_fire: torch.Tensor,
    logits: torch.Tensor,
    output_path,
    threshold: float = 0.65,
    title: str = "",
):
    """
    Save a 4-panel prediction visualization.

    Panels:
        1. Input current fire mask t
        2. Ground truth next-day fire mask t+1
        3. Predicted probability map
        4. Overlay:
            yellow = correct overlap
            red    = false positive
            green  = missed fire

    Args:
        x:
            Input tensor [40, H, W] or [1, 40, H, W]

        y_fire:
            Ground truth [1, H, W] or [1, 1, H, W]

        logits:
            Model output [1, H, W] or [1, 1, H, W]

        output_path:
            Save path.

        threshold:
            Probability threshold.

        title:
            Optional figure title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if x.dim() == 4:
        x = x[0]

    if y_fire.dim() == 4:
        y_fire = y_fire[0]

    if logits.dim() == 4:
        logits = logits[0]

    current_fire = x[30].detach().cpu().numpy()
    gt_fire = y_fire[0].detach().cpu().numpy()

    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    pred_mask = (probs >= threshold).astype(np.float32)

    overlay = make_prediction_overlay(gt_fire, pred_mask)
    metrics = compute_binary_metrics(gt_fire, pred_mask)

    current_pixels = int((current_fire > 0.5).sum())
    target_pixels = int((gt_fire > 0.5).sum())
    pred_pixels = int(pred_mask.sum())

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(current_fire, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Input Fire Mask t\npixels={current_pixels}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt_fire, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Ground Truth t+1\npixels={target_pixels}")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(probs, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(
        "Predicted Probability\n"
        f"mean={probs.mean():.3f}, max={probs.max():.3f}"
    )
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title(
        f"Overlay @ {threshold}\n"
        f"IoU={metrics['iou']:.3f}, F1={metrics['f1']:.3f}"
    )
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "current_fire_pixels": current_pixels,
        "target_fire_pixels": target_pixels,
        "pred_fire_pixels": pred_pixels,
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "prob_mean": float(probs.mean()),
        "prob_median": float(np.median(probs)),
        "threshold": threshold,
        "output_path": str(output_path),
        **metrics,
    }