# -*- coding: utf-8 -*-
"""
Metrics for binary wildfire fire-mask segmentation.

Model output:
    logits: [B, 1, H, W]

Ground truth:
    targets: [B, 1, H, W]
"""

from typing import Dict

import torch


def binary_predictions_from_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert raw logits into binary predictions.

    Args:
        logits:
            Raw model output [B, 1, H, W]

        threshold:
            Probability threshold.

    Returns:
        preds:
            Binary tensor [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return preds


def binary_segmentation_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute TP, FP, FN, TN for binary segmentation.
    """
    preds = binary_predictions_from_logits(logits, threshold=threshold)

    preds = preds.view(-1)
    targets = targets.view(-1).float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def binary_segmentation_metrics_from_counts(
    counts: Dict[str, float],
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute IoU, precision, recall, and F1 from counts.
    """
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Directly compute metrics from logits and targets.
    """
    counts = binary_segmentation_counts(
        logits=logits,
        targets=targets,
        threshold=threshold,
    )

    metrics = binary_segmentation_metrics_from_counts(counts)
    metrics.update(counts)

    return metrics


def merge_counts(total: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    """
    Add TP/FP/FN/TN counts across batches.
    """
    for key in ["tp", "fp", "fn", "tn"]:
        total[key] = total.get(key, 0.0) + new.get(key, 0.0)

    return total


if __name__ == "__main__":
    logits = torch.randn(1, 1, 512, 512)
    targets = torch.randint(0, 2, (1, 1, 512, 512)).float()

    metrics = binary_segmentation_metrics(
        logits=logits,
        targets=targets,
        threshold=0.5,
    )

    print(metrics)