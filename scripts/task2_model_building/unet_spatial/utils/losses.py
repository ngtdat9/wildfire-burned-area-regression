# -*- coding: utf-8 -*-
"""
Loss functions for U-Net wildfire fire-mask prediction.

Main loss:
    BCEWithLogitsLoss + Dice Loss

Input:
    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Dice score measures overlap between prediction and target.
    Dice loss = 1 - Dice score.

    This is useful for fire segmentation because fire pixels may be rare.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:
                Raw model output, shape [B, 1, H, W]

            targets:
                Ground truth mask, shape [B, 1, H, W], values 0 or 1
        """
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss and DiceLoss.

    total_loss = BCE + dice_weight * DiceLoss
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss


if __name__ == "__main__":
    criterion = BCEDiceLoss()

    logits = torch.randn(1, 1, 512, 512)
    targets = torch.randint(0, 2, (1, 1, 512, 512)).float()

    loss = criterion(logits, targets)

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)
    print("Loss:", loss.item())