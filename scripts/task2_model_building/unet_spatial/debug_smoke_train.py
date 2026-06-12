# -*- coding: utf-8 -*-
"""
Local smoke test for U-Net training step.

This does NOT use real wildfire data.
It only checks:

1. model forward pass
2. loss calculation
3. backward pass
4. optimizer step

Use this locally before running real Colab training.
"""

import torch
import torch.optim as optim

from unet_model import UNetFire
from utils.losses import BCEDiceLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetFire(
        in_channels=40,
        base_channels=16,
        out_channels=1,
    ).to(device)

    criterion = BCEDiceLoss(
        bce_weight=1.0,
        dice_weight=1.0,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Fake batch
    x = torch.randn(1, 40, 256, 256).to(device)
    y = torch.randint(0, 2, (1, 1, 256, 256)).float().to(device)

    model.train()

    for step in range(3):
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        print(f"Step {step + 1} | Loss: {loss.item():.6f}")

    print("Smoke training step completed successfully.")
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)


if __name__ == "__main__":
    main()