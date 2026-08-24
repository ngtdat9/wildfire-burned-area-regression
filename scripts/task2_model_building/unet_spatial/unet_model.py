# -*- coding: utf-8 -*-
"""
U-Net model for Task 2 wildfire next-day fire mask prediction.

Input:
    x: [B, 40, H, W]

Where:
    40 = 4 days × 10 channels per day
       = 4 × (1 fire mask + 6 weather + 3 terrain)

Output:
    fire_logits: [B, 1, H, W]

The output is raw logits. Apply torch.sigmoid() during evaluation/inference.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two convolution layers:
        Conv2d -> BatchNorm -> ReLU
        Conv2d -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block:
        MaxPool2d -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block:
        ConvTranspose2d -> concatenate skip connection -> DoubleConv
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Crop skip if needed due to odd shapes.
        # For 512x512 this should not be necessary, but this keeps the model safer.
        if x.shape[-2:] != skip.shape[-2:]:
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]

            skip = skip[
                :,
                :,
                diff_h // 2: skip.shape[-2] - (diff_h - diff_h // 2),
                diff_w // 2: skip.shape[-1] - (diff_w - diff_w // 2),
            ]

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetFire(nn.Module):
    """
    Single-head U-Net for next-day active fire mask prediction.

    Args:
        in_channels:
            Number of input channels. For our U-Net:
                4 days × 10 channels = 40

        base_channels:
            Controls model size.
            Use 16 for memory-safe Colab training.
            Use 32 only if GPU memory allows.
    """

    def __init__(
        self,
        in_channels: int = 40,
        base_channels: int = 16,
        out_channels: int = 1,
    ):
        super().__init__()

        b = base_channels

        self.enc1 = DoubleConv(in_channels, b)        # 512 -> 512
        self.enc2 = DownBlock(b, b * 2)               # 512 -> 256
        self.enc3 = DownBlock(b * 2, b * 4)           # 256 -> 128
        self.enc4 = DownBlock(b * 4, b * 8)           # 128 -> 64

        self.bottleneck = DownBlock(b * 8, b * 16)    # 64 -> 32

        self.up4 = UpBlock(b * 16, b * 8, b * 8)      # 32 -> 64
        self.up3 = UpBlock(b * 8, b * 4, b * 4)       # 64 -> 128
        self.up2 = UpBlock(b * 4, b * 2, b * 2)       # 128 -> 256
        self.up1 = UpBlock(b * 2, b, b)               # 256 -> 512

        self.fire_head = nn.Conv2d(
            b,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 40, H, W]

        Returns:
            fire_logits: [B, 1, H, W]
        """
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.bottleneck(s4)

        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        fire_logits = self.fire_head(x)

        return fire_logits


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNetFire(
        in_channels=40,
        base_channels=16,
        out_channels=1,
    )

    x = torch.randn(1, 40, 512, 512)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Trainable parameters:", count_parameters(model))