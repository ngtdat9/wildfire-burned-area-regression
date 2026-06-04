# -*- coding: utf-8 -*-
import numpy as np

path = "/content/drive/MyDrive/Wildfire Area Regression /Dataset/tensors/train/20568194.npz"

data = np.load(path)
print(data.files)

for key in data.files:
    print(key, data[key].shape, data[key].dtype, data[key].min(), data[key].max())

import numpy as np

terrain = data["terrain"]

print("NaN count:", np.isnan(terrain).sum())
print("Total pixels:", terrain.size)
print("NaN ratio:", np.isnan(terrain).sum() / terrain.size)

print("nanmin:", np.nanmin(terrain))
print("nanmax:", np.nanmax(terrain))
print("nanmean:", np.nanmean(terrain))

from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class WildfireTemporalDataset(Dataset):
    """
    Task 2 DataLoader for temporal wildfire spread prediction.

    Output:
        x:      [T, 10, H, W]
                T = 4
                channels = 1 fire mask + 6 weather + 3 terrain

        y_fire: [1, H, W]
                next-day fire mask
    """

    def __init__(
    self,
    manifest_path,
    tensor_root,
    split,
    window_size=4,
    fill_weather_nan=0.0,
):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = self.manifest_path.parent
        self.split = split
        self.window_size = window_size
        self.fill_weather_nan = fill_weather_nan

        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.fire_records = [
            fire for fire in self.manifest["fires"]
            if fire.get("success") is True and fire.get("split") == split
        ]

        self.samples = []

        for fire in self.fire_records:
            fire_id = fire["fire_id"]
            n_samples = fire["n_samples"]

            for start_day in range(n_samples):
                self.samples.append({
                    "fire_id": fire_id,
                    "split": split,
                    "start_day": start_day,
                    "height": fire["height"],
                    "width": fire["width"],
                    "n_fire_days": fire["n_fire_days"],
                })

    def __len__(self):
        return len(self.samples)

    def _get_tensor_path(self, fire_id):
        return self.dataset_root / self.split / f"{fire_id}.npz"

    def _fill_terrain_nan_with_band_median(self, terrain):
        """
        terrain: [3, H, W]

        Replace NaN/inf values in each terrain band using that band's median.
        This is better than global zero filling because elevation, slope,
        and aspect have different value ranges.
        """
        terrain = terrain.copy()

        for band_idx in range(terrain.shape[0]):
            band = terrain[band_idx]

            valid_mask = np.isfinite(band)

            if valid_mask.any():
                median_value = np.median(band[valid_mask])
            else:
                median_value = 0.0

            terrain[band_idx] = np.nan_to_num(
                band,
                nan=median_value,
                posinf=median_value,
                neginf=median_value,
            )

        return terrain

    def __getitem__(self, idx):
        item = self.samples[idx]

        fire_id = item["fire_id"]
        start_day = item["start_day"]
        tensor_path = self._get_tensor_path(fire_id)

        if not tensor_path.exists():
            raise FileNotFoundError(f"Tensor file not found: {tensor_path}")

        data = np.load(tensor_path)

        fire_masks = data["fire_masks"].astype(np.float32)  # [N, H, W]
        weather = data["weather"].astype(np.float32)        # [N, 6, H, W]
        terrain = data["terrain"].astype(np.float32)        # [3, H, W]

        terrain = self._fill_terrain_nan_with_band_median(terrain)

        weather = np.nan_to_num(
            weather,
            nan=self.fill_weather_nan,
            posinf=self.fill_weather_nan,
            neginf=self.fill_weather_nan,
        )

        W = self.window_size
        i = start_day

        x_fire = fire_masks[i:i + W]          # [T, H, W]
        x_fire = x_fire[:, None, :, :]        # [T, 1, H, W]

        x_weather = weather[i:i + W]          # [T, 6, H, W]

        x_terrain = np.repeat(
            terrain[None, :, :, :],
            repeats=W,
            axis=0
        )                                    # [T, 3, H, W]

        x = np.concatenate(
            [x_fire, x_weather, x_terrain],
            axis=1
        )                                    # [T, 10, H, W]

        y_fire = fire_masks[i + W]            # [H, W]
        y_fire = y_fire[None, :, :]           # [1, H, W]

        return {
            "x": torch.from_numpy(x).float(),
            "y_fire": torch.from_numpy(y_fire).float(),
            "fire_id": fire_id,
            "split": self.split,
            "start_day": start_day,
        }

from torch.utils.data import DataLoader

manifest_path = "/content/drive/MyDrive/Wildfire_Area_Regression /Dataset/tensors/tensor_manifest.json"
tensor_root = "/content/drive/MyDrive/Wildfire_Area_Regression /Dataset/tensors"

train_ds = WildfireTemporalDataset(
    manifest_path=manifest_path,
    tensor_root=tensor_root,
    split="train",
    window_size=4,
)

print("Train samples:", len(train_ds))

sample = train_ds[0]

print("x shape:", sample["x"].shape)
print("y_fire shape:", sample["y_fire"].shape)
print("fire_id:", sample["fire_id"])
print("start_day:", sample["start_day"])
print("x NaN count:", torch.isnan(sample["x"]).sum().item())
print("y_fire min/max:", sample["y_fire"].min().item(), sample["y_fire"].max().item())