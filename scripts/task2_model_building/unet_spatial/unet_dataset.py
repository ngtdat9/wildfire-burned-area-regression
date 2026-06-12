# -*- coding: utf-8 -*-
"""
U-Net Dataset Loader for Task 2: Wildfire Fire Spread Prediction.

This loader uses memory-mapped .npy cache:

tensors_npy_cache/
├── train/<fire_id>/
│   ├── fire_masks.npy   [N, H, W]
│   ├── weather.npy      [N, 6, H, W]
│   └── terrain.npy      [3, H, W]

For U-Net:
    Temporal input [T, 10, H, W] is flattened into [T*10, H, W].

Default:
    T = 4
    channels per day = 1 fire + 6 weather + 3 terrain = 10
    U-Net input channels = 4 * 10 = 40
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WildfireUNetDataset(Dataset):
    """
    Dataset for U-Net spatial baseline.

    Returns:
        x:      torch.FloatTensor [40, crop_size, crop_size]
        y_fire: torch.FloatTensor [1, crop_size, crop_size]

    Important:
        This dataset does NOT randomly split fires.
        It uses the already prepared train/val/test cache folders.
    """

    def __init__(
        self,
        cache_root: str,
        split: str,
        window_size: int = 4,
        crop_size: Optional[int] = 512,
        crop_mode: Optional[str] = None,
        positive_crop_prob: float = 0.7,
        limit_samples: Optional[int] = None,
        fill_nan_value: float = 0.0,
    ):
        """
        Args:
            cache_root:
                Path to tensors_npy_cache.

            split:
                One of: train, val, test.

            window_size:
                Number of previous days used as input.

            crop_size:
                Spatial crop size. Use 512 for training.
                If None, returns full image. Not recommended for training.

            crop_mode:
                "random", "center", or "fire_aware".
                If None:
                    train -> fire_aware
                    val/test -> fire_aware

            positive_crop_prob:
                Used only when crop_mode="fire_aware".
                Probability of choosing a fire-centered crop.
                If fire pixels are unavailable, fallback is random/center.

            limit_samples:
                Optional sample limit for debugging.

            fill_nan_value:
                Value used to replace NaN/inf in weather/terrain.
        """
        self.cache_root = Path(cache_root)
        self.split = split
        self.window_size = window_size
        self.crop_size = crop_size
        self.positive_crop_prob = positive_crop_prob
        self.fill_nan_value = fill_nan_value

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}. Expected train, val, or test.")

        if crop_mode is None:
            self.crop_mode = "fire_aware"
        else:
            if crop_mode not in {"random", "center", "fire_aware"}:
                raise ValueError("crop_mode must be 'random', 'center', or 'fire_aware'.")
            self.crop_mode = crop_mode

        self.split_dir = self.cache_root / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")

        self.fire_dirs = sorted([p for p in self.split_dir.iterdir() if p.is_dir()])

        if len(self.fire_dirs) == 0:
            raise RuntimeError(f"No fire folders found in: {self.split_dir}")

        self.samples: List[Dict] = []
        self._build_sample_index()

        if limit_samples is not None:
            self.samples = self.samples[:limit_samples]

        self._mmap_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _build_sample_index(self) -> None:
        """
        Build list of temporal windows.

        For N days and window_size = 4:
            start_day = 0 -> input days 0,1,2,3; target day 4
            start_day = 1 -> input days 1,2,3,4; target day 5
        """
        for fire_dir in self.fire_dirs:
            fire_id = fire_dir.name
            fire_masks_path = fire_dir / "fire_masks.npy"
            weather_path = fire_dir / "weather.npy"
            terrain_path = fire_dir / "terrain.npy"

            if not fire_masks_path.exists():
                continue
            if not weather_path.exists():
                continue
            if not terrain_path.exists():
                continue

            fire_masks = np.load(fire_masks_path, mmap_mode="r")
            n_days = fire_masks.shape[0]
            n_samples = n_days - self.window_size

            if n_samples <= 0:
                continue

            for start_day in range(n_samples):
                self.samples.append({
                    "fire_id": fire_id,
                    "fire_dir": fire_dir,
                    "start_day": start_day,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_fire_arrays(self, fire_id: str, fire_dir: Path) -> Dict[str, np.ndarray]:
        """
        Load memory-mapped arrays for one fire.
        """
        if fire_id not in self._mmap_cache:
            self._mmap_cache[fire_id] = {
                "fire_masks": np.load(fire_dir / "fire_masks.npy", mmap_mode="r"),
                "weather": np.load(fire_dir / "weather.npy", mmap_mode="r"),
                "terrain": np.load(fire_dir / "terrain.npy", mmap_mode="r"),
            }

        return self._mmap_cache[fire_id]

    def _crop_from_center(
        self,
        center_y: int,
        center_x: int,
        height: int,
        width: int,
    ) -> Tuple[int, int, int, int]:
        """
        Create crop coordinates around a center pixel.
        """
        crop = self.crop_size

        top = int(center_y - crop // 2)
        left = int(center_x - crop // 2)

        top = max(0, min(top, height - crop))
        left = max(0, min(left, width - crop))

        bottom = top + crop
        right = left + crop

        return top, left, bottom, right

    def _get_center_crop_coords(self, height: int, width: int) -> Tuple[int, int, int, int]:
        crop = self.crop_size

        top = (height - crop) // 2
        left = (width - crop) // 2

        bottom = top + crop
        right = left + crop

        return top, left, bottom, right

    def _get_random_crop_coords(self, height: int, width: int) -> Tuple[int, int, int, int]:
        crop = self.crop_size

        top = np.random.randint(0, height - crop + 1)
        left = np.random.randint(0, width - crop + 1)

        bottom = top + crop
        right = left + crop

        return top, left, bottom, right

    def _get_fire_aware_crop_coords(
        self,
        fire_masks: np.ndarray,
        start_day: int,
        height: int,
        width: int,
    ) -> Tuple[int, int, int, int, str]:
        """
        Pick a crop around fire pixels.

        Uses union of:
            current fire mask at day t       = start_day + window_size - 1
            target fire mask at day t + 1    = start_day + window_size

        For train:
            Randomly picks one positive fire pixel.

        For val/test:
            Uses median fire-pixel location for deterministic crop.

        If no fire pixels exist:
            fallback to random crop for train
            fallback to center crop for val/test
        """
        W = self.window_size

        current_day = start_day + W - 1
        target_day = start_day + W

        current_mask = fire_masks[current_day]
        target_mask = fire_masks[target_day]

        fire_union = (current_mask > 0) | (target_mask > 0)

        ys, xs = np.where(fire_union)

        if len(ys) == 0:
            if self.split == "train":
                top, left, bottom, right = self._get_random_crop_coords(height, width)
                return top, left, bottom, right, "fallback_random_no_fire"
            else:
                top, left, bottom, right = self._get_center_crop_coords(height, width)
                return top, left, bottom, right, "fallback_center_no_fire"

        # In training, keep some random/background crops too.
        if self.split == "train":
            use_positive = np.random.rand() < self.positive_crop_prob

            if not use_positive:
                top, left, bottom, right = self._get_random_crop_coords(height, width)
                return top, left, bottom, right, "random_background_mixed"

            idx = np.random.randint(0, len(ys))
            center_y = int(ys[idx])
            center_x = int(xs[idx])

            top, left, bottom, right = self._crop_from_center(
                center_y=center_y,
                center_x=center_x,
                height=height,
                width=width,
            )

            return top, left, bottom, right, "fire_aware_random"

        # For validation/test, deterministic crop around median fire location.
        center_y = int(np.median(ys))
        center_x = int(np.median(xs))

        top, left, bottom, right = self._crop_from_center(
            center_y=center_y,
            center_x=center_x,
            height=height,
            width=width,
        )

        return top, left, bottom, right, "fire_aware_center"

    def _get_crop_coords(
        self,
        fire_masks: np.ndarray,
        start_day: int,
        height: int,
        width: int,
    ) -> Tuple[int, int, int, int, str]:
        """
        Select crop coordinates.
        """
        if self.crop_size is None:
            return 0, 0, height, width, "full"

        crop = self.crop_size

        if height < crop or width < crop:
            raise ValueError(
                f"Image smaller than crop_size. Image={(height, width)}, crop={crop}"
            )

        if self.crop_mode == "center":
            top, left, bottom, right = self._get_center_crop_coords(height, width)
            return top, left, bottom, right, "center"

        if self.crop_mode == "random":
            top, left, bottom, right = self._get_random_crop_coords(height, width)
            return top, left, bottom, right, "random"

        if self.crop_mode == "fire_aware":
            return self._get_fire_aware_crop_coords(
                fire_masks=fire_masks,
                start_day=start_day,
                height=height,
                width=width,
            )

        raise ValueError(f"Unknown crop_mode: {self.crop_mode}")

    def _safe_float_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert to float32 and replace NaN/inf.
        """
        arr = arr.astype(np.float32, copy=False)
        arr = np.nan_to_num(
            arr,
            nan=self.fill_nan_value,
            posinf=self.fill_nan_value,
            neginf=self.fill_nan_value,
        )
        return arr

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]

        fire_id = item["fire_id"]
        fire_dir = item["fire_dir"]
        start_day = item["start_day"]

        arrays = self._load_fire_arrays(fire_id, fire_dir)

        fire_masks = arrays["fire_masks"]  # [N, H, W]
        weather = arrays["weather"]        # [N, 6, H, W]
        terrain = arrays["terrain"]        # [3, H, W]

        _, height, width = fire_masks.shape

        top, left, bottom, right, crop_source = self._get_crop_coords(
            fire_masks=fire_masks,
            start_day=start_day,
            height=height,
            width=width,
        )

        W = self.window_size
        i = start_day

        # Input fire masks: [T, H, W] -> [T, 1, crop, crop]
        x_fire = fire_masks[i:i + W, top:bottom, left:right].astype(np.float32)
        x_fire = x_fire[:, None, :, :]

        # Input weather: [T, 6, crop, crop]
        x_weather = weather[i:i + W, :, top:bottom, left:right]
        x_weather = self._safe_float_array(x_weather)

        # Static terrain: [3, crop, crop]
        x_terrain = terrain[:, top:bottom, left:right]
        x_terrain = self._safe_float_array(x_terrain)

        # Repeat terrain across T days: [T, 3, crop, crop]
        x_terrain = np.repeat(x_terrain[None, :, :, :], repeats=W, axis=0)

        # Combine per day: [T, 10, crop, crop]
        x_temporal = np.concatenate(
            [x_fire, x_weather, x_terrain],
            axis=1,
        )

        # Flatten time into channels:
        # [T, 10, crop, crop] -> [40, crop, crop]
        T, C, H, Wd = x_temporal.shape
        x_unet = x_temporal.reshape(T * C, H, Wd)

        # Target next-day fire mask: [crop, crop] -> [1, crop, crop]
        y_fire = fire_masks[i + self.window_size, top:bottom, left:right].astype(np.float32)
        y_fire = y_fire[None, :, :]

        return {
            "x": torch.from_numpy(x_unet).float(),
            "y_fire": torch.from_numpy(y_fire).float(),
            "fire_id": fire_id,
            "split": self.split,
            "start_day": start_day,
            "crop": {
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "source": crop_source,
            },
        }


def describe_sample(sample: Dict) -> None:
    """
    Helper function for debugging one sample.
    """
    x = sample["x"]
    y_fire = sample["y_fire"]

    print("fire_id:", sample["fire_id"])
    print("split:", sample["split"])
    print("start_day:", sample["start_day"])
    print("crop:", sample["crop"])
    print("x shape:", tuple(x.shape))
    print("y_fire shape:", tuple(y_fire.shape))
    print("x dtype:", x.dtype)
    print("y_fire dtype:", y_fire.dtype)
    print("x NaN count:", torch.isnan(x).sum().item())
    print("y_fire min/max:", y_fire.min().item(), y_fire.max().item())
    print("input current fire pixels:", int((x[30] > 0.5).sum().item()))
    print("target fire pixels:", int((y_fire[0] > 0.5).sum().item()))

class WildfireUNetFireFilteredDataset(WildfireUNetDataset):
    """
     Fire-filtered version of WildfireUNetDataset.

     This dataset removes temporal windows where both:
        - current fire mask is empty
        - target fire mask is empty

    It still uses the same official train/val/test folders.
    It does NOT create a new random split.
    """

    def __init__(
        self,
        *args,
        require_fire: bool = True,
        min_fire_pixels: int = 1,
        **kwargs,
    ):
        """
        Args:
            require_fire:
                If True, only keep samples where current or target fire mask
                contains at least min_fire_pixels.

            min_fire_pixels:
                Minimum number of fire pixels required in the union of:
                    current fire mask at day t
                    target fire mask at day t+1
        """
        self.require_fire = require_fire
        self.min_fire_pixels = min_fire_pixels

        super().__init__(*args, **kwargs)

    def _build_sample_index(self) -> None:
        """
        Build list of temporal windows, optionally filtering empty fire windows.

        For N days and window_size = 4:
            start_day = 0 -> input days 0,1,2,3; target day 4

        Fire-aware filtering checks:
            current day = start_day + window_size - 1
            target day  = start_day + window_size
        """
        for fire_dir in self.fire_dirs:
            fire_id = fire_dir.name

            fire_masks_path = fire_dir / "fire_masks.npy"
            weather_path = fire_dir / "weather.npy"
            terrain_path = fire_dir / "terrain.npy"

            if not fire_masks_path.exists():
                continue
            if not weather_path.exists():
                continue
            if not terrain_path.exists():
                continue

            fire_masks = np.load(fire_masks_path, mmap_mode="r")

            n_days = fire_masks.shape[0]
            n_samples = n_days - self.window_size

            if n_samples <= 0:
                continue

            for start_day in range(n_samples):
                if self.require_fire:
                    current_day = start_day + self.window_size - 1
                    target_day = start_day + self.window_size

                    current_mask = fire_masks[current_day] > 0
                    target_mask = fire_masks[target_day] > 0

                    union_pixels = int((current_mask | target_mask).sum())

                    if union_pixels < self.min_fire_pixels:
                        continue

                    self.samples.append({
                        "fire_id": fire_id,
                        "fire_dir": fire_dir,
                        "start_day": start_day,
                        "union_fire_pixels": union_pixels,
                    })

                else:
                    self.samples.append({
                        "fire_id": fire_id,
                        "fire_dir": fire_dir,
                        "start_day": start_day,
                        "union_fire_pixels": None,
                    })

class WildfireUNetNormalizedFireFilteredDataset(WildfireUNetFireFilteredDataset):
    """
    Fire-filtered U-Net dataset with channel-wise normalization.

    Fire masks:
        kept as 0/1

    Weather:
        normalized using train-set weather mean/std

    Terrain:
        normalized using train-set terrain mean/std
    """

    def __init__(
        self,
        *args,
        normalization_stats_path: str,
        eps: float = 1e-6,
        **kwargs,
    ):
        import json

        self.normalization_stats_path = Path(normalization_stats_path)
        self.eps = eps

        if not self.normalization_stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats file not found: {self.normalization_stats_path}"
            )

        with open(self.normalization_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.weather_mean = np.array(stats["weather_mean"], dtype=np.float32)
        self.weather_std = np.array(stats["weather_std"], dtype=np.float32)

        self.terrain_mean = np.array(stats["terrain_mean"], dtype=np.float32)
        self.terrain_std = np.array(stats["terrain_std"], dtype=np.float32)

        self.weather_std = np.maximum(self.weather_std, eps)
        self.terrain_std = np.maximum(self.terrain_std, eps)

        super().__init__(*args, **kwargs)

    def _normalize_weather(self, weather: np.ndarray) -> np.ndarray:
        """
        weather shape:
            [T, 6, H, W]
        """
        weather = self._safe_float_array(weather)

        mean = self.weather_mean[None, :, None, None]
        std = self.weather_std[None, :, None, None]

        weather = (weather - mean) / std

        return weather.astype(np.float32)

    def _normalize_terrain(self, terrain: np.ndarray) -> np.ndarray:
        """
        terrain shape:
            [3, H, W]
        """
        terrain = self._safe_float_array(terrain)

        mean = self.terrain_mean[:, None, None]
        std = self.terrain_std[:, None, None]

        terrain = (terrain - mean) / std

        return terrain.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]

        fire_id = item["fire_id"]
        fire_dir = item["fire_dir"]
        start_day = item["start_day"]

        arrays = self._load_fire_arrays(fire_id, fire_dir)

        fire_masks = arrays["fire_masks"]  # [N, H, W]
        weather = arrays["weather"]        # [N, 6, H, W]
        terrain = arrays["terrain"]        # [3, H, W]

        _, height, width = fire_masks.shape

        top, left, bottom, right, crop_source = self._get_crop_coords(
            fire_masks=fire_masks,
            start_day=start_day,
            height=height,
            width=width,
        )

        W = self.window_size
        i = start_day

        # Fire masks stay 0/1
        x_fire = fire_masks[i:i + W, top:bottom, left:right].astype(np.float32)
        x_fire = x_fire[:, None, :, :]  # [T, 1, H, W]

        # Weather normalization
        x_weather = weather[i:i + W, :, top:bottom, left:right]
        x_weather = self._normalize_weather(x_weather)

        # Terrain normalization
        x_terrain = terrain[:, top:bottom, left:right]
        x_terrain = self._normalize_terrain(x_terrain)

        # Repeat terrain across T days
        x_terrain = np.repeat(x_terrain[None, :, :, :], repeats=W, axis=0)

        # Combine: [T, 10, H, W]
        x_temporal = np.concatenate(
            [x_fire, x_weather, x_terrain],
            axis=1,
        )

        # Flatten time into channels: [40, H, W]
        T, C, H, Wd = x_temporal.shape
        x_unet = x_temporal.reshape(T * C, H, Wd)

        # Target next-day fire mask
        y_fire = fire_masks[i + self.window_size, top:bottom, left:right].astype(np.float32)
        y_fire = y_fire[None, :, :]

        return {
            "x": torch.from_numpy(x_unet).float(),
            "y_fire": torch.from_numpy(y_fire).float(),
            "fire_id": fire_id,
            "split": self.split,
            "start_day": start_day,
            "crop": {
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "source": crop_source,
            },
        }
