"""
visualize_fire_mask_on_basemap.py
=================================
Plot one aligned fire-mask band on an OpenStreetMap basemap and save PNG.

Usage (from project root):

    python scripts/visualize_fire_mask_on_basemap.py --fire-id 20562846
    python scripts/visualize_fire_mask_on_basemap.py --fire-id 20562846 --band middle
    python scripts/visualize_fire_mask_on_basemap.py --fire-id 20562846 --band 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

DEFAULT_INPUT_DIR = Path("data/aligned_compressed")
FIRE_MASK_NAME = "labels/fire_mask_stacked_30m_aligned.tif"
DEFAULT_OUTPUT_DIR = Path("outputs/visualizations")
DST_CRS = CRS.from_epsg(3857)


def parse_band_arg(band_arg: str, band_count: int) -> int:
    """Return 1-based band index from CLI string."""
    key = band_arg.strip().lower()
    if key == "first":
        return 1
    if key == "middle":
        return max(1, band_count // 2)
    if key == "last":
        return band_count
    try:
        idx = int(band_arg)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --band value: {band_arg!r}. "
            "Use first, middle, last, or an integer."
        ) from exc
    if idx < 1 or idx > band_count:
        raise ValueError(
            f"Band {idx} out of range (1..{band_count})."
        )
    return idx


def reproject_band_to_web_mercator(
    data: np.ndarray,
    src_crs,
    src_transform,
) -> tuple[np.ndarray, object]:
    """Reproject a single 2-D band to EPSG:3857."""
    height, width = data.shape
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        DST_CRS,
        width,
        height,
        *rasterio.transform.array_bounds(height, width, src_transform),
    )
    dst = np.zeros((dst_height, dst_width), dtype=data.dtype)
    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=DST_CRS,
        resampling=Resampling.nearest,
    )
    return dst, dst_transform


def burned_pixel_count(data: np.ndarray, nodata) -> int:
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    return int(np.sum(valid & (data == 1)))


def default_output_path(fire_id: str, band_label: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{fire_id}_fire_mask_{band_label}.png"


def visualize(
    fire_id: str,
    input_dir: Path,
    band_arg: str,
    output_path: Path | None,
    dpi: int,
    figsize: tuple[float, float],
) -> Path:
    mask_path = input_dir / fire_id / FIRE_MASK_NAME
    if not mask_path.is_file():
        raise FileNotFoundError(f"Fire mask not found: {mask_path}")

    with rasterio.open(mask_path) as src:
        band_count = src.count
        band_idx = parse_band_arg(band_arg, band_count)
        data = src.read(band_idx)
        nodata = src.nodata
        src_crs = src.crs
        src_transform = src.transform

        if src.descriptions and len(src.descriptions) >= band_idx:
            band_desc = src.descriptions[band_idx - 1] or f"band {band_idx}"
        else:
            band_desc = f"band {band_idx}"

    burned = burned_pixel_count(data, nodata)
    reprojected, dst_transform = reproject_band_to_web_mercator(
        data, src_crs, src_transform
    )

    height, width = reprojected.shape
    left, bottom, right, top = array_bounds(height, width, dst_transform)

    rgba = np.zeros((height, width, 4), dtype=np.float32)
    fire_pixels = reprojected == 1
    rgba[fire_pixels] = [1.0, 0.2, 0.0, 0.65]  # red/orange, semi-transparent

    if output_path is None:
        band_label = band_arg.strip().lower().replace(" ", "_")
        output_path = default_output_path(fire_id, band_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        rgba,
        extent=(left, right, bottom, top),
        origin="upper",
        zorder=2,
    )
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=DST_CRS, zorder=1)

    ax.set_title(f"Fire {fire_id} — {band_desc}", fontsize=12, fontweight="bold")
    ax.set_axis_off()

    info = (
        f"bands: {band_count}\n"
        f"selected: {band_idx}\n"
        f"burned px: {burned:,}"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        zorder=3,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print("Fire mask basemap visualization")
    print(f"{'=' * 60}")
    print(f"Fire ID            : {fire_id}")
    print(f"Raster CRS         : {src_crs}")
    print(f"Band count         : {band_count}")
    print(f"Selected band      : {band_idx} ({band_desc})")
    print(f"Burned pixel count : {burned:,}")
    print(f"Output PNG         : {output_path.resolve()}")
    print(f"{'=' * 60}\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one fire-mask band on an OpenStreetMap basemap."
    )
    parser.add_argument("--fire-id", required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--band",
        default="last",
        help="Band to show: first, middle, last, or integer (default: last).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--figsize",
        default="8,8",
        help="Figure size in inches, e.g. 8,8",
    )
    args = parser.parse_args()

    try:
        w, h = (float(x.strip()) for x in args.figsize.split(","))
        figsize = (w, h)
    except ValueError:
        print("[ERROR] --figsize must be two comma-separated numbers, e.g. 8,8")
        sys.exit(1)

    try:
        visualize(
            fire_id=args.fire_id,
            input_dir=args.input_dir,
            band_arg=args.band,
            output_path=args.output,
            dpi=args.dpi,
            figsize=figsize,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
