"""
visualize_multiple_fire_extents_on_basemap.py
=============================================
Draw fire-mask raster extents as black rectangles on an OpenStreetMap
regional overview map (similar to a QGIS extent screenshot).

Usage (from project root):

    python scripts/task1/visualization/visualize_multiple_fire_extents_on_basemap.py --limit 20
    python scripts/task1/visualization/visualize_multiple_fire_extents_on_basemap.py --limit 20 \\
        --output outputs/visualizations/fire_extent_overview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from shapely.geometry import box

DEFAULT_INPUT_DIR = Path("data/aligned_compressed")
FIRE_MASK_NAME = "labels/fire_mask_stacked_30m_aligned.tif"
DEFAULT_OUTPUT = Path("outputs/visualizations/fire_extent_overview.png")
DST_CRS = "EPSG:3857"


def list_fire_ids(input_dir: Path, limit: int) -> list[str]:
    ids = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
    return ids[:limit]


def extent_polygon_for_fire(fire_id: str, input_dir: Path) -> gpd.GeoDataFrame | None:
    mask_path = input_dir / fire_id / FIRE_MASK_NAME
    if not mask_path.is_file():
        print(f"  [SKIP] {fire_id}: missing {FIRE_MASK_NAME}")
        return None

    with rasterio.open(mask_path) as src:
        b = src.bounds
        geom = box(b.left, b.bottom, b.right, b.top)
        gdf = gpd.GeoDataFrame(
            {"fire_id": [fire_id]},
            geometry=[geom],
            crs=src.crs,
        )
    return gdf.to_crs(DST_CRS)


def visualize_overview(
    input_dir: Path,
    limit: int,
    output_path: Path,
    dpi: int,
    figsize: tuple[float, float],
    padding: float = 0.05,
) -> Path:
    fire_ids = list_fire_ids(input_dir, limit)
    if not fire_ids:
        raise FileNotFoundError(f"No fire folders under {input_dir}")

    frames: list[gpd.GeoDataFrame] = []
    for fid in fire_ids:
        gdf = extent_polygon_for_fire(fid, input_dir)
        if gdf is not None:
            frames.append(gdf)

    if not frames:
        raise FileNotFoundError(
            f"No fire-mask files found under {input_dir} (checked {len(fire_ids)} folders)."
        )

    all_extents = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))

    minx, miny, maxx, maxy = all_extents.total_bounds
    pad_x = (maxx - minx) * padding
    pad_y = (maxy - miny) * padding
    xlim = (minx - pad_x, maxx + pad_x)
    ylim = (miny - pad_y, maxy + pad_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    all_extents.plot(
        ax=ax,
        facecolor="black",
        edgecolor="black",
        linewidth=0.5,
        alpha=1.0,
        zorder=2,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        crs=DST_CRS,
        zorder=1,
    )

    ax.set_title(
        f"Fire mask extents (n={len(all_extents)})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_axis_off()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print("Multi-fire extent overview")
    print(f"{'=' * 60}")
    print(f"Fires requested : {limit}")
    print(f"Extents drawn   : {len(all_extents)}")
    print(f"Output PNG      : {output_path.resolve()}")
    print(f"{'=' * 60}\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fire-mask extents as black boxes on OpenStreetMap."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", default="12,10")
    args = parser.parse_args()

    try:
        w, h = (float(x.strip()) for x in args.figsize.split(","))
        figsize = (w, h)
    except ValueError:
        print("[ERROR] --figsize must be two comma-separated numbers, e.g. 12,10")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        sys.exit(1)

    try:
        visualize_overview(
            input_dir=args.input_dir,
            limit=args.limit,
            output_path=args.output,
            dpi=args.dpi,
            figsize=figsize,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
