"""
qgis_style_fire_mask.py
=======================
Print QGIS symbology instructions for stacked binary fire-mask GeoTIFFs.

Binary 0/1 rasters often appear as solid black rectangles in QGIS when
loaded with default single-band gray styling. This helper documents the
correct visualization steps.

Usage (from project root):

    python scripts/qgis_style_fire_mask.py
    python scripts/qgis_style_fire_mask.py --fire-id 20778153
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_INPUT_DIR = Path("data/aligned_compressed")
FIRE_MASK_NAME = "labels/fire_mask_stacked_30m_aligned.tif"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print QGIS styling instructions for fire-mask stacks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Aligned dataset root (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--fire-id",
        default="20778153",
        help="Example fire ID for path in instructions.",
    )
    args = parser.parse_args()

    example_path = args.input_dir / args.fire_id / FIRE_MASK_NAME

    print(
        f"""
================================================================================
QGIS styling guide — fire_mask_stacked_30m_aligned.tif
================================================================================

Why maps may look like black rectangles
---------------------------------------
The fire mask is a **binary cumulative mask** (0 = not burned yet,
1 = burned by that date). QGIS often loads single-band rasters with
**Singleband gray** symbology. Values 0 and 1 both map to very dark
grays, so the layer can look like a solid black rectangle even when
the data is correct.

Example file:
  {example_path}

Step-by-step in QGIS
--------------------
1. Load the layer (or use your import script with LOAD_MASK = True).

2. Right-click ``fire_mask_stacked_30m_aligned.tif`` in the Layers panel.

3. Choose **Properties** → **Symbology** (QGIS 3.x).

4. Change renderer to **Paletted / Unique values** (or **Paletted/Unique**).

5. Click **Classify** (or add entries manually):
   - Value **0** → color **transparent** (alpha 0) or very light gray
   - Value **1** → color **red** or **orange** (e.g. #FF4500)

6. Under **Band**, try **later bands**, not only Band 1:
   - Band 1  = cumulative burned area through day 1 (often all zeros)
   - Middle band = partial fire progression
   - Last band  = should match final burned footprint most closely

7. Optional: open **Layer styling** panel and use the band slider /
   temporal controller if you stack multiple bands for animation.

8. Compare visually with:
   - ``labels/burned_area_30m_aligned.tif`` (final footprint)
   - ``labels/severity_30m_aligned.tif`` (damage classes)

Validate data before blaming symbology
--------------------------------------
Black rectangles do **not** automatically mean corrupt data. Confirm with:

    python scripts/validate_aligned_fire_data.py --fire-id {args.fire_id}

or for a quick batch check:

    python scripts/validate_aligned_fire_data.py --limit 5

If ``overall_valid`` is True in the CSV report, the GeoTIFF metadata and
pixel values are consistent; adjust QGIS styling as above.

Import script note
------------------
If your QGIS import script sets only:

    LOAD_MASK = True
    LOAD_BURNED_AREA = False
    LOAD_SEVERITY = False
    ...

you will only see the fire mask. Enable other layers to cross-check
against burned area and severity.

================================================================================
"""
    )


if __name__ == "__main__":
    main()
