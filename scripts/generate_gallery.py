#!/usr/bin/env python3
"""Generate gallery images for the discrete-lic web app.

Run from the project root:
    python scripts/generate_gallery.py

Writes PNGs to web/gallery/ and updates web/gallery/gallery.json.
Skips images that already exist unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lic_main import _build_parser, generate  # noqa: E402
from lic_render import save_image              # noqa: E402

GALLERY_DIR = ROOT / "web" / "gallery"

# ---------------------------------------------------------------------------
# Gallery definition
# Each entry: (filename, caption, extra_args dict)
# Base args: 800×800, seed=42, pixel_mode=voronoi, color=pure_rgb, steps=30
# ---------------------------------------------------------------------------

GALLERY = [
    (
        "pure_rgb_perlin_voronoi.png",
        "Pure RGB · Perlin curl · Voronoi",
        {"field": "perlin_curl", "color": "pure_rgb", "pixel_mode": "voronoi", "voronoi_cells": 1200},
    ),
    (
        "pure_rgb_rotation_voronoi.png",
        "Pure RGB · Rotation · Voronoi",
        {"field": "rotation", "color": "pure_rgb", "pixel_mode": "voronoi", "voronoi_cells": 1000},
    ),
    (
        "pure_rgb_spiral_voronoi.png",
        "Pure RGB · Spiral · Voronoi",
        {"field": "spiral", "color": "pure_rgb", "pixel_mode": "voronoi", "voronoi_cells": 800},
    ),
    (
        "pure_rgb_double_vortex_voronoi.png",
        "Pure RGB · Double vortex · Voronoi",
        {"field": "double_vortex", "color": "pure_rgb", "pixel_mode": "voronoi", "voronoi_cells": 900},
    ),
    (
        "pure_rgb_wave_voronoi.png",
        "Pure RGB · Wave · Voronoi",
        {"field": "wave", "color": "pure_rgb", "pixel_mode": "voronoi", "voronoi_cells": 600},
    ),
    (
        "pure_rgb_saddle_voronoi.png",
        "Pure RGB · Saddle · Voronoi",
        {"field": "saddle", "color": "pure_rgb", "pixel_mode": "voronoi", "voronoi_cells": 700},
    ),
    (
        "pure_rgb_complex_voronoi.png",
        "Pure RGB · z² · Voronoi",
        {"field": "complex", "field_expr": "z**2", "color": "pure_rgb", "pixel_mode": "voronoi", "voronoi_cells": 900},
    ),
    (
        "pure_rgb_perlin_grid.png",
        "Pure RGB · Perlin curl · Grid",
        {"field": "perlin_curl", "color": "pure_rgb", "pixel_mode": "grid", "steps": 20},
    ),
    (
        "angle_hsv_perlin_voronoi.png",
        "Angle HSV · Perlin curl · Voronoi",
        {"field": "perlin_curl", "color": "angle_hsv", "pixel_mode": "voronoi", "voronoi_cells": 1000},
    ),
    (
        "angle_hsv_double_vortex.png",
        "Angle HSV · Double vortex · Grid",
        {"field": "double_vortex", "color": "angle_hsv", "pixel_mode": "grid", "steps": 50, "kernel": "gaussian"},
    ),
    (
        "angle_hsv_complex.png",
        "Angle HSV · sin(4z) · Grid",
        {"field": "complex", "field_expr": "np.sin(z*4)", "color": "angle_hsv", "pixel_mode": "grid"},
    ),
    (
        "colormap_wave_voronoi.png",
        "Plasma colormap · Wave · Voronoi",
        {"field": "wave", "color": "colormap", "colormap": "plasma", "pixel_mode": "voronoi", "voronoi_cells": 800},
    ),
    (
        "colormap_spiral.png",
        "Inferno colormap · Spiral · Grid",
        {"field": "spiral", "color": "colormap", "colormap": "inferno", "pixel_mode": "grid", "steps": 40},
    ),
    (
        "rgb_perlin_voronoi.png",
        "RGB noise · Perlin curl · Voronoi",
        {"field": "perlin_curl", "color": "rgb", "noise": "hsv", "pixel_mode": "voronoi", "voronoi_cells": 1200},
    ),
]

# Defaults shared by all gallery images
DEFAULTS = dict(
    width=800,
    height=800,
    seed=42,
    steps=30,
    kernel="box",
    kernel_sigma=0.3,
    boundary="nearest",
    enhance="stretch",
    gamma=1.0,
    noise="white_rgb",
    noise_points=200,
    noise_block=8,
    dpi=96,
    pure_mode="stochastic",
    pure_sharpen=2.5,
    streamline_stride=4,
    colormap="viridis",
    sat=0.9,
    voronoi_cells=800,
    pixel_mode="voronoi",
    field="rotation",
    field_expr="z**2",
    color="pure_rgb",
    step_size=None,
    # yt fields (unused for offline gallery)
    mhd_sample="MHDSloshing",
    mhd_vector="vorticity",
    mhd_slice_axis="y",
    mhd_resolution=1024,
    mhd_width_kpc=500.0,
    wd_dataset=None,
    wd_resolution=1024,
    wd_slice_axis="theta",
)


def make_args(**overrides) -> SimpleNamespace:
    d = dict(DEFAULTS)
    d.update(overrides)
    return SimpleNamespace(**d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Regenerate images that already exist.")
    cli = ap.parse_args()

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    meta = []
    total = len(GALLERY)

    for i, (filename, caption, overrides) in enumerate(GALLERY, 1):
        path = GALLERY_DIR / filename
        print(f"[{i}/{total}] {filename}")

        if path.exists() and not cli.force:
            print(f"       skipped (exists). Use --force to regenerate.")
            meta.append({"filename": filename, "caption": caption})
            continue

        args = make_args(**overrides)
        t0 = time.perf_counter()
        try:
            img = generate(args)
        except Exception as e:
            print(f"       ERROR: {e}", file=sys.stderr)
            continue

        save_image(img, path, dpi=args.dpi)
        print(f"       done in {time.perf_counter() - t0:.1f} s")
        meta.append({"filename": filename, "caption": caption})

    with open(GALLERY_DIR / "gallery.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {len(meta)} entries to web/gallery/gallery.json")


if __name__ == "__main__":
    main()
