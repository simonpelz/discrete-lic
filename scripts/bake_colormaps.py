#!/usr/bin/env python3
"""Bake matplotlib colormaps into 256x1 PNG lookup tables for the WebGL color stage.

Each PNG is a 256-wide, 1-tall RGB image where column i holds cmap(i/255).
The browser color pass (app/gl/color.js, `colormap` mode) samples these as a
1D lookup texture, exactly mirroring `apply_colormap` in src/lic_color.py
(which does `cmap(clip(lic_gray, 0, 1))[..., :3]`).

Run:
    /home/simon/Code/Kunst/random\\ colors/.venv/bin/python scripts/bake_colormaps.py
or with system python (matplotlib must be importable).

Outputs into app/assets/colormaps/<name>.png .
"""
from __future__ import annotations

import os
from pathlib import Path

# Use a writable cache dir before importing matplotlib (sandbox-safe).
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl-cache"))

import numpy as np
from matplotlib import colormaps
from PIL import Image

# Colormaps exposed in the UI / referenced by the Python pipeline.
COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "turbo", "coolwarm", "RdYlBu", "Spectral",
]

N = 256


def bake_one(name: str, out_dir: Path) -> Path:
    cmap = colormaps[name]
    xs = np.linspace(0.0, 1.0, N, dtype=np.float64)
    rgba = cmap(xs)                       # (N, 4) float64 in [0,1]
    rgb = (rgba[:, :3] * 255.0).round().clip(0, 255).astype(np.uint8)
    img = rgb.reshape(1, N, 3)            # (H=1, W=256, 3)
    out = out_dir / f"{name}.png"
    Image.fromarray(img, mode="RGB").save(out)
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "app" / "assets" / "colormaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in COLORMAPS:
        p = bake_one(name, out_dir)
        written.append(p)
        print(f"baked {name:9s} -> {p}")
    print(f"\n{len(written)} colormap LUTs written to {out_dir}")


if __name__ == "__main__":
    main()
