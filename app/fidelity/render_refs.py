#!/usr/bin/env python3
"""Render Python-CLI reference images for browser fidelity comparison.

The in-browser WebGL pipeline is a reimplementation; this produces the "ground
truth" the browser output is visually compared against (side-by-side + SSIM).
Each case writes <name>.png plus <name>.json with the exact params, so the
browser harness can render the same parameter set and diff.

Usage:
    python3 app/fidelity/render_refs.py            # pure-math cases (system python)
    VENV=../.venv/bin/python python3 app/fidelity/render_refs.py --with-yt

Outputs to app/fidelity/refs/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]          # discrete-lic/
SRC_MAIN = REPO / "src" / "lic_main.py"
REFS = HERE / "refs"

# Fixed small size keeps CLI renders fast; the browser renders the same size.
W = H = 512
SEED = 17

# (name, requires_yt, params-as-CLI-dict). Keys are CLI flags without leading --.
CASES = [
    ("grid_rotation_purergb", False, {
        "field": "rotation", "pixel-mode": "grid", "noise": "pure_rgb",
        "color": "pure_rgb", "pure-mode": "stochastic", "pure-sharpen": "2.5",
        "steps": "30", "kernel": "box", "enhance": "stretch",
    }),
    ("grid_perlin_anglehsv", False, {
        "field": "perlin_curl", "pixel-mode": "grid", "noise": "white",
        "color": "angle_hsv", "sat": "0.9", "steps": "40", "kernel": "gaussian",
        "enhance": "stretch",
    }),
    ("grid_spiral_colormap", False, {
        "field": "spiral", "pixel-mode": "grid", "noise": "white",
        "color": "colormap", "colormap": "plasma", "steps": "30",
        "enhance": "stretch",
    }),
    ("grid_complex_z2_rgb", False, {
        "field": "complex", "field-expr": "z**2", "pixel-mode": "grid",
        "noise": "white_rgb", "color": "rgb", "steps": "30", "enhance": "stretch",
    }),
    ("voronoi_doublevortex_purergb", False, {
        "field": "double_vortex", "pixel-mode": "voronoi", "voronoi-cells": "1200",
        "color": "pure_rgb", "pure-mode": "stochastic", "pure-sharpen": "4.1",
        "steps": "37", "enhance": "stretch",
    }),
    # The artwork's mode/field. Needs yt + the MHDSloshing sample.
    ("voronoi_mhd_purergb", True, {
        "field": "mhd_cluster", "mhd-vector": "vorticity", "mhd-slice-axis": "y",
        "mhd-resolution": "512", "pixel-mode": "voronoi", "voronoi-cells": "1500",
        "color": "pure_rgb", "pure-mode": "stochastic", "pure-sharpen": "4.1",
        "steps": "37", "enhance": "stretch",
    }),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-yt", action="store_true",
                    help="also render yt-backed cases (needs VENV with yt + dataset)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for pure-math cases (default: current)")
    ap.add_argument("--venv-python", default=str(REPO.parent / ".venv" / "bin" / "python"),
                    help="interpreter for yt cases")
    args = ap.parse_args()

    REFS.mkdir(parents=True, exist_ok=True)
    rendered = []
    for name, needs_yt, params in CASES:
        if needs_yt and not args.with_yt:
            print(f"skip  {name} (yt; pass --with-yt)")
            continue
        interp = args.venv_python if needs_yt else args.python
        out_png = REFS / f"{name}.png"
        cmd = [interp, str(SRC_MAIN),
               "--width", str(W), "--height", str(H), "--seed", str(SEED),
               "--output", str(out_png)]
        for k, v in params.items():
            cmd += [f"--{k}", v]
        print(f"render {name} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED:\n{r.stderr[-800:]}")
            continue
        # Browser-side params manifest (CONTRACT names; underscores, native types).
        manifest = {k.replace("-", "_"): _coerce(v) for k, v in params.items()}
        manifest.update({"width": W, "height": H, "seed": SEED})
        (REFS / f"{name}.json").write_text(json.dumps(manifest, indent=2))
        rendered.append(name)
        print(f"  ok -> {out_png.name}")

    print(f"\n{len(rendered)} reference(s) in {REFS}")
    return 0


def _coerce(v: str):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


if __name__ == "__main__":
    raise SystemExit(main())
