#!/usr/bin/env python3
"""Bake yt-backed vector fields into a browser-loadable asset for field.js.

Writes:
    app/assets/fields/<name>.bin   — raw RG, top-left authored, UNIT field
    app/assets/fields/<name>.json  — sidecar (resolution, encoding, slice meta)

Encoding mirrors CONTRACT.md / field.js: the field is normalised to UNIT length
(stagnation -> 0). For the default "unorm8" encoding we store the *encoded* value
enc = v*0.5+0.5 quantised to [0,255] (so field.js reads byte/255 directly). For
"float16"/"float32" we store the raw unit components in [-1,1] and field.js
re-encodes. Rows are written TOP first (row 0 = image top), matching the Python
evaluate_field() convention (x=(col+.5)/W, y=(row+.5)/H, row 0 at top) and the
CLAUDE.md y-slice layout for mhd_cluster.

BAKING CAVEAT
-------------
The yt MHDSloshing sample (~1.5 GB) downloads on first use. In a sandboxed
environment yt's cache may be unwritable or the sample host unreachable. If the
real bake fails, this script ALSO writes a SYNTHETIC placeholder (a
double_vortex / perlin_curl field rendered to the same .bin format) tagged
"synthetic": true in the sidecar, so field.js + the pipeline work end-to-end
NOW. The REAL bake must then be re-run where yt's MHDSloshing sample is
available (delete the synthetic asset first, or pass --force).

Usage:
    .venv/bin/python scripts/export_field_assets.py                # try mhd, fallback synthetic
    .venv/bin/python scripts/export_field_assets.py --synthetic    # force synthetic only
    .venv/bin/python scripts/export_field_assets.py --resolution 1024 --encoding float16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # discrete-lic/
PROJECT = os.path.dirname(REPO)                   # random colors/  (has src/, .venv/)
SRC = os.path.join(PROJECT, "src")
ASSET_DIR = os.path.join(REPO, "app", "assets", "fields")

sys.path.insert(0, SRC)


def sample_field(field, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a VectorField on a (resolution, resolution) grid, UNIT-normalise.

    Matches lic_core.evaluate_field: x=(col+.5)/W, y=(row+.5)/H, row 0 at top.
    Returns (vx, vy) float32 unit arrays, shape (resolution, resolution),
    row 0 = top.
    """
    n = resolution
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    y = (np.arange(n, dtype=np.float64) + 0.5) / n
    xx, yy = np.meshgrid(x, y)
    vx, vy = field(xx, yy)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    mag = np.sqrt(vx * vx + vy * vy)
    nz = mag > 0.0
    vx[nz] /= mag[nz]
    vy[nz] /= mag[nz]
    vx[~nz] = 0.0
    vy[~nz] = 0.0
    return vx.astype(np.float32), vy.astype(np.float32)


def write_asset(name: str, vx: np.ndarray, vy: np.ndarray, encoding: str,
                meta_extra: dict) -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)
    h, w = vx.shape
    # interleave RG, row 0 (top) first
    if encoding == "unorm8":
        enc_x = np.clip(np.round((vx * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
        enc_y = np.clip(np.round((vy * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
        inter = np.empty((h, w, 2), dtype=np.uint8)
        inter[..., 0] = enc_x
        inter[..., 1] = enc_y
        raw = inter.tobytes()
    elif encoding == "float16":
        inter = np.empty((h, w, 2), dtype=np.float16)
        inter[..., 0] = vx.astype(np.float16)
        inter[..., 1] = vy.astype(np.float16)
        raw = inter.tobytes()
    elif encoding == "float32":
        inter = np.empty((h, w, 2), dtype=np.float32)
        inter[..., 0] = vx
        inter[..., 1] = vy
        raw = inter.tobytes()
    else:
        raise ValueError(f"unknown encoding {encoding!r}")

    bin_path = os.path.join(ASSET_DIR, f"{name}.bin")
    json_path = os.path.join(ASSET_DIR, f"{name}.json")
    with open(bin_path, "wb") as f:
        f.write(raw)

    sidecar = {
        "name": name,
        "bin": f"{name}.bin",
        "resolution": int(w),
        "width": int(w),
        "height": int(h),
        "channels": 2,
        "encoding": encoding,
        "encoded_as": "enc = v*0.5+0.5" if encoding == "unorm8" else "raw unit [-1,1]",
        "value_range": [-1.0, 1.0],
        "unit_normalised": True,
        "origin": "top-left (row 0 = image top)",
        "sample_in_shader_at": "vec2(uv.x, 1.0 - uv.y)",
    }
    sidecar.update(meta_extra)
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  wrote {bin_path} ({len(raw)} bytes) + {os.path.basename(json_path)}")


def bake_mhd(resolution: int, encoding: str) -> bool:
    """Attempt the real mhd_cluster bake. Returns True on success."""
    from lic_fields import mhd_cluster  # noqa: import after sys.path insert

    print("Attempting real mhd_cluster bake (yt MHDSloshing, y-slice, vorticity)...")
    # Mirrors CLAUDE.md final-work convention: y-slice, vorticity, default sample.
    field = mhd_cluster(
        sample="MHDSloshing",
        vector="vorticity",
        slice_axis="y",
        resolution=resolution,
        width_kpc=500.0,
        flip_y=True,
    )
    vx, vy = sample_field(field, resolution)
    write_asset("mhd_cluster", vx, vy, encoding, {
        "synthetic": False,
        "source": "yt MHDSloshing sample",
        "vector": "vorticity",
        "slice_axis": "y",
        "in_plane_components": ["vorticity_x", "vorticity_z"],
        "width_kpc": 500.0,
        "flip_y": True,
        "note": "Real yt bake. In-plane vorticity_x/_z on a y-slice through the "
                "domain centre, matching the CLAUDE.md final-work setup.",
    })
    return True


def bake_synthetic(name: str, resolution: int, encoding: str, reason: str) -> None:
    """Render a synthetic stand-in (double_vortex + perlin_curl) to the asset."""
    from lic_fields import double_vortex, perlin_curl

    print(f"Writing SYNTHETIC placeholder for '{name}' ({reason})...")
    dv = double_vortex()
    pc = perlin_curl(scale=4.0, octaves=5, seed=17)

    n = resolution
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    y = (np.arange(n, dtype=np.float64) + 0.5) / n
    xx, yy = np.meshgrid(x, y)
    ax, ay = dv(xx, yy)
    bx, by = pc(xx, yy)
    # blend: large-scale vortices + turbulent curl, to mimic cluster vorticity
    vx = np.asarray(ax) + 0.6 * np.asarray(bx)
    vy = np.asarray(ay) + 0.6 * np.asarray(by)
    mag = np.sqrt(vx * vx + vy * vy)
    nz = mag > 0.0
    vx = np.where(nz, vx / np.where(nz, mag, 1.0), 0.0)
    vy = np.where(nz, vy / np.where(nz, mag, 1.0), 0.0)
    write_asset(name, vx.astype(np.float32), vy.astype(np.float32), encoding, {
        "synthetic": True,
        "source": "synthetic double_vortex + perlin_curl blend",
        "reason": reason,
        "note": "PLACEHOLDER. The real yt MHDSloshing bake must be run where "
                "the ~1.5 GB sample is available (delete this asset, re-run "
                "export_field_assets.py without --synthetic).",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", type=int, default=1024,
                    help="FRB / grid resolution (square). Default 1024.")
    ap.add_argument("--encoding", choices=["unorm8", "float16", "float32"],
                    default="unorm8", help="Asset byte encoding. Default unorm8.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Skip yt; write only the synthetic placeholder.")
    ap.add_argument("--name", default="mhd_cluster",
                    help="Asset base name. Default mhd_cluster.")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="Soft time budget (s) for the yt fetch before giving up.")
    args = ap.parse_args()

    if args.synthetic:
        bake_synthetic(args.name, args.resolution, args.encoding, "forced --synthetic")
        return

    t0 = time.time()
    try:
        ok = bake_mhd(args.resolution, args.encoding)
        if ok:
            dt = time.time() - t0
            print(f"Real bake succeeded in {dt:.1f}s.")
            return
    except Exception as exc:  # noqa: BLE001 — broad on purpose, fall back cleanly
        reason = f"yt bake failed: {type(exc).__name__}: {exc}"
        print(reason, file=sys.stderr)
        bake_synthetic(args.name, args.resolution, args.encoding, reason)


if __name__ == "__main__":
    main()
