#!/usr/bin/env python3
"""LIC (Line Integral Convolution) generative art generator.

Applies LIC to a noise texture along a configurable vector field, producing
images with a silky, flow-following texture.

Output format is inferred from the --output extension:
  .png    lossless PNG (default)
  .tif    TIFF with LZW compression
  .jpg    JPEG
  .pdf    single-page PDF, image sized to fit at --dpi

Quick examples — grid mode (default)
-------------------------------------
  python lic_main.py
  python lic_main.py --field rotation --color rgb
  python lic_main.py --field perlin_curl --noise white --color colormap --colormap plasma
  python lic_main.py --field double_vortex --color angle_hsv --steps 50 --kernel gaussian
  python lic_main.py --field wave --noise voronoi --color rgb --width 1200 --height 1600
  python lic_main.py --field complex --field-expr "z**2" --color rgb
  python lic_main.py --field complex --field-expr "np.sin(z * 4)" --noise white_rgb --color rgb
  python lic_main.py --color pure_rgb --pure-mode stochastic --field spiral
  python lic_main.py --color pure_rgb --pure-mode streamline --streamline-stride 6
  python lic_main.py --width 4000 --height 6000 --field perlin_curl --output big.tif

Quick examples — Voronoi-pixel mode
-------------------------------------
  python lic_main.py --pixel-mode voronoi --voronoi-cells 800 --field perlin_curl --color rgb
  python lic_main.py --pixel-mode voronoi --voronoi-cells 300 --field rotation --color angle_hsv
  python lic_main.py --pixel-mode voronoi --voronoi-cells 1500 --field double_vortex --noise hsv --color rgb
  python lic_main.py --pixel-mode voronoi --voronoi-cells 600 --field spiral --color pure_rgb
  python lic_main.py --pixel-mode voronoi --voronoi-cells 200 --field wave --color colormap --colormap plasma --steps 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import lic_color as lcolor
from lic_core import compute_lic, evaluate_field
from lic_fields import FIELDS, complex_func
from lic_noise import (
    block_noise, hsv_noise, pure_rgb_noise,
    voronoi_noise, white_noise, white_noise_rgb,
)
from lic_render import save_image
from lic_voronoi import (
    build_seeds, build_cell_map,
    cell_noise_white, cell_noise_white_rgb, cell_noise_hsv, cell_noise_pure_rgb,
    compute_voronoi_lic, render_cells_polygon,
    enhance_cells, colorize_cells_angle_hsv, pure_rgb_cells_stochastic,
)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_field(args):
    if args.field == "complex":
        expr = args.field_expr
        try:
            f = eval(f"lambda z: {expr}", {"np": np, "__builtins__": {}})
            f(complex(0.5, 0.5))
        except Exception as e:
            raise ValueError(f"Invalid --field-expr {expr!r}: {e}")
        return complex_func(f)
    if args.field not in FIELDS:
        raise ValueError(f"Unknown field {args.field!r}. Available: {', '.join(FIELDS)}, complex")
    if args.field == "wd_merger":
        return FIELDS["wd_merger"](
            dataset_path=args.wd_dataset,
            resolution=args.wd_resolution,
            slice_axis=args.wd_slice_axis,
        )
    if args.field == "mhd_cluster":
        width_kpc = args.mhd_width_kpc if args.mhd_width_kpc and args.mhd_width_kpc > 0 else None
        return FIELDS["mhd_cluster"](
            sample=args.mhd_sample,
            vector=args.mhd_vector,
            slice_axis=args.mhd_slice_axis,
            resolution=args.mhd_resolution,
            width_kpc=width_kpc,
        )
    return FIELDS[args.field]()


def _build_noise(args, H: int, W: int, rng: np.random.Generator):
    """Return noise array (H, W) or (H, W, 3)."""
    n = args.noise
    if n == "white":
        return white_noise(H, W, rng)
    if n == "white_rgb":
        return white_noise_rgb(H, W, rng)
    if n == "hsv":
        return hsv_noise(H, W, rng)
    if n == "voronoi":
        return voronoi_noise(H, W, rng, num_points=args.noise_points)
    if n == "block":
        return block_noise(H, W, rng, block_size=args.noise_block)
    if n == "pure_rgb":
        return pure_rgb_noise(H, W, rng)
    raise ValueError(f"Unknown noise type {n!r}")


# ---------------------------------------------------------------------------
# Color application
# ---------------------------------------------------------------------------

def _apply_color(args, lic_result, noise_arr, vx, vy, rng):
    strategy = args.color

    if strategy == "rgb":
        lic_e = lcolor.enhance_contrast(lic_result, args.enhance, args.gamma)
        return lcolor.rgb_lic_to_image(lic_e)

    if strategy == "colormap":
        gray = lic_result.mean(axis=2) if lic_result.ndim == 3 else lic_result
        lic_e = lcolor.enhance_contrast(gray, args.enhance, args.gamma)
        return lcolor.apply_colormap(lic_e, args.colormap)

    if strategy == "angle_hsv":
        if lic_result.ndim == 3:
            gray = lic_result.mean(axis=2)
        else:
            gray = lic_result
        gray = lcolor.enhance_contrast(gray, args.enhance, args.gamma)
        return lcolor.colorize_by_angle(gray, vx, vy, sat=args.sat)

    if strategy == "hsv_noise":
        if noise_arr.ndim != 3:
            print("Warning: hsv_noise color mode expects 3-channel noise. "
                  "Using first channel as value.", file=sys.stderr)
            v_ch = noise_arr
            hue_arr = np.zeros_like(noise_arr)
        else:
            r, g, b = noise_arr[:, :, 0], noise_arr[:, :, 1], noise_arr[:, :, 2]
            cmax = np.maximum(np.maximum(r, g), b)
            cmin = np.minimum(np.minimum(r, g), b)
            delta = cmax - cmin
            hue_arr = np.zeros_like(r)
            m = (cmax == g) & (delta > 0)
            hue_arr[m] = (((b[m] - r[m]) / delta[m]) % 6.0 + 2.0) / 6.0
            m = (cmax == b) & (delta > 0)
            hue_arr[m] = (((r[m] - g[m]) / delta[m]) + 4.0) / 6.0
            m = (cmax == r) & (delta > 0)
            hue_arr[m] = (((g[m] - b[m]) / delta[m]) % 6.0) / 6.0
            v_ch = cmax

        H, W = v_ch.shape
        lic_v = compute_lic(
            v_ch, vx, vy,
            num_steps=args.steps,
            step_size=args.step_size,
            kernel=args.kernel,
            kernel_sigma=args.kernel_sigma,
            boundary=args.boundary,
        )
        lic_v = lcolor.enhance_contrast(lic_v, args.enhance, args.gamma)
        return lcolor.hsv_lic_to_image(lic_v, hue_arr, sat=args.sat)

    if strategy == "pure_rgb":
        if args.pure_mode == "stochastic":
            return lcolor.pure_rgb_stochastic(lic_result, rng, sharpness=args.pure_sharpen)
        else:
            return lcolor.pure_rgb_streamline(
                lic_result, vx, vy, rng,
                num_steps=args.steps,
                step_size=args.step_size,
                stride=args.streamline_stride,
            )

    raise ValueError(f"Unknown color strategy {strategy!r}")


# ---------------------------------------------------------------------------
# Voronoi-pixel pipeline helpers
# ---------------------------------------------------------------------------

def _build_cell_noise(args, N: int, rng: np.random.Generator) -> np.ndarray:
    if args.color == "pure_rgb":
        return cell_noise_pure_rgb(N, rng)
    n = args.noise
    if n in ("voronoi", "block"):
        raise ValueError(
            f"--noise {n!r} cannot be used with --pixel-mode voronoi. "
            "Use white, white_rgb, hsv, or pure_rgb."
        )
    if n == "white":
        return cell_noise_white(N, rng)
    if n in ("white_rgb", "hsv_noise"):
        return cell_noise_white_rgb(N, rng)
    if n == "hsv":
        return cell_noise_hsv(N, rng)
    if n == "pure_rgb":
        return cell_noise_pure_rgb(N, rng)
    raise ValueError(f"Unknown noise type {n!r}")


def _apply_color_voronoi(args, cell_lic, seeds, vx, vy, rng) -> np.ndarray:
    """Map per-cell LIC values to per-cell RGB colours. Returns (N, 3) uint8."""
    strategy = args.color

    if strategy == "rgb":
        cell_e = enhance_cells(cell_lic, args.enhance, args.gamma)
        return (np.clip(cell_e, 0.0, 1.0) * 255).astype(np.uint8)

    if strategy == "colormap":
        gray = cell_lic.mean(axis=1) if cell_lic.ndim == 2 else cell_lic
        gray = enhance_cells(gray, args.enhance, args.gamma)
        try:
            from matplotlib import colormaps
            cmap = colormaps[args.colormap]
        except ImportError:
            raise ImportError("matplotlib is required for --color colormap.")
        rgba = cmap(gray.astype(np.float64))
        return (rgba[:, :3] * 255).astype(np.uint8)

    if strategy == "angle_hsv":
        gray = cell_lic.mean(axis=1) if cell_lic.ndim == 2 else cell_lic
        gray = enhance_cells(gray, args.enhance, args.gamma)
        return colorize_cells_angle_hsv(gray, seeds, vx, vy, sat=args.sat)

    if strategy == "pure_rgb":
        if args.pure_mode == "streamline":
            print("Note: --pure-mode streamline is not supported in voronoi pixel mode; "
                  "using stochastic.", file=sys.stderr)
        return pure_rgb_cells_stochastic(cell_lic, rng, sharpness=args.pure_sharpen)

    if strategy == "hsv_noise":
        raise ValueError("--color hsv_noise is not supported with --pixel-mode voronoi. "
                         "Use --color rgb with --noise hsv instead.")

    raise ValueError(f"Unknown color strategy {strategy!r}")


def _run_voronoi_pipeline(args, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Full Voronoi-pixel LIC pipeline. Returns (H, W, 3) uint8."""
    N = args.voronoi_cells
    step_info = f"step={args.step_size:.4f}" if args.step_size else f"step=1/√{N}≈{1/N**0.5:.4f}"
    print(f"Voronoi-pixel LIC  {W}×{H} px  |  {N} cells  field={args.field}  color={args.color}")
    print(f"     steps={args.steps}  kernel={args.kernel}  {step_info}  boundary={args.boundary}")

    print("  [1/5] evaluating vector field…")
    field = _build_field(args)
    vx, vy = evaluate_field(field, H, W)

    print(f"  [2/5] tessellating {N} Voronoi cells…")
    seeds = build_seeds(N, rng)
    cell_map = build_cell_map(H, W, seeds)

    print("  [3/5] generating per-cell noise…")
    cell_noise_arr = _build_cell_noise(args, N, rng)

    print(f"  [4/5] running Voronoi LIC ({args.steps} steps each direction)…")
    cell_lic = compute_voronoi_lic(
        seeds, cell_noise_arr, vx, vy, cell_map,
        num_steps=args.steps,
        step_size=args.step_size,
        kernel=args.kernel,
        kernel_sigma=args.kernel_sigma,
        boundary=args.boundary,
    )

    print("  [5/5] applying colour strategy and rendering polygons…")
    cell_colors = _apply_color_voronoi(args, cell_lic, seeds, vx, vy, rng)
    return render_cells_polygon(cell_colors, seeds, H, W)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(args) -> Image.Image:
    """Run the full LIC pipeline and return a PIL Image.

    args can be an argparse.Namespace or any object with the same attributes.
    This is the programmatic entry point used by the web server.
    """
    W = args.width
    H = args.height if args.height is not None else args.width
    rng = np.random.default_rng(args.seed)

    if args.pixel_mode == "voronoi":
        rgb_arr = _run_voronoi_pipeline(args, H, W, rng)
    else:
        print(f"LIC  {W}×{H} px  |  field={args.field}  noise={args.noise}  color={args.color}")
        print(f"     steps={args.steps}  kernel={args.kernel}  boundary={args.boundary}")

        print("  [1/4] evaluating vector field…")
        field = _build_field(args)
        vx, vy = evaluate_field(field, H, W)

        print("  [2/4] generating noise texture…")
        if args.color == "pure_rgb" and args.noise != "pure_rgb":
            noise_arr = pure_rgb_noise(H, W, rng)
        elif args.color == "hsv_noise" and args.noise not in ("hsv", "white_rgb"):
            print("        (hsv_noise color mode: using hsv noise regardless of --noise)")
            noise_arr = hsv_noise(H, W, rng)
        else:
            noise_arr = _build_noise(args, H, W, rng)

        print(f"  [3/4] running LIC ({args.steps} steps each direction)…")
        lic_result = compute_lic(
            noise_arr, vx, vy,
            num_steps=args.steps,
            step_size=args.step_size,
            kernel=args.kernel,
            kernel_sigma=args.kernel_sigma,
            boundary=args.boundary,
        )

        print("  [4/4] applying color strategy…")
        rgb_arr = _apply_color(args, lic_result, noise_arr, vx, vy, rng)

    return lcolor.to_pil_image(rgb_arr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--output", type=Path, default=Path("lic_out.png"),
                    help="Output path. Format from extension: .png .tif .jpg .pdf")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=None,
                    help="Height in pixels (default: same as --width)")
    ap.add_argument("--dpi", type=int, default=150,
                    help="DPI for metadata and PDF sizing")

    ap.add_argument("--field", default="rotation",
                    choices=list(FIELDS.keys()) + ["complex"],
                    help="Vector field. Use 'complex' with --field-expr.")
    ap.add_argument("--field-expr", default="z**2",
                    help="Python/numpy expression for --field complex, e.g. 'z**2', 'np.sin(z*4)'.")
    ap.add_argument("--wd-dataset", type=str, default=None,
                    help="HDF5 path for --field wd_merger (omit to fetch sample via yt).")
    ap.add_argument("--wd-resolution", type=int, default=1024)
    ap.add_argument("--wd-slice-axis", type=str, default="theta")
    ap.add_argument("--mhd-sample", type=str, default="MHDSloshing",
                    help="yt sample for --field mhd_cluster.")
    ap.add_argument("--mhd-vector", type=str, default="vorticity",
                    choices=["vorticity", "magnetic_field", "velocity"])
    ap.add_argument("--mhd-slice-axis", type=str, default="y", choices=["x", "y", "z"])
    ap.add_argument("--mhd-resolution", type=int, default=1024)
    ap.add_argument("--mhd-width-kpc", type=float, default=500.0,
                    help="Slice width in kpc (<=0 = full domain).")

    ap.add_argument("--noise", default="white_rgb",
                    choices=["white", "white_rgb", "hsv", "voronoi", "block", "pure_rgb"],
                    help="Input noise texture.")
    ap.add_argument("--noise-points", type=int, default=200)
    ap.add_argument("--noise-block", type=int, default=8)

    ap.add_argument("--steps", type=int, default=30,
                    help="Integration steps per direction.")
    ap.add_argument("--step-size", type=float, default=None,
                    help="Euler step in [0,1] coords (default: 1/max(W,H)).")
    ap.add_argument("--kernel", default="box",
                    choices=["box", "gaussian", "raised_cosine"])
    ap.add_argument("--kernel-sigma", type=float, default=0.3)
    ap.add_argument("--boundary", default="nearest", choices=["nearest", "wrap"])

    ap.add_argument("--color", default="rgb",
                    choices=["rgb", "colormap", "angle_hsv", "hsv_noise", "pure_rgb"],
                    help="Color mapping strategy.")
    ap.add_argument("--colormap", default="viridis",
                    help="Matplotlib colormap (--color colormap).")
    ap.add_argument("--sat", type=float, default=0.9,
                    help="Saturation for angle_hsv / hsv_noise.")
    ap.add_argument("--pure-mode", default="stochastic",
                    choices=["stochastic", "streamline"],
                    help="Pure RGB sub-mode.")
    ap.add_argument("--pure-sharpen", type=float, default=2.5,
                    help="Sharpness exponent for pure_rgb stochastic.")
    ap.add_argument("--streamline-stride", type=int, default=4)

    ap.add_argument("--enhance", default="stretch",
                    choices=["stretch", "equalize", "gamma", "none"])
    ap.add_argument("--gamma", type=float, default=1.0)

    ap.add_argument("--pixel-mode", default="grid", choices=["grid", "voronoi"],
                    dest="pixel_mode")
    ap.add_argument("--voronoi-cells", type=int, default=800, dest="voronoi_cells")

    ap.add_argument("--seed", type=int, default=17)

    return ap


def main() -> None:
    args = _build_parser().parse_args()
    t0 = time.perf_counter()
    try:
        pil_img = generate(args)
    except (ValueError, ImportError) as e:
        sys.exit(str(e))
    save_image(pil_img, args.output, dpi=args.dpi)
    print(f"Done in {time.perf_counter() - t0:.1f} s  →  {args.output}")


if __name__ == "__main__":
    main()
