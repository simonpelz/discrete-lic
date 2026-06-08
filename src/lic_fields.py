"""Vector field definitions for LIC art generation.

Each factory function returns a VectorField callable:
    field(x, y) -> (vx, vy)
where x, y are (H, W) numpy arrays with values in [0, 1],
and vx, vy are the corresponding unnormalised vector components.
The LIC core normalises vectors to unit length before integration.

The FIELDS registry maps name strings to factory functions for use by the CLI.
For the 'complex' field, use complex_func() directly with a Python callable.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

VectorField = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def uniform(angle_deg: float = 0.0) -> VectorField:
    """Constant-direction field — produces straight parallel lines."""
    rad = math.radians(angle_deg)
    vx0 = math.cos(rad)
    vy0 = math.sin(rad)

    def field(x, y):
        return np.full_like(x, vx0), np.full_like(y, vy0)

    return field


def rotation(cx: float = 0.5, cy: float = 0.5, clockwise: bool = False) -> VectorField:
    """Pure rotation around (cx, cy) — produces concentric circles."""
    sign = -1.0 if clockwise else 1.0

    def field(x, y):
        return -sign * (y - cy), sign * (x - cx)

    return field


def source(cx: float = 0.5, cy: float = 0.5, strength: float = 1.0) -> VectorField:
    """Point source (strength > 0) or sink (strength < 0) — produces radial rays."""
    def field(x, y):
        return strength * (x - cx), strength * (y - cy)

    return field


def sink(cx: float = 0.5, cy: float = 0.5) -> VectorField:
    """Point sink — radial inward flow."""
    return source(cx, cy, strength=-1.0)


def saddle(cx: float = 0.5, cy: float = 0.5) -> VectorField:
    """Saddle point at (cx, cy) — hyperbolic flow, X-shaped streamlines."""
    def field(x, y):
        return x - cx, -(y - cy)

    return field


def shear(direction: str = "horizontal") -> VectorField:
    """Linear shear. horizontal: vx=y, vy=0. vertical: vx=0, vy=x."""
    if direction == "horizontal":
        def field(x, y):
            return y.copy(), np.zeros_like(y)
    else:
        def field(x, y):
            return np.zeros_like(x), x.copy()

    return field


def double_vortex(
    x1: float = 0.3,
    y1: float = 0.5,
    x2: float = 0.7,
    y2: float = 0.5,
    s1: float = 1.0,
    s2: float = -1.0,
) -> VectorField:
    """Superposition of two counter-rotating vortices — figure-8 / alternating spirals."""
    r1 = rotation(x1, y1, clockwise=(s1 < 0))
    r2 = rotation(x2, y2, clockwise=(s2 < 0))

    def field(x, y):
        vx1, vy1 = r1(x, y)
        vx2, vy2 = r2(x, y)
        return vx1 + vx2, vy1 + vy2

    return field


def wave(kx: float = 3.0, ky: float = 2.0, phase: float = 0.0) -> VectorField:
    """Sinusoidal wave field — woven / braided streamline pattern."""
    def field(x, y):
        vx = np.sin(2.0 * np.pi * ky * y + phase)
        vy = np.sin(2.0 * np.pi * kx * x)
        return vx, vy

    return field


def perlin_curl(scale: float = 4.0, octaves: int = 4, seed: int = 0) -> VectorField:
    """Curl of a layered sine-wave potential — turbulent, divergence-free, no extra deps.

    Potential: P(x,y) = Σ_i A_i · sin(kx_i·x + ky_i·y + φ_i)
    Field (curl of P): vx = ∂P/∂y, vy = −∂P/∂x
    """
    rng = np.random.default_rng(seed)
    A = np.array([1.0 / (i + 1) for i in range(octaves)])
    angles = rng.uniform(0, 2 * np.pi, octaves)
    freq = scale * (2.0 ** np.arange(octaves, dtype=float))
    kx_arr = freq * np.cos(angles)
    ky_arr = freq * np.sin(angles)
    phi = rng.uniform(0, 2 * np.pi, octaves)

    def field(x, y):
        vx = np.zeros_like(x, dtype=float)
        vy = np.zeros_like(y, dtype=float)
        for a, kxi, kyi, p in zip(A, kx_arr, ky_arr, phi):
            cos_arg = np.cos(kxi * x + kyi * y + p)
            vx += a * kyi * cos_arg   # dP/dy
            vy -= a * kxi * cos_arg   # -dP/dx
        return vx, vy

    return field


def spiral(
    cx: float = 0.5,
    cy: float = 0.5,
    inward: float = 0.5,
    rot: float = 1.0,
) -> VectorField:
    """Logarithmic spiral: blend of sink and rotation.

    inward > 0 spirals toward (cx, cy), rot controls rotational speed.
    """
    def field(x, y):
        dx = x - cx
        dy = y - cy
        return -inward * dx - rot * dy, -inward * dy + rot * dx

    return field


def wd_merger(
    dataset_path: str | None = None,
    resolution: int = 1024,
    slice_axis: str = "theta",
    bx_field=("gas", "magnetic_field_r"),
    by_field=("gas", "magnetic_field_z"),
    flip_y: bool = True,
) -> VectorField:
    """Load the in-plane magnetic field of a WD merger snapshot via yt.

    Mirrors the data behind:
        yt.SlicePlot(ds, 'theta', 'magnetic_field_strength', origin='native')
        s.annotate_line_integral_convolution('magnetic_field_x', 'magnetic_field_y', ...)
    by slicing the dataset along ``slice_axis`` and sampling
    (magnetic_field_x, magnetic_field_y) on a fixed-resolution buffer. The
    LIC core normalises these to unit length before integration, so the raw
    physical magnitudes are unimportant.

    Parameters
    ----------
    dataset_path : path to any yt-readable dataset; if None, the
        ``WDMerger_hdf5_chk_1000`` sample is fetched via ``yt.load_sample``.
    resolution : FRB resolution in pixels along each axis.
    slice_axis : axis perpendicular to the slice plane (matches the
        user's original snippet: ``'theta'``).
    flip_y : yt's FRB has y increasing upward, while this codebase has y
        increasing downward (row 0 = top). Default True flips back to match.

    Requires ``yt``. If it isn't installed, the factory raises ImportError
    with an install hint — system-wide `pip install yt`, or in a venv:
        python3 -m venv .venv && source .venv/bin/activate
        pip install yt numpy scipy pillow reportlab matplotlib
    """
    try:
        import yt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'wd_merger' field requires the 'yt' package.\n"
            "Install it system-wide:\n"
            "    pip install yt\n"
            "or in a virtualenv (preferred for isolation):\n"
            "    python3 -m venv .venv && source .venv/bin/activate\n"
            "    pip install yt numpy scipy pillow reportlab matplotlib"
        ) from exc

    if dataset_path is None:
        ds = yt.load_sample("WDMerger_hdf5_chk_1000")
    else:
        ds = yt.load(dataset_path)

    axis_idx = ds.coordinates.axis_id[slice_axis]
    coord = float(ds.domain_center[axis_idx])
    sl = ds.slice(slice_axis, coord)

    dw = ds.domain_width
    perp = [i for i in range(3) if i != axis_idx]
    width = (dw[perp[0]], dw[perp[1]])
    frb = sl.to_frb(width=width, resolution=resolution)

    bx_arr = np.asarray(frb[bx_field], dtype=np.float64)
    by_arr = np.asarray(frb[by_field], dtype=np.float64)
    bx_arr = np.nan_to_num(bx_arr, nan=0.0, posinf=0.0, neginf=0.0)
    by_arr = np.nan_to_num(by_arr, nan=0.0, posinf=0.0, neginf=0.0)
    H_arr, W_arr = bx_arr.shape

    def field(x, y):
        ix = np.clip((x * W_arr).astype(np.int64), 0, W_arr - 1)
        if flip_y:
            iy = np.clip(((1.0 - y) * H_arr).astype(np.int64), 0, H_arr - 1)
        else:
            iy = np.clip((y * H_arr).astype(np.int64), 0, H_arr - 1)
        return bx_arr[iy, ix], by_arr[iy, ix]

    return field


def mhd_cluster(
    sample: str = "MHDSloshing",
    vector: str = "vorticity",
    slice_axis: str = "y",
    resolution: int = 1024,
    width_kpc: float | None = 500.0,
    flip_y: bool = True,
) -> VectorField:
    """In-plane slice of a vector field from a yt MHD cluster sample.

    Defaults to ZuHone's ``MHDSloshing`` magnetised-sloshing-cluster Athena
    run, sliced along ``y`` (so the in-plane components are
    ``vorticity_x`` and ``vorticity_z``). Switch ``vector`` to
    ``"magnetic_field"`` for clean B-field structure or ``"velocity"`` for
    the bulk flow.

    Parameters
    ----------
    sample : yt sample dataset name (anything ``yt.load_sample`` accepts).
    vector : ``"vorticity"``, ``"magnetic_field"``, or ``"velocity"``.
        The two in-plane components ``(<vector>_a, <vector>_b)`` are
        sampled on the FRB.
    slice_axis : ``"x"``, ``"y"``, or ``"z"`` — axis perpendicular to the
        slice plane.
    resolution : FRB resolution in pixels along each axis.
    width_kpc : full slice width in kpc; ``None`` uses the dataset's full
        in-plane domain width.
    flip_y : yt's FRB has y increasing upward, this codebase has y
        increasing downward (row 0 = top). Default True flips back.
    """
    try:
        import yt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'mhd_cluster' field requires the 'yt' package.\n"
            "Install it system-wide:\n"
            "    pip install yt\n"
            "or in a virtualenv (preferred for isolation):\n"
            "    python3 -m venv .venv && source .venv/bin/activate\n"
            "    pip install yt numpy scipy pillow reportlab matplotlib"
        ) from exc

    in_plane = {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }
    if slice_axis not in in_plane:
        raise ValueError(f"slice_axis must be one of x/y/z, got {slice_axis!r}")
    a, b = in_plane[slice_axis]
    bx_field = ("gas", f"{vector}_{a}")
    by_field = ("gas", f"{vector}_{b}")

    ds = yt.load_sample(sample)
    axis_idx = ds.coordinates.axis_id[slice_axis]
    coord = float(ds.domain_center[axis_idx])
    sl = ds.slice(slice_axis, coord)

    if width_kpc is None:
        dw = ds.domain_width
        perp = [i for i in range(3) if i != axis_idx]
        width = (dw[perp[0]], dw[perp[1]])
    else:
        width = ((width_kpc, "kpc"), (width_kpc, "kpc"))
    frb = sl.to_frb(width=width, resolution=resolution)

    bx_arr = np.asarray(frb[bx_field], dtype=np.float64)
    by_arr = np.asarray(frb[by_field], dtype=np.float64)
    bx_arr = np.nan_to_num(bx_arr, nan=0.0, posinf=0.0, neginf=0.0)
    by_arr = np.nan_to_num(by_arr, nan=0.0, posinf=0.0, neginf=0.0)
    H_arr, W_arr = bx_arr.shape

    def field(x, y):
        ix = np.clip((x * W_arr).astype(np.int64), 0, W_arr - 1)
        if flip_y:
            iy = np.clip(((1.0 - y) * H_arr).astype(np.int64), 0, H_arr - 1)
        else:
            iy = np.clip((y * H_arr).astype(np.int64), 0, H_arr - 1)
        return bx_arr[iy, ix], by_arr[iy, ix]

    return field


def complex_func(f: Callable[[complex], complex]) -> VectorField:
    """Wrap a complex analytic function f(z) → w as a vector field.

    vx = Re(w), vy = Im(w), where z = x + 1j·y.

    Examples:
        complex_func(lambda z: z**2)
        complex_func(lambda z: 1 / z)
        complex_func(lambda z: np.sin(z))
        complex_func(lambda z: (z - 0.5 - 0.5j)**3)
    """
    def field(x, y):
        z = x.astype(complex) + 1j * y.astype(complex)
        w = f(z)
        w = np.asarray(w)
        # Guard against infinities at poles
        vx = np.where(np.isfinite(w.real), w.real, 0.0).astype(float)
        vy = np.where(np.isfinite(w.imag), w.imag, 0.0).astype(float)
        return vx, vy

    return field


FIELDS: dict[str, Callable] = {
    "uniform": uniform,
    "rotation": rotation,
    "source": source,
    "sink": sink,
    "saddle": saddle,
    "shear": shear,
    "double_vortex": double_vortex,
    "wave": wave,
    "perlin_curl": perlin_curl,
    "spiral": spiral,
    "wd_merger": wd_merger,
    "mhd_cluster": mhd_cluster,
    # "complex" handled specially in lic_main.py
}
