"""Line Integral Convolution core algorithm.

evaluate_field() evaluates a VectorField callable on a pixel grid and
normalises each vector to unit length.

compute_lic() convolves a noise texture along streamlines of a vector field,
producing the characteristic silky fibre texture of LIC.

trace_streamline_pixels() traces a single streamline from a seed pixel and
returns the list of pixel indices it visits — used by the pure_rgb streamline
paint mode in lic_color.py.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# Field evaluation
# ---------------------------------------------------------------------------

def evaluate_field(field, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a VectorField on an (H, W) pixel grid.

    Pixel-centre coordinates are mapped to [0, 1].
    Returned (vx, vy) arrays have unit-length vectors; zero-magnitude
    pixels (stagnation points) remain as zeros.
    """
    x = (np.arange(W, dtype=np.float64) + 0.5) / W
    y = (np.arange(H, dtype=np.float64) + 0.5) / H
    xx, yy = np.meshgrid(x, y)
    vx, vy = field(xx, yy)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    mag = np.sqrt(vx ** 2 + vy ** 2)
    nz = mag > 0.0
    vx[nz] /= mag[nz]
    vy[nz] /= mag[nz]
    return vx, vy


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def _make_kernel(n: int, shape: str, sigma: float) -> np.ndarray:
    """1-D convolution kernel of length n. Centre element is index n//2."""
    t = np.linspace(-1.0, 1.0, n)
    if shape == "box":
        return np.ones(n, dtype=np.float64)
    if shape == "gaussian":
        return np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    if shape == "raised_cosine":
        return 0.5 * (1.0 + np.cos(np.pi * t))
    raise ValueError(f"Unknown kernel shape: {shape!r}. Choose box, gaussian, or raised_cosine.")


# ---------------------------------------------------------------------------
# Bilinear field sampling
# ---------------------------------------------------------------------------

def _sample_field_at(arr: np.ndarray, cy: np.ndarray, cx: np.ndarray, boundary: str) -> np.ndarray:
    """Sample a (H, W) array at sub-pixel positions given normalised coords."""
    H, W = arr.shape
    row = cy * (H - 1)
    col = cx * (W - 1)
    mode = "nearest" if boundary == "nearest" else "wrap"
    return map_coordinates(arr, [row.ravel(), col.ravel()],
                           order=1, mode=mode, prefilter=False).reshape(cy.shape)


def _clamp(cx: np.ndarray, cy: np.ndarray, boundary: str):
    if boundary == "wrap":
        return cx % 1.0, cy % 1.0
    return np.clip(cx, 0.0, 1.0), np.clip(cy, 0.0, 1.0)


# ---------------------------------------------------------------------------
# LIC
# ---------------------------------------------------------------------------

def compute_lic(
    noise: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    num_steps: int = 30,
    step_size: float | None = None,
    kernel: str = "box",
    kernel_sigma: float = 0.3,
    boundary: str = "nearest",
) -> np.ndarray:
    """Compute Line Integral Convolution of a noise texture along a vector field.

    Args:
        noise:        float32 array, shape (H, W) or (H, W, 3), values in [0, 1].
        vx, vy:       unit-vector field arrays from evaluate_field(), shape (H, W).
        num_steps:    integration steps in each direction from each pixel centre.
        step_size:    Euler step in normalised [0, 1] coords.
                      Default: 1 / max(H, W)  ≈ 1 pixel per step.
        kernel:       'box', 'gaussian', or 'raised_cosine'.
        kernel_sigma: sigma for gaussian kernel (fraction of half-streamline length).
        boundary:     'nearest' (streamlines stop at edges) or 'wrap' (tiling fields).

    Returns:
        float32 array of same shape as noise, values in [0, 1].

    Memory: ~44 bytes per pixel (independent of num_steps).
    Performance: roughly 1–4 s per million pixels at 30 steps on a modern CPU.
    """
    H, W = noise.shape[:2]
    is_rgb = noise.ndim == 3
    n_ch = noise.shape[2] if is_rgb else 1
    work = noise[:, :, np.newaxis] if not is_rgb else noise
    work = work.astype(np.float64)

    if step_size is None:
        step_size = 1.0 / max(H, W)

    total = 2 * num_steps + 1
    weights = _make_kernel(total, kernel, kernel_sigma)
    total_weight = weights.sum()

    # Pixel-centre coordinates in [0, 1]
    x0 = (np.arange(W, dtype=np.float64) + 0.5) / W
    y0 = (np.arange(H, dtype=np.float64) + 0.5) / H
    xx0, yy0 = np.meshgrid(x0, y0)  # (H, W)

    accum = np.zeros((H, W, n_ch), dtype=np.float64)

    def accumulate(cx: np.ndarray, cy: np.ndarray, w: float) -> None:
        row = cy * (H - 1)
        col = cx * (W - 1)
        mode = "nearest" if boundary == "nearest" else "wrap"
        for ch in range(n_ch):
            s = map_coordinates(work[:, :, ch], [row.ravel(), col.ravel()],
                                order=1, mode=mode, prefilter=False).reshape(H, W)
            accum[:, :, ch] += w * s

    # Centre pixel
    accumulate(xx0, yy0, weights[num_steps])

    # Forward pass: step along +field direction
    cx, cy = xx0.copy(), yy0.copy()
    for i in range(1, num_steps + 1):
        dvx = _sample_field_at(vx, cy, cx, boundary)
        dvy = _sample_field_at(vy, cy, cx, boundary)
        cx = cx + step_size * dvx
        cy = cy + step_size * dvy
        cx, cy = _clamp(cx, cy, boundary)
        accumulate(cx, cy, weights[num_steps + i])

    # Backward pass: step along -field direction
    cx, cy = xx0.copy(), yy0.copy()
    for i in range(1, num_steps + 1):
        dvx = _sample_field_at(vx, cy, cx, boundary)
        dvy = _sample_field_at(vy, cy, cx, boundary)
        cx = cx - step_size * dvx
        cy = cy - step_size * dvy
        cx, cy = _clamp(cx, cy, boundary)
        accumulate(cx, cy, weights[num_steps - i])

    result = (accum / total_weight).astype(np.float32)
    return result[:, :, 0] if not is_rgb else result


# ---------------------------------------------------------------------------
# Streamline tracing (used by pure_rgb streamline-paint mode)
# ---------------------------------------------------------------------------

def trace_streamline_pixels(
    r0: int,
    c0: int,
    vx: np.ndarray,
    vy: np.ndarray,
    step_size: float,
    num_steps: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Trace a streamline forward from pixel (r0, c0), returning visited pixel indices.

    Uses field values at the current integer-pixel position (no bilinear interpolation)
    for speed — suitable for the streamline paint mode which traces many short paths.
    Returns a deduplicated list of (row, col) tuples.
    """
    pixels: list[tuple[int, int]] = [(r0, c0)]
    px_step = step_size * max(H, W)  # convert normalised step to pixel units
    xr = float(c0) + 0.5
    yr = float(r0) + 0.5
    for _ in range(num_steps):
        ri = min(int(yr), H - 1)
        ci = min(int(xr), W - 1)
        dvx = float(vx[ri, ci])
        dvy = float(vy[ri, ci])
        xr = max(0.5, min(W - 0.5, xr + px_step * dvx))
        yr = max(0.5, min(H - 0.5, yr + px_step * dvy))
        nr = min(int(yr), H - 1)
        nc = min(int(xr), W - 1)
        if (nr, nc) != pixels[-1]:
            pixels.append((nr, nc))
    return pixels
