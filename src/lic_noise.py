"""Noise texture generators for LIC input.

All public functions return float32 ndarray:
  - shape (H, W)   for single-channel (greyscale) noise
  - shape (H, W, 3) for RGB noise
Values are in [0, 1].
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Shared HSV → RGB (vectorised, avoids per-pixel colorsys calls)
# ---------------------------------------------------------------------------

def _hsv_to_rgb_vec(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV → RGB. h, s, v broadcastable arrays in [0, 1].
    Returns float32 array of same shape + trailing channel dim 3."""
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    h6 = h * 6.0
    i = h6.astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p,
        np.where(i == 3, p, np.where(i == 4, t, v)))))
    g = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v,
        np.where(i == 3, q, np.where(i == 4, p, p)))))
    b = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t,
        np.where(i == 3, v, np.where(i == 4, v, q)))))
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def white_noise(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform white noise in [0, 1], shape (H, W). Classic LIC input."""
    return rng.random((H, W)).astype(np.float32)


def white_noise_rgb(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Independent white noise per RGB channel, shape (H, W, 3)."""
    return rng.random((H, W, 3)).astype(np.float32)


def hsv_noise(
    H: int,
    W: int,
    rng: np.random.Generator,
    sat_range: tuple[float, float] = (0.85, 1.0),
    val_range: tuple[float, float] = (0.85, 1.0),
) -> np.ndarray:
    """Per-pixel random HSV colour with vivid saturation/value. Returns RGB (H, W, 3).

    Matches the existing project's random_hsv_color philosophy but at pixel scale.
    """
    h = rng.random((H, W)).astype(np.float32)
    s = rng.uniform(sat_range[0], sat_range[1], (H, W)).astype(np.float32)
    v = rng.uniform(val_range[0], val_range[1], (H, W)).astype(np.float32)
    return _hsv_to_rgb_vec(h, s, v)


def voronoi_noise(
    H: int,
    W: int,
    rng: np.random.Generator,
    num_points: int = 200,
    sat_range: tuple[float, float] = (0.85, 1.0),
    val_range: tuple[float, float] = (0.85, 1.0),
) -> np.ndarray:
    """Coarse Voronoi cell noise — each cell gets a vivid random HSV colour.

    Uses KDTree nearest-neighbour for fast pixel rasterisation; no polygon
    clipping required.
    """
    pts = np.stack([
        rng.random(num_points) * W,
        rng.random(num_points) * H,
    ], axis=1)
    h = rng.random(num_points).astype(np.float32)
    s = rng.uniform(sat_range[0], sat_range[1], num_points).astype(np.float32)
    v = rng.uniform(val_range[0], val_range[1], num_points).astype(np.float32)
    colors = _hsv_to_rgb_vec(h, s, v)  # (num_points, 3)

    cols, rows = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    grid_pts = np.stack([cols.ravel(), rows.ravel()], axis=1)
    _, indices = cKDTree(pts).query(grid_pts)
    return colors[indices].reshape(H, W, 3)


def block_noise(H: int, W: int, rng: np.random.Generator, block_size: int = 8) -> np.ndarray:
    """Grid of uniform random blocks, shape (H, W). Middle ground between pixel and Voronoi."""
    bh = math.ceil(H / block_size)
    bw = math.ceil(W / block_size)
    blocks = rng.random((bh, bw)).astype(np.float32)
    result = np.repeat(np.repeat(blocks, block_size, axis=0), block_size, axis=1)
    return result[:H, :W]


def pure_rgb_noise(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Each pixel is exactly one pure primary — R, G, or B — shape (H, W, 3).

    Used as input for --color pure_rgb. After LIC each output pixel's (r, g, b)
    values form a probability distribution over the three pure primaries.
    """
    idx = rng.integers(0, 3, size=(H, W))
    result = np.zeros((H, W, 3), dtype=np.float32)
    result[idx == 0, 0] = 1.0
    result[idx == 1, 1] = 1.0
    result[idx == 2, 2] = 1.0
    return result
