"""Voronoi-pixel LIC: irregular Voronoi cells as the noise substrate.

The image plane is partitioned into N Voronoi cells via a scatter of random
seed points (nearest-seed tessellation). Each cell holds one noise value
(scalar or RGB). LIC traces a streamline from every cell's seed point and
accumulates the noise values of cells visited along the way — sampling is
piecewise-constant (no interpolation). The final image paints each pixel with
its cell's LIC colour, revealing the organic irregular cell boundaries inside
the flow texture.

Coordinate convention (same as lic_core.py)
--------------------------------------------
  cx  = column fraction in [0, 1]  ↔  the x / horizontal direction
  cy  = row fraction    in [0, 1]  ↔  the y / vertical direction
  seeds[:, 0] = cx,  seeds[:, 1] = cy

Public API
----------
build_seeds(N, rng)                   → (N, 2) float64
build_cell_map(H, W, seeds)           → (H, W) int32
cell_noise_white(N, rng)              → (N,)   float32
cell_noise_white_rgb(N, rng)          → (N, 3) float32
cell_noise_hsv(N, rng, ...)           → (N, 3) float32
cell_noise_pure_rgb(N, rng)           → (N, 3) float32
compute_voronoi_lic(...)              → (N,) or (N, 3) float32
render_cells(cell_values, cell_map)        → (H, W) or (H, W, 3) float32
render_cells_polygon(cell_colors, seeds, H, W, ssaa) → (H, W, 3) uint8
enhance_cells(arr, method, gamma)          → same shape, float32
colorize_cells_angle_hsv(...)             → (N, 3) uint8
pure_rgb_cells_stochastic(...)            → (N, 3) uint8
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from lic_core import _make_kernel


# ---------------------------------------------------------------------------
# Tessellation
# ---------------------------------------------------------------------------

def build_seeds(N: int, rng: np.random.Generator) -> np.ndarray:
    """Scatter N seed points uniformly in [0, 1]².

    Returns (N, 2) float64: each row is (cx, cy) = (col-frac, row-frac).
    """
    return rng.random((N, 2))


def build_cell_map(H: int, W: int, seeds: np.ndarray) -> np.ndarray:
    """Rasterise nearest-seed Voronoi tessellation onto the pixel grid.

    For each pixel centre, finds the nearest seed and records its index.

    seeds:   (N, 2) float64, (col-frac, row-frac)
    Returns: (H, W) int32 — Voronoi cell index for every pixel
    """
    cx = (np.arange(W, dtype=np.float64) + 0.5) / W   # (W,) col fractions
    cy = (np.arange(H, dtype=np.float64) + 0.5) / H   # (H,) row fractions
    XX, YY = np.meshgrid(cx, cy)                       # (H, W) each
    pixel_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (H·W, 2)
    _, idx = cKDTree(seeds).query(pixel_xy)
    return idx.reshape(H, W).astype(np.int32)


# ---------------------------------------------------------------------------
# Per-cell noise generators
# ---------------------------------------------------------------------------

def _hsv_to_rgb_1d(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV → RGB for 1-D arrays of length N. Returns (N, 3) float32."""
    h, s, v = (np.asarray(a, np.float32) for a in (h, s, v))
    h6 = h * 6.0
    i = h6.astype(np.int32) % 6
    f = (h6 - np.floor(h6)).astype(np.float32)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p,
        np.where(i == 3, p, np.where(i == 4, t, v)))))
    g = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v,
        np.where(i == 3, q, np.where(i == 4, p, p)))))
    b = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t,
        np.where(i == 3, v, np.where(i == 4, v, q)))))
    return np.stack([r, g, b], axis=1).astype(np.float32)


def cell_noise_white(N: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform scalar noise per cell. Returns (N,) float32."""
    return rng.random(N).astype(np.float32)


def cell_noise_white_rgb(N: int, rng: np.random.Generator) -> np.ndarray:
    """Independent white noise per RGB channel per cell. Returns (N, 3) float32."""
    return rng.random((N, 3)).astype(np.float32)


def cell_noise_hsv(
    N: int,
    rng: np.random.Generator,
    sat_range: tuple[float, float] = (0.85, 1.0),
    val_range: tuple[float, float] = (0.85, 1.0),
) -> np.ndarray:
    """Vivid random HSV colour per cell. Returns (N, 3) float32 RGB."""
    h = rng.random(N).astype(np.float32)
    s = rng.uniform(*sat_range, N).astype(np.float32)
    v = rng.uniform(*val_range, N).astype(np.float32)
    return _hsv_to_rgb_1d(h, s, v)


def cell_noise_pure_rgb(N: int, rng: np.random.Generator) -> np.ndarray:
    """Each cell is exactly one pure primary (R, G, or B). Returns (N, 3) float32."""
    idx = rng.integers(0, 3, size=N)
    out = np.zeros((N, 3), dtype=np.float32)
    out[idx == 0, 0] = 1.0
    out[idx == 1, 1] = 1.0
    out[idx == 2, 2] = 1.0
    return out


# ---------------------------------------------------------------------------
# Voronoi LIC core
# ---------------------------------------------------------------------------

def compute_voronoi_lic(
    seeds: np.ndarray,
    cell_noise: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    cell_map: np.ndarray,
    num_steps: int = 30,
    step_size: float | None = None,
    kernel: str = "box",
    kernel_sigma: float = 0.3,
    boundary: str = "nearest",
) -> np.ndarray:
    """Compute LIC with Voronoi cells as the noise atoms.

    All N streamlines are traced simultaneously via vectorised numpy. At each
    Euler step the current position is converted to a pixel index and looked up
    in cell_map to get the cell's noise value — piecewise-constant, no
    interpolation. This is what gives the output its mosaic character: every
    pixel in a cell shares exactly the same LIC value.

    The default step_size is 1/√N (≈ one average inter-seed distance), so each
    step crosses roughly one cell. With num_steps=20 the streamline visits ~20
    cells per direction, producing clear flow structure in the mosaic.

    Args:
        seeds:       (N, 2) float64 — (col-frac, row-frac) seed positions in [0,1]²
        cell_noise:  (N,) or (N, 3) float32 — per-cell noise values
        vx, vy:      (H, W) float64 — unit vector field from evaluate_field()
        cell_map:    (H, W) int32   — Voronoi cell index for every pixel
        num_steps:   integration steps in each direction from each seed
        step_size:   Euler step in [0,1] normalised coords
                     Default: 1/√N ≈ one average cell width
        kernel:      'box', 'gaussian', or 'raised_cosine'
        kernel_sigma: sigma for gaussian kernel
        boundary:    'nearest' (stop at image edges) or 'wrap' (tiling)

    Returns:
        (N,) or (N, 3) float32 — per-cell LIC result in [0, 1]
    """
    H, W = vx.shape
    N = len(seeds)
    is_rgb = cell_noise.ndim == 2
    n_ch = cell_noise.shape[1] if is_rgb else 1

    if step_size is None:
        step_size = 1.0 / np.sqrt(N)

    n_total = 2 * num_steps + 1
    weights = _make_kernel(n_total, kernel, kernel_sigma)
    total_weight = float(weights.sum())

    noise_work = cell_noise if is_rgb else cell_noise[:, np.newaxis]  # (N, n_ch)
    accum = np.zeros((N, n_ch), dtype=np.float64)

    def _sample(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Read cell noise at N normalised (col-frac, row-frac) positions → (N, n_ch)."""
        row = np.clip((py * H).astype(np.int32), 0, H - 1)
        col = np.clip((px * W).astype(np.int32), 0, W - 1)
        return noise_work[cell_map[row, col]]

    def _field(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row = np.clip((py * H).astype(np.int32), 0, H - 1)
        col = np.clip((px * W).astype(np.int32), 0, W - 1)
        return vx[row, col], vy[row, col]

    def _bound(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if boundary == "wrap":
            return px % 1.0, py % 1.0
        return np.clip(px, 0.0, 1.0), np.clip(py, 0.0, 1.0)

    sx, sy = seeds[:, 0].copy(), seeds[:, 1].copy()

    # Centre sample
    accum += weights[num_steps] * _sample(sx, sy)

    # Forward pass (+field direction)
    cx, cy = sx.copy(), sy.copy()
    for i in range(1, num_steps + 1):
        dvx, dvy = _field(cx, cy)
        cx, cy = _bound(cx + step_size * dvx, cy + step_size * dvy)
        accum += weights[num_steps + i] * _sample(cx, cy)

    # Backward pass (−field direction)
    cx, cy = sx.copy(), sy.copy()
    for i in range(1, num_steps + 1):
        dvx, dvy = _field(cx, cy)
        cx, cy = _bound(cx - step_size * dvx, cy - step_size * dvy)
        accum += weights[num_steps - i] * _sample(cx, cy)

    result = (accum / total_weight).astype(np.float32)
    return result[:, 0] if not is_rgb else result


# ---------------------------------------------------------------------------
# Rendering and colour helpers
# ---------------------------------------------------------------------------

def render_cells(cell_values: np.ndarray, cell_map: np.ndarray) -> np.ndarray:
    """Paint each pixel with its cell's value via fancy indexing.

    cell_values: (N,) or (N, C) float32
    cell_map:    (H, W) int32
    Returns      (H, W) or (H, W, C) float32 — every pixel shares its cell's value.
    """
    return cell_values[cell_map]


def render_cells_polygon(
    cell_colors: np.ndarray,
    seeds: np.ndarray,
    H: int,
    W: int,
    ssaa: int = 2,
) -> np.ndarray:
    """Render Voronoi cells as filled polygons for crisp, straight boundaries.

    The KDTree cell_map used during LIC integration has pixelated staircase
    boundaries — acceptable for integration accuracy, but ugly to look at.
    This function computes the exact Voronoi polygon for each cell via
    scipy.spatial.Voronoi + Sutherland-Hodgman clipping, then fills them with
    PIL at ssaa× the output resolution and downsamples with LANCZOS filtering.
    The result has anti-aliased straight edges independent of output resolution.

    cell_colors: (N, 3) uint8 — one colour per cell
    seeds:       (N, 2) float64 — (col-frac, row-frac), same ordering as cell_colors
    ssaa:        supersampling factor for anti-aliased edges (default 2).
                 Rendering N polygons is fast, so ssaa=2 adds negligible cost.

    Returns (H, W, 3) uint8.
    """
    from PIL import Image, ImageDraw
    from scipy.spatial import Voronoi

    from lic_geometry import clip_polygon_to_rect, voronoi_finite_polygons_2d

    WS, HS = W * ssaa, H * ssaa

    # Seeds in supersampled pixel coordinates: (x, y) = (col, row) for PIL
    seeds_px = seeds * np.array([WS, HS], dtype=np.float64)

    vor = Voronoi(seeds_px)
    # regions[i] → vertex index list for seed i (same ordering as seeds_px)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=max(WS, HS) * 2)

    img = Image.new("RGB", (WS, HS))
    draw = ImageDraw.Draw(img)

    for cell_idx, region in enumerate(regions):
        poly = vertices[region]                         # (n_verts, 2) float64
        clipped = clip_polygon_to_rect(poly, 0, 0, WS, HS)
        if len(clipped) < 3:
            continue
        color = tuple(int(c) for c in cell_colors[cell_idx])
        # clip_polygon_to_rect returns a mix of tuples and numpy arrays;
        # PIL's draw.polygon requires a uniform flat-coordinate sequence.
        pts = [(float(p[0]), float(p[1])) for p in clipped]
        draw.polygon(pts, fill=color)

    if ssaa > 1:
        img = img.resize((W, H), Image.LANCZOS)

    return np.array(img)


def enhance_cells(arr: np.ndarray, method: str = "stretch", gamma: float = 1.0) -> np.ndarray:
    """Contrast enhancement for (N,) or (N, C) float32 per-cell arrays.

    Mirrors lic_color.enhance_contrast but works on 1-D / 2-D per-cell data.
    'equalize' falls back to 'stretch' (histogram equalization is less
    meaningful for N ≪ H·W samples).
    """
    arr = np.asarray(arr, dtype=np.float32)
    if method == "none":
        return arr
    if method == "gamma":
        return np.clip(arr ** gamma, 0.0, 1.0).astype(np.float32)
    if method in ("stretch", "equalize"):
        if arr.ndim == 1:
            lo, hi = float(arr.min()), float(arr.max())
            if hi > lo:
                return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            return arr.copy()
        out = np.empty_like(arr)
        for ch in range(arr.shape[1]):
            col = arr[:, ch]
            lo, hi = float(col.min()), float(col.max())
            out[:, ch] = np.clip((col - lo) / (hi - lo), 0.0, 1.0) if hi > lo else col
        return out.astype(np.float32)
    raise ValueError(f"Unknown enhance method: {method!r}")


def colorize_cells_angle_hsv(
    lic_gray: np.ndarray,
    seeds: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    sat: float = 0.9,
) -> np.ndarray:
    """Colour each cell by its seed's field angle; brightness from LIC value.

    lic_gray: (N,) float32 — enhanced LIC value per cell
    seeds:    (N, 2) float64 — (col-frac, row-frac)
    Returns   (N, 3) uint8
    """
    H, W = vx.shape
    row = np.clip((seeds[:, 1] * H).astype(np.int32), 0, H - 1)
    col = np.clip((seeds[:, 0] * W).astype(np.int32), 0, W - 1)
    svx = vx[row, col]
    svy = vy[row, col]
    angle = np.arctan2(svy, svx)
    hue = ((angle + np.pi) / (2.0 * np.pi)).astype(np.float32)
    s_arr = np.full_like(hue, sat)
    v_arr = np.clip(lic_gray, 0.0, 1.0).astype(np.float32)
    rgb = _hsv_to_rgb_1d(hue, s_arr, v_arr)
    return (rgb * 255).astype(np.uint8)


def pure_rgb_cells_stochastic(
    lic_rgb: np.ndarray,
    rng: np.random.Generator,
    sharpness: float = 2.5,
) -> np.ndarray:
    """Per-cell pure-RGB stochastic sampling (same logic as lic_color version).

    lic_rgb:   (N, 3) float32 — per-cell LIC values (channels ≈ probability dist.)
    sharpness: exponent applied before sampling; higher → dominant colour wins more.
    Returns    (N, 3) uint8 where every cell is (255,0,0), (0,255,0), or (0,0,255).
    """
    N = len(lic_rgb)
    total = lic_rgb.sum(axis=1, keepdims=True)
    probs = np.where(total > 1e-8, lic_rgb / total,
                     np.full_like(lic_rgb, 1.0 / 3.0))
    if sharpness != 1.0:
        probs = np.power(probs, sharpness)
    probs = probs.astype(np.float64)
    probs /= probs.sum(axis=1, keepdims=True)
    cumprobs = probs.cumsum(axis=1)
    u = rng.random(N)
    choices = np.clip((u[:, np.newaxis] >= cumprobs).sum(axis=1), 0, 2)
    pure = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    return pure[choices]
