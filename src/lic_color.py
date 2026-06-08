"""Color strategies: map LIC output to displayable RGB images.

Strategies
----------
rgb_lic_to_image     — direct float32 RGB → uint8 (for white_rgb or hsv noise input)
apply_colormap       — greyscale LIC + matplotlib colormap
colorize_by_angle    — hue from field direction, luminance from LIC value
hsv_lic_to_image     — LIC ran on V channel only, hue preserved from hsv_noise
pure_rgb_stochastic  — per-pixel categorical sampling from LIC probability distribution
pure_rgb_streamline  — streamline-painted single-colour fibers

Helpers
-------
enhance_contrast     — stretch / equalize / gamma post-processing
to_pil_image         — ndarray → PIL Image
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Vectorised HSV → RGB (avoids per-pixel colorsys calls)
# ---------------------------------------------------------------------------

def _hsv_to_rgb_vec(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
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
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Contrast enhancement
# ---------------------------------------------------------------------------

def _equalize_channel(ch: np.ndarray) -> np.ndarray:
    ch_u8 = (np.clip(ch, 0.0, 1.0) * 255).astype(np.uint8)
    hist, _ = np.histogram(ch_u8.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = int(cdf[cdf > 0].min()) if (cdf > 0).any() else 0
    n = ch_u8.size
    denom = max(n - cdf_min, 1)
    lut = np.round((cdf - cdf_min) / denom * 255).clip(0, 255).astype(np.uint8)
    return (lut[ch_u8] / 255.0).astype(np.float32)


def enhance_contrast(arr: np.ndarray, method: str = "stretch", gamma: float = 1.0) -> np.ndarray:
    """Enhance contrast of a float32 array.

    method:
      'stretch'  — linear rescale each channel to [0, 1]  (recommended default)
      'equalize' — histogram equalisation per channel
      'gamma'    — apply gamma correction (gamma < 1 brightens, > 1 darkens)
      'none'     — pass through unchanged
    """
    arr = np.asarray(arr, dtype=np.float32)
    if method == "none":
        return arr
    if method == "gamma":
        return np.clip(arr ** gamma, 0.0, 1.0).astype(np.float32)
    if method == "stretch":
        if arr.ndim == 2:
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            return arr.copy()
        out = np.empty_like(arr)
        for ch in range(arr.shape[2]):
            sl = arr[:, :, ch]
            lo, hi = sl.min(), sl.max()
            out[:, :, ch] = np.clip((sl - lo) / (hi - lo), 0.0, 1.0) if hi > lo else sl
        return out.astype(np.float32)
    if method == "equalize":
        if arr.ndim == 2:
            return _equalize_channel(arr)
        out = np.empty_like(arr)
        for ch in range(arr.shape[2]):
            out[:, :, ch] = _equalize_channel(arr[:, :, ch])
        return out
    raise ValueError(f"Unknown contrast method: {method!r}. Choose stretch, equalize, gamma, or none.")


# ---------------------------------------------------------------------------
# Standard color strategies
# ---------------------------------------------------------------------------

def rgb_lic_to_image(lic_rgb: np.ndarray) -> np.ndarray:
    """Float32 (H, W, 3) → uint8 (H, W, 3). Simplest path for colour LIC."""
    return (np.clip(lic_rgb, 0.0, 1.0) * 255).astype(np.uint8)


def apply_colormap(lic_gray: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Greyscale LIC float32 (H, W) → RGB uint8 (H, W, 3) via matplotlib colormap."""
    try:
        from matplotlib import colormaps
        cmap = colormaps[colormap]
    except ImportError:
        raise ImportError(
            "matplotlib is required for --color colormap. "
            "Install it or choose another color strategy."
        )
    rgba = cmap(np.clip(lic_gray, 0.0, 1.0))  # (H, W, 4) float64
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def colorize_by_angle(
    lic_gray: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    sat: float = 0.9,
    val_scale: float = 1.0,
) -> np.ndarray:
    """Hue from vector-field angle, value from LIC output.

    Reveals the field structure in colour: direction → hue, LIC texture → brightness.
    """
    angle = np.arctan2(vy, vx)
    hue = ((angle + np.pi) / (2.0 * np.pi)).astype(np.float32)
    value = np.clip(lic_gray * val_scale, 0.0, 1.0).astype(np.float32)
    sat_arr = np.full_like(hue, sat)
    rgb = _hsv_to_rgb_vec(hue, sat_arr, value)
    return (rgb * 255).astype(np.uint8)


def hsv_lic_to_image(
    lic_value: np.ndarray,
    hue_arr: np.ndarray,
    sat: float = 0.9,
) -> np.ndarray:
    """LIC was run on the V channel of HSV noise. Reconstruct RGB with original hues.

    lic_value: (H, W) float32 — LIC-processed value channel
    hue_arr:   (H, W) float32 — original per-pixel hue from hsv_noise
    """
    v = np.clip(lic_value, 0.0, 1.0).astype(np.float32)
    h = np.asarray(hue_arr, dtype=np.float32)
    s = np.full_like(h, sat)
    rgb = _hsv_to_rgb_vec(h, s, v)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pure RGB pixel strategies
# ---------------------------------------------------------------------------

def pure_rgb_stochastic(
    lic_rgb: np.ndarray,
    rng: np.random.Generator,
    sharpness: float = 2.5,
) -> np.ndarray:
    """Each output pixel is one pure primary, sampled from the LIC probability distribution.

    lic_rgb:   (H, W, 3) float32 — channels sum to ~1 after LIC of pure_rgb_noise.
    sharpness: exponent applied to probabilities before sampling.
               1.0 = raw LIC distribution (often too flat to show structure).
               >1  = pushes the dominant colour toward certainty, making streamlines
                     more visible. Equivalent to softmax temperature T = 1/sharpness.
               Very high values (>10) approach argmax / hard assignment.

    Returns (H, W, 3) uint8 where every pixel is (255,0,0), (0,255,0), or (0,0,255).
    """
    H, W, _ = lic_rgb.shape
    total = lic_rgb.sum(axis=2, keepdims=True)
    probs = np.where(total > 1e-8, lic_rgb / total, np.full_like(lic_rgb, 1.0 / 3.0))

    if sharpness != 1.0:
        probs = np.power(probs, sharpness)

    probs = probs.reshape(-1, 3).astype(np.float64)
    probs /= probs.sum(axis=1, keepdims=True)

    cumprobs = probs.cumsum(axis=1)
    u = rng.random(H * W)
    choices = np.clip((u[:, np.newaxis] >= cumprobs).sum(axis=1), 0, 2)

    pure = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    return pure[choices].reshape(H, W, 3)


def pure_rgb_streamline(
    lic_rgb: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    rng: np.random.Generator,
    num_steps: int = 30,
    step_size: float | None = None,
    stride: int = 4,
) -> np.ndarray:
    """Streamline-paint pure RGB: each streamline is painted one dominant pure colour.

    Traces streamlines from a strided seed grid. Each streamline takes a majority
    vote from the LIC probability distribution along its path and is painted
    uniformly with the winning pure colour. All seeds are stepped simultaneously
    via vectorised numpy operations.

    stride:   seed spacing in pixels — larger = fewer, longer, more visible fibres.
    Pixels not covered by any seeded streamline get stochastic fallback.
    """
    H, W, _ = lic_rgb.shape
    if step_size is None:
        step_size = 1.0 / max(H, W)
    px_step = step_size * max(H, W)

    pure = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    # Seed grid
    seed_r, seed_c = np.meshgrid(
        np.arange(stride // 2, H, stride),
        np.arange(stride // 2, W, stride),
        indexing="ij",
    )
    sr = seed_r.ravel().astype(np.int32)
    sc = seed_c.ravel().astype(np.int32)
    N = len(sr)
    S = num_steps + 1  # steps including start

    # Trace all seeds simultaneously, recording every visited pixel
    paths_r = np.empty((N, S), dtype=np.int32)
    paths_c = np.empty((N, S), dtype=np.int32)
    paths_r[:, 0] = sr
    paths_c[:, 0] = sc

    # Accumulate LIC colour along each seed's path
    color_accum = lic_rgb[sr, sc].astype(np.float32).copy()  # (N, 3)

    xr = sc.astype(np.float64) + 0.5
    yr = sr.astype(np.float64) + 0.5

    for step in range(1, S):
        ri = np.clip(yr.astype(np.int32), 0, H - 1)
        ci = np.clip(xr.astype(np.int32), 0, W - 1)
        # Direct array indexing — no scipy overhead
        dvx = vx[ri, ci]
        dvy = vy[ri, ci]
        xr = np.clip(xr + px_step * dvx, 0.5, W - 0.5)
        yr = np.clip(yr + px_step * dvy, 0.5, H - 0.5)
        ri_new = np.clip(yr.astype(np.int32), 0, H - 1)
        ci_new = np.clip(xr.astype(np.int32), 0, W - 1)
        paths_r[:, step] = ri_new
        paths_c[:, step] = ci_new
        color_accum += lic_rgb[ri_new, ci_new]

    # Determine winning colour per seed (majority vote along path)
    avg = color_accum / S
    choice = np.argmax(avg, axis=1)          # (N,) — index into pure colours
    seed_colors = pure[choice]               # (N, 3) uint8

    # Paint: flatten all path pixels and assign seed colours
    flat_r = paths_r.ravel()                 # (N*S,)
    flat_c = paths_c.ravel()
    flat_colors = np.repeat(seed_colors, S, axis=0)  # (N*S, 3)

    result = np.zeros((H, W, 3), dtype=np.uint8)
    assigned = np.zeros((H, W), dtype=bool)
    result[flat_r, flat_c] = flat_colors
    assigned[flat_r, flat_c] = True

    # Stochastic fallback for uncovered pixels
    if not assigned.all():
        stoch = pure_rgb_stochastic(lic_rgb, rng)
        result[~assigned] = stoch[~assigned]

    return result


# ---------------------------------------------------------------------------
# PIL conversion
# ---------------------------------------------------------------------------

def to_pil_image(arr: np.ndarray) -> Image.Image:
    """Convert (H, W, 3) uint8 array to PIL Image."""
    return Image.fromarray(arr, mode="RGB")
