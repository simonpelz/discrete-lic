// noise.js — CPU-generated noise textures for the grid LIC input.
//
// OWNED BY AGENT A. Mirrors lic_noise.py semantics:
//   "white"     -> uniform [0,1) replicated across RGB (grayscale); A=1.
//   "white_rgb" -> independent uniform [0,1) per channel; A=1.
//   "pure_rgb"  -> each texel one-hot over a pure primary R/G/B; A=1.
//
// The noise texture is rgba32f (per CONTRACT). The LIC pass convolves all four
// channels uniformly; alpha is carried as a constant 1.0 and ignored downstream.
// Generation is CPU-side (Float32Array uploaded once) so the stream is fully
// determined by the seeded rng — matching the Python reference which builds the
// noise array up front.

import { createTexture } from "./glutil.js";

// makeNoise(gl, kind, w, h, rng) -> { texture }
// `rng` is a makeRng(seed) object; we consume rng.random()/rng.randint() so the
// same seed yields the same texture.
export function makeNoise(gl, kind, w, h, rng) {
  const n = w * h;
  const data = new Float32Array(n * 4);

  if (kind === "white") {
    for (let i = 0; i < n; i++) {
      const v = rng.random();
      const o = i * 4;
      data[o] = v; data[o + 1] = v; data[o + 2] = v; data[o + 3] = 1.0;
    }
  } else if (kind === "white_rgb") {
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      data[o] = rng.random();
      data[o + 1] = rng.random();
      data[o + 2] = rng.random();
      data[o + 3] = 1.0;
    }
  } else if (kind === "pure_rgb") {
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      const c = rng.randint(3); // 0=R, 1=G, 2=B
      data[o] = c === 0 ? 1.0 : 0.0;
      data[o + 1] = c === 1 ? 1.0 : 0.0;
      data[o + 2] = c === 2 ? 1.0 : 0.0;
      data[o + 3] = 1.0;
    }
  } else {
    throw new Error("makeNoise: unsupported kind '" + kind + "' (use white, white_rgb, pure_rgb).");
  }

  // NEAREST filtering: noise is piecewise-constant per texel, like the Python
  // array; the LIC shader does its own bilinear-equivalent sampling of indices.
  const texture = createTexture(gl, w, h, "rgba32f", data, gl.NEAREST);
  return { texture };
}
