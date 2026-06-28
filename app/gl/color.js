// color.js — final color stage for the in-browser LIC pipeline (Agent C).
//
// Public API:
//   applyColor(gl, ctx, licTarget, fieldTarget, params) -> Promise<void>
//     Renders the final RGBA8 image to the canvas (bind target null).
//
// Mirrors src/lic_color.py. The single fragment shader (color.glsl) implements:
//   rgb        -> rgb_lic_to_image   (clamp LIC -> 8-bit)
//   pure_rgb   -> pure_rgb_stochastic (exact-ish; GLSL hash RNG, NOT numpy RNG)
//                 pure_rgb_streamline (APPROXIMATED — see note below)
//   angle_hsv  -> colorize_by_angle  (hue from field angle, value from LIC gray)
//   colormap   -> apply_colormap     (grayscale LIC -> 256x1 matplotlib LUT PNG)
//
// `enhance` (stretch / gamma / none) is applied to the LIC sample inside the
// shader BEFORE the color strategy, matching enhance_contrast() running first in
// the Python pipeline. 'equalize' is treated as 'stretch' (the Voronoi path does
// the same). STRETCH needs per-channel min/max over the whole LIC texture; we
// read it back on the CPU via readFloatPixels (simple and exact).
//
// STREAMLINE DEVIATION
// --------------------
// pure_rgb_streamline in Python traces a *strided seed grid*, majority-votes a
// pure colour per streamline, paints every visited pixel with it, and fills
// uncovered pixels with the stochastic fallback. That is order-dependent
// scatter-painting with sparse coverage — not expressible as an independent
// per-fragment shader. The GPU port instead traces a short forward streamline
// *from every pixel*, accumulates the (enhanced) LIC distribution along it, and
// paints the pixel its own dominant primary (argmax). Result: similar coherent
// single-colour fibres aligned with the flow, but denser/fully-covered and with
// no seed-grid sparsity or stochastic fallback. Visual character matches; exact
// fibre layout does not. Stochastic mode is the faithful one.

import {
  createProgram, drawQuad, bindTarget, bindTextureUnit,
  createTexture, readFloatPixels,
} from "./glutil.js";

// ---------------------------------------------------------------------------
// Shader source: load color.glsl once (single source of truth), relative to
// this module. Awaited lazily on first applyColor call and cached.
// ---------------------------------------------------------------------------
let _fragSrcPromise = null;
function loadFragSrc() {
  if (!_fragSrcPromise) {
    const url = new URL("./color.glsl", import.meta.url);
    _fragSrcPromise = fetch(url).then((r) => {
      if (!r.ok) throw new Error("color.js: failed to load color.glsl (" + r.status + ")");
      return r.text();
    });
  }
  return _fragSrcPromise;
}

const MODE = { rgb: 0, pure_rgb: 1, angle_hsv: 2, colormap: 3 };
const ENHANCE = { none: 0, stretch: 1, gamma: 2, equalize: 1 }; // equalize -> stretch

// Cache the compiled program + uniform locations on ctx so we compile once.
function getProgram(gl, ctx, fragSrc) {
  if (!ctx._colorProg) {
    const prog = createProgram(gl, fragSrc);
    ctx._colorProg = prog;
    ctx._colorLoc = {
      uMode: gl.getUniformLocation(prog, "uMode"),
      uPureMode: gl.getUniformLocation(prog, "uPureMode"),
      uEnhance: gl.getUniformLocation(prog, "uEnhance"),
      uLicMin: gl.getUniformLocation(prog, "uLicMin"),
      uLicMax: gl.getUniformLocation(prog, "uLicMax"),
      uGamma: gl.getUniformLocation(prog, "uGamma"),
      uSharpen: gl.getUniformLocation(prog, "uSharpen"),
      uSat: gl.getUniformLocation(prog, "uSat"),
      uSeed: gl.getUniformLocation(prog, "uSeed"),
      uTexel: gl.getUniformLocation(prog, "uTexel"),
      uSteps: gl.getUniformLocation(prog, "uSteps"),
      uStepPx: gl.getUniformLocation(prog, "uStepPx"),
    };
  }
  return { prog: ctx._colorProg, loc: ctx._colorLoc };
}

// ---------------------------------------------------------------------------
// Colormap LUT textures: lazily fetch app/assets/colormaps/<name>.png as a
// 256x1 RGB texture, cached per ctx.
// ---------------------------------------------------------------------------
function colormapURL(name) {
  // color.js lives in app/gl/; colormaps live in app/assets/colormaps/.
  return new URL("../assets/colormaps/" + name + ".png", import.meta.url);
}

async function loadColormap(gl, ctx, name) {
  if (!ctx._colormaps) ctx._colormaps = new Map();
  const cache = ctx._colormaps;
  if (cache.has(name)) return cache.get(name);

  const promise = (async () => {
    const url = colormapURL(name);
    let imgBitmap;
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error("status " + resp.status);
      const blob = await resp.blob();
      imgBitmap = await createImageBitmap(blob);
    } catch (e) {
      throw new Error("color.js: failed to load colormap '" + name + "': " + e.message);
    }
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // LUT is authored left->right = 0->1; sample with CLAMP + LINEAR for smooth
    // interpolation between the 256 entries.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgBitmap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  })();

  cache.set(name, promise);
  return promise;
}

// A 1x1 placeholder so colormap sampler always has a bound texture even when the
// LUT isn't needed (non-colormap modes) — keeps the draw well-defined.
function dummyTex(gl, ctx) {
  if (!ctx._colorDummyTex) {
    ctx._colorDummyTex = createTexture(
      gl, 1, 1, "rgba8", new Uint8Array([0, 0, 0, 255]), gl.NEAREST,
    );
  }
  return ctx._colorDummyTex;
}

// ---------------------------------------------------------------------------
// Per-channel min/max over the LIC texture (CPU readback) for stretch enhance.
// Mirrors numpy's per-channel arr.min()/arr.max() global stretch.
// ---------------------------------------------------------------------------
function licMinMax(gl, licTarget) {
  const f = readFloatPixels(gl, licTarget); // length = w*h*channels
  const ch = licTarget.kind === "r32f" ? 1 : (licTarget.kind === "rg32f" ? 2 : 4);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const n = licTarget.w * licTarget.h;
  for (let i = 0; i < n; i++) {
    const o = i * ch;
    for (let c = 0; c < 3; c++) {
      // r32f: replicate single channel into all three (grayscale).
      const v = ch === 1 ? f[o] : f[o + Math.min(c, ch - 1)];
      if (v < min[c]) min[c] = v;
      if (v > max[c]) max[c] = v;
    }
  }
  for (let c = 0; c < 3; c++) {
    if (!isFinite(min[c])) min[c] = 0.0;
    if (!isFinite(max[c])) max[c] = 1.0;
  }
  return { min, max };
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
export async function applyColor(gl, ctx, licTarget, fieldTarget, params) {
  const fragSrc = await loadFragSrc();
  const { prog, loc } = getProgram(gl, ctx, fragSrc);

  const colorMode = params.color in MODE ? params.color : "rgb";
  const mode = MODE[colorMode];
  const enhance = ENHANCE[params.enhance] !== undefined ? ENHANCE[params.enhance] : ENHANCE.stretch;

  // Resolve colormap texture up front (await before any GL state we want stable).
  let cmapTex = dummyTex(gl, ctx);
  if (colorMode === "colormap") {
    cmapTex = await loadColormap(gl, ctx, params.colormap || "viridis");
  }

  // Stretch min/max (only needed for stretch enhance, and pure_rgb skips enhance
  // entirely — matching lic_main.py — so no readback there).
  let mn = [0, 0, 0];
  let mx = [1, 1, 1];
  if (enhance === ENHANCE.stretch && colorMode !== "pure_rgb") {
    const r = licMinMax(gl, licTarget);
    mn = r.min; mx = r.max;
  }

  // Geometry for the streamline trace approximation.
  const W = licTarget.w, H = licTarget.h;
  const maxDim = Math.max(W, H);
  const steps = Math.max(0, Math.min(64, params.steps | 0)); // shader caps at 64
  let stepSize = params.step_size;
  if (stepSize === null || stepSize === undefined) stepSize = 1.0 / maxDim;
  const stepPx = stepSize * maxDim;

  bindTarget(gl, null); // -> canvas
  gl.disable(gl.BLEND);
  gl.useProgram(prog);

  bindTextureUnit(gl, prog, "uLic", licTarget.texture, 0);
  bindTextureUnit(gl, prog, "uField", fieldTarget ? fieldTarget.texture : dummyTex(gl, ctx), 1);
  bindTextureUnit(gl, prog, "uColormap", cmapTex, 2);

  gl.uniform1i(loc.uMode, mode);
  gl.uniform1i(loc.uPureMode, params.pure_mode === "streamline" ? 1 : 0);
  gl.uniform1i(loc.uEnhance, enhance);
  gl.uniform3f(loc.uLicMin, mn[0], mn[1], mn[2]);
  gl.uniform3f(loc.uLicMax, mx[0], mx[1], mx[2]);
  gl.uniform1f(loc.uGamma, Number(params.gamma) || 1.0);
  gl.uniform1f(loc.uSharpen, params.pure_sharpen != null ? Number(params.pure_sharpen) : 2.5);
  gl.uniform1f(loc.uSat, params.sat != null ? Number(params.sat) : 0.9);
  gl.uniform1f(loc.uSeed, (Number(params.seed) || 0) % 100000);
  gl.uniform2f(loc.uTexel, 1.0 / W, 1.0 / H);
  gl.uniform1i(loc.uSteps, steps);
  gl.uniform1f(loc.uStepPx, stepPx);

  drawQuad(gl);
}
