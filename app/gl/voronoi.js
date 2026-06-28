// voronoi.js — Voronoi-pixel LIC path for the in-browser WebGL2 port (Agent E).
//
//   export function renderVoronoi(gl, ctx, params)
//
// This is the mode the real artwork uses. It mirrors src/lic_voronoi.py:
//   build_seeds → build_cell_map(JFA) → cell_noise_* → compute_voronoi_lic
//   → per-cell color → render_cells(paint)
//
// Pipeline (all GPU except seed generation / JFA-init scatter):
//   1. Seeds: N = voronoi_cells random (cx,cy) in [0,1] via makeRng(seed).
//      Uploaded as an Nx1 rg32f texture (uSeeds).
//   2. JFA cell map: a W×H R32UI texture storing nearest-seed INDEX+1 per pixel.
//      Built by ping-ponging jfa.frag with steps W/2, W/4, ... 1. Replaces
//      scipy.cKDTree. (Encoding: index+1, 0 = empty; positions come from uSeeds,
//      so the metric is decoupled from the >255-capable index payload.)
//   3. Per-cell noise: an Nx1 rgba32f texture, one texel per cell, generated CPU-
//      side with the seeded rng to match cell_noise_white/white_rgb/hsv/pure_rgb.
//   4. Per-cell LIC: voronoi-lic.frag traces each cell's streamline (packed into
//      a PACKW×PACKH rgba32f target) sampling the cell map piecewise-constant.
//   5. Per-cell color: a packed Nx? RGBA8 texture. pure_rgb is sampled ONCE PER
//      CELL (categorical pick, sharpen exponent) — NOT per pixel — so the mosaic
//      stays crisp. angle_hsv / rgb / colormap are deterministic per-cell funcs.
//   6. Paint: read cell map per screen pixel → index the per-cell color texture
//      → RGBA8 to canvas (render_cells). SSAA 2× target, then box-downsample to
//      the canvas for crisp edges. Exactly ONE net Y-flip happens here.
//
// DEPENDENCIES: field.js (Agent B) createFieldTexture, imported defensively like
// pipeline.js — with a rotation/double_vortex fallback if absent. rng.js makeRng.

import {
  createProgram, createTarget, createTexture,
  bindTarget, bindTextureUnit, drawQuad,
} from "./glutil.js";
import { makeRng } from "./rng.js";

// ---------------------------------------------------------------------------
// Shader sources fetched once (the two canonical .frag files) + inline passes.
// ---------------------------------------------------------------------------
let _jfaSrc = null;
let _licSrc = null;

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("voronoi.js: failed to fetch " + url + " (" + res.status + ")");
  return res.text();
}

async function getPrograms(gl, ctx) {
  if (ctx._voro) return ctx._voro;
  if (_jfaSrc == null) _jfaSrc = await fetchText(new URL("./jfa.frag", import.meta.url));
  if (_licSrc == null) _licSrc = await fetchText(new URL("./voronoi-lic.frag", import.meta.url));
  const v = {
    jfa: createProgram(gl, _jfaSrc),
    lic: createProgram(gl, _licSrc),
    color: createProgram(gl, COLOR_FRAG),
    paint: createProgram(gl, PAINT_FRAG),
  };
  ctx._voro = v;
  return v;
}

// Defensive field import (mirrors pipeline.js). Falls back to a synthetic
// rotation/double_vortex field if field.js is missing.
let _fieldMod = null;
let _fieldTried = false;
async function loadFieldMod() {
  if (_fieldTried) return _fieldMod;
  _fieldTried = true;
  try { _fieldMod = await import("./field.js"); } catch (e) { _fieldMod = null; }
  return _fieldMod;
}

// ---------------------------------------------------------------------------
// renderVoronoi — public entry point (matches pipeline.renderVoronoi delegation)
// ---------------------------------------------------------------------------
export async function renderVoronoi(gl, ctx, params) {
  const p = params;
  const N = Math.max(1, p.voronoi_cells | 0);
  const canvasW = gl.canvas.width;
  const canvasH = gl.canvas.height;

  const progs = await getPrograms(gl, ctx);

  // No blending anywhere in the voronoi path: every pass writes exact values
  // (JFA indices, LIC accum, categorical colours, render_cells lookup).
  gl.disable(gl.BLEND);

  // Free the transient textures (seed/noise/rand) created last render — their
  // contents depend on seed/params, so they're regenerated each call. The
  // ping-pong/LIC/colour TARGETS are cached and reused (size-keyed) elsewhere.
  freeTransients(gl, ctx);

  // --- 1. FIELD (Agent B, defensive) ------------------------------------
  const fieldMod = await loadFieldMod();
  let field;
  if (fieldMod && typeof fieldMod.createFieldTexture === "function") {
    field = await fieldMod.createFieldTexture(gl, p); // { texture, w, h }
  } else {
    field = makeFallbackField(gl, ctx, p);
  }
  const W = field.w, H = field.h; // cell map authored at field resolution

  // --- 2. SEEDS ---------------------------------------------------------
  const rng = makeRng(p.seed);
  // build_seeds: rng.random((N,2)) — row i = (cx, cy). Draw cx then cy.
  const seedData = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    seedData[i * 2] = rng.random();     // cx (col-frac)
    seedData[i * 2 + 1] = rng.random(); // cy (row-frac, bottom-origin frame)
  }
  const seedTex = createTexture(gl, N, 1, "rg32f", seedData, gl.NEAREST);
  trackTransient(ctx, seedTex);

  // --- 3. JFA CELL MAP --------------------------------------------------
  const cellMap = buildCellMap(gl, ctx, progs.jfa, seedTex, seedData, N, W, H);

  // --- 4. PER-CELL NOISE ------------------------------------------------
  const noiseTex = makeCellNoise(gl, p, N, rng);
  trackTransient(ctx, noiseTex);

  // --- 5. PER-CELL LIC --------------------------------------------------
  const pack = packDims(N);
  const lic = getTarget(ctx, "voroLic", pack.w, pack.h, "rgba32f");
  runCellLic(gl, progs.lic, lic, field, seedTex, cellMap, noiseTex, N, pack, W, H, p);

  // --- 6. PER-CELL COLOR ------------------------------------------------
  const colorTex = getTarget(ctx, "voroColor", pack.w, pack.h, "rgba8");
  runCellColor(gl, ctx, progs.color, colorTex, lic, field, seedTex, N, pack, W, H, p, rng);

  // --- 7. PAINT (render_cells) → SSAA → canvas --------------------------
  paint(gl, ctx, progs.paint, cellMap, colorTex.texture, N, pack, W, H, canvasW, canvasH);
}

// ---------------------------------------------------------------------------
// Transient texture registry: seed/noise/rand textures are content-dependent on
// seed+params, so they're regenerated every render and freed at the next call.
// ---------------------------------------------------------------------------
function trackTransient(ctx, tex) {
  if (!ctx._voroTransients) ctx._voroTransients = [];
  ctx._voroTransients.push(tex);
}
function freeTransients(gl, ctx) {
  if (!ctx._voroTransients) return;
  for (const t of ctx._voroTransients) gl.deleteTexture(t);
  ctx._voroTransients = [];
}

// ---------------------------------------------------------------------------
// Target cache (per ctx, keyed by name+size+kind).
// ---------------------------------------------------------------------------
function getTarget(ctx, key, w, h, kind) {
  if (!ctx._voroTargets) ctx._voroTargets = {};
  const t = ctx._voroTargets[key];
  if (t && t.w === w && t.h === h && t.kind === kind) return t;
  const nt = createTarget(ctx.gl, w, h, kind);
  ctx._voroTargets[key] = nt;
  return nt;
}

// An integer-format render target (R32UI) for the JFA cell map. glutil's FORMATS
// don't include R32UI, so build it here.
function getUintTarget(ctx, key, w, h) {
  const gl = ctx.gl;
  if (!ctx._voroTargets) ctx._voroTargets = {};
  const t = ctx._voroTargets[key];
  if (t && t.w === w && t.h === h) return t;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, w, h, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error("voronoi.js: R32UI framebuffer incomplete");
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  const nt = { texture: tex, fbo, w, h, kind: "r32ui" };
  ctx._voroTargets[key] = nt;
  return nt;
}

// Pack N cells into a near-square PACKW×PACKH grid (cell = iy*PACKW + ix).
function packDims(N) {
  const w = Math.max(1, Math.ceil(Math.sqrt(N)));
  const h = Math.max(1, Math.ceil(N / w));
  return { w, h };
}

// ---------------------------------------------------------------------------
// JFA cell map: init nearest-seed scatter (index+1) on the CPU, then ping-pong.
// ---------------------------------------------------------------------------
function buildCellMap(gl, ctx, prog, seedTex, seedData, N, W, H) {
  // Seed-init scatter: each seed claims its own pixel (last writer wins on a
  // collision, harmless — JFA propagates the true nearest from there).
  const init = new Uint32Array(W * H); // 0 = empty
  for (let i = 0; i < N; i++) {
    const cx = seedData[i * 2];
    const cy = seedData[i * 2 + 1];
    let col = Math.floor(cx * W);
    let row = Math.floor(cy * H);
    if (col < 0) col = 0; else if (col >= W) col = W - 1;
    if (row < 0) row = 0; else if (row >= H) row = H - 1;
    init[row * W + col] = i + 1; // index+1
  }

  const a = getUintTarget(ctx, "jfaA", W, H);
  const b = getUintTarget(ctx, "jfaB", W, H);
  // Upload init into A.
  gl.bindTexture(gl.TEXTURE_2D, a.texture);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RED_INTEGER, gl.UNSIGNED_INT, init);
  gl.bindTexture(gl.TEXTURE_2D, null);

  let src = a, dst = b;
  gl.useProgram(prog);
  gl.uniform2f(gl.getUniformLocation(prog, "uResolution"), W, H);
  gl.uniform1i(gl.getUniformLocation(prog, "uNumSeeds"), N);

  // Steps: largest power of two < max(W,H), halving down to 1.
  let step = 1;
  while (step < Math.max(W, H)) step <<= 1;
  step >>= 1;

  for (; step >= 1; step >>= 1) {
    bindTarget(gl, dst);
    gl.useProgram(prog);
    gl.uniform1i(gl.getUniformLocation(prog, "uStep"), step);
    bindTextureUnit(gl, prog, "uPrev", src.texture, 0);
    bindTextureUnit(gl, prog, "uSeeds", seedTex, 1);
    drawQuad(gl);
    const tmp = src; src = dst; dst = tmp;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return src.texture; // R32UI: nearest-seed index+1 per pixel
}

// ---------------------------------------------------------------------------
// Per-cell noise (CPU), matching cell_noise_* in lic_voronoi.py. Stored Nx1
// rgba32f. The LIC pass convolves all 4 channels; coloring reads what it needs.
// ---------------------------------------------------------------------------
function makeCellNoise(gl, p, N, rng) {
  const data = new Float32Array(N * 4);
  const kind = cellNoiseKind(p);
  if (kind === "pure_rgb") {
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      const c = rng.randint(3); // 0=R,1=G,2=B (cell_noise_pure_rgb)
      data[o] = c === 0 ? 1 : 0;
      data[o + 1] = c === 1 ? 1 : 0;
      data[o + 2] = c === 2 ? 1 : 0;
      data[o + 3] = 1;
    }
  } else if (kind === "white") {
    for (let i = 0; i < N; i++) {
      const v = rng.random();
      const o = i * 4;
      data[o] = v; data[o + 1] = v; data[o + 2] = v; data[o + 3] = 1;
    }
  } else if (kind === "hsv") {
    // cell_noise_hsv: h uniform, s/v in [0.85,1]; convert to RGB.
    for (let i = 0; i < N; i++) {
      const h = rng.random();
      const s = 0.85 + 0.15 * rng.random();
      const val = 0.85 + 0.15 * rng.random();
      const [r, g, b] = hsvToRgb(h, s, val);
      const o = i * 4;
      data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 1;
    }
  } else { // white_rgb (default)
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      data[o] = rng.random();
      data[o + 1] = rng.random();
      data[o + 2] = rng.random();
      data[o + 3] = 1;
    }
  }
  return createTexture(gl, N, 1, "rgba32f", data, gl.NEAREST);
}

// Decide the per-cell noise kind. pure_rgb color implies pure_rgb noise; else
// honour params.noise, defaulting to white_rgb (voronoi mode bans block/voronoi
// noise per CLAUDE.md, so those degrade to white_rgb).
function cellNoiseKind(p) {
  if (p.color === "pure_rgb") return "pure_rgb";
  if (p.noise === "white" || p.noise === "white_rgb" || p.noise === "hsv") return p.noise;
  // angle_hsv / colormap derive a gray = mean(RGB) of the cell LIC (see the
  // oracle's cell_lic.mean(axis=1)); white_rgb is the default substrate.
  return "white_rgb";
}

// ---------------------------------------------------------------------------
// Per-cell LIC pass.
// ---------------------------------------------------------------------------
function runCellLic(gl, prog, lic, field, seedTex, cellMap, noiseTex, N, pack, W, H, p) {
  const stepSize = (p.step_size != null) ? Number(p.step_size) : (1.0 / Math.sqrt(N));
  const steps = Math.max(0, Math.min(64, p.steps | 0));
  const KERNEL_ID = { box: 0, gaussian: 1, raised_cosine: 2 };
  const BOUNDARY_ID = { nearest: 0, wrap: 1 };

  bindTarget(gl, lic);
  gl.useProgram(prog);
  bindTextureUnit(gl, prog, "uField", field.texture, 0);
  bindTextureUnit(gl, prog, "uSeeds", seedTex, 1);
  bindTextureUnit(gl, prog, "uCellMap", cellMap, 2);
  bindTextureUnit(gl, prog, "uNoise", noiseTex, 3);
  gl.uniform2f(gl.getUniformLocation(prog, "uResolution"), W, H);
  gl.uniform2i(gl.getUniformLocation(prog, "uPack"), pack.w, pack.h);
  gl.uniform1i(gl.getUniformLocation(prog, "uNumSeeds"), N);
  gl.uniform1i(gl.getUniformLocation(prog, "uSteps"), steps);
  gl.uniform1f(gl.getUniformLocation(prog, "uStepSize"), stepSize);
  gl.uniform1i(gl.getUniformLocation(prog, "uKernel"), KERNEL_ID[p.kernel] ?? 0);
  gl.uniform1f(gl.getUniformLocation(prog, "uKernelSigma"), Number(p.kernel_sigma) || 0.3);
  gl.uniform1i(gl.getUniformLocation(prog, "uBoundary"), BOUNDARY_ID[p.boundary] ?? 0);
  drawQuad(gl);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// ---------------------------------------------------------------------------
// Per-cell color pass. Computes one RGBA8 colour per cell into the packed grid.
//   pure_rgb stochastic: normalize → sharpen exponent → categorical pick (per
//     cell, ONE draw) → a pure primary. The per-cell seed-derived random is the
//     crispness-preserving key — coloring is per CELL, not per pixel.
//   angle_hsv: hue from field angle at the seed, value from (enhanced) LIC gray.
//   rgb / colormap: deterministic function of the (enhanced) LIC value.
// Stretch enhance needs per-channel min/max over the N cell LIC values; computed
// here on the CPU (N is small) and passed as uniforms, mirroring enhance_cells.
// ---------------------------------------------------------------------------
function runCellColor(gl, ctx, prog, colorTex, lic, field, seedTex, N, pack, W, H, p, rng) {
  const MODE = { rgb: 0, pure_rgb: 1, angle_hsv: 2, colormap: 3 };
  const ENHANCE = { none: 0, stretch: 1, gamma: 2, equalize: 1 };
  const mode = MODE[p.color] !== undefined ? MODE[p.color] : 0;
  const enhance = ENHANCE[p.enhance] !== undefined ? ENHANCE[p.enhance] : 1;

  // Per-cell stochastic key: one uniform random per cell, consumed in-shader to
  // mirror pure_rgb_cells_stochastic's `u = rng.random(N)`. We draw them here
  // from the same rng stream (after seeds + noise) and upload as an Nx1 r/g tex.
  let randTex = null;
  if (p.color === "pure_rgb") {
    const r = new Float32Array(pack.w * pack.h * 2);
    for (let i = 0; i < N; i++) r[i * 2] = rng.random();
    randTex = createTexture(gl, pack.w, pack.h, "rg32f", r, gl.NEAREST);
    trackTransient(ctx, randTex);
  }

  // Stretch min/max over the N cell LIC values (skip for pure_rgb — matches the
  // Python pipeline, which does NOT enhance before pure_rgb sampling).
  let mn = [0, 0, 0], mx = [1, 1, 1];
  // angle_hsv / colormap stretch on the GRAY = mean(RGB) channel; rgb stretches
  // per RGB channel — mirror enhance_cells operating on the right array.
  const grayMode = (p.color === "angle_hsv" || p.color === "colormap");
  if (enhance === 1 && p.color !== "pure_rgb") {
    const buf = new Float32Array(pack.w * pack.h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, lic.fbo);
    gl.readPixels(0, 0, pack.w, pack.h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    mn = [Infinity, Infinity, Infinity];
    mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < N; i++) {
      if (grayMode) {
        const g = (buf[i * 4] + buf[i * 4 + 1] + buf[i * 4 + 2]) / 3.0;
        if (g < mn[0]) mn[0] = g;
        if (g > mx[0]) mx[0] = g;
      } else {
        for (let c = 0; c < 3; c++) {
          const v = buf[i * 4 + c];
          if (v < mn[c]) mn[c] = v;
          if (v > mx[c]) mx[c] = v;
        }
      }
    }
    if (grayMode) { mn[1] = mn[2] = mn[0]; mx[1] = mx[2] = mx[0]; }
    for (let c = 0; c < 3; c++) {
      if (!isFinite(mn[c])) mn[c] = 0;
      if (!isFinite(mx[c])) mx[c] = 1;
    }
  }

  bindTarget(gl, colorTex);
  gl.useProgram(prog);
  bindTextureUnit(gl, prog, "uLic", lic.texture, 0);
  bindTextureUnit(gl, prog, "uField", field.texture, 1);
  bindTextureUnit(gl, prog, "uSeeds", seedTex, 2);
  bindTextureUnit(gl, prog, "uRand", randTex || seedTex, 3);
  gl.uniform2i(gl.getUniformLocation(prog, "uPack"), pack.w, pack.h);
  gl.uniform1i(gl.getUniformLocation(prog, "uNumSeeds"), N);
  gl.uniform1i(gl.getUniformLocation(prog, "uMode"), mode);
  gl.uniform1i(gl.getUniformLocation(prog, "uEnhance"), enhance);
  gl.uniform3f(gl.getUniformLocation(prog, "uLicMin"), mn[0], mn[1], mn[2]);
  gl.uniform3f(gl.getUniformLocation(prog, "uLicMax"), mx[0], mx[1], mx[2]);
  gl.uniform1f(gl.getUniformLocation(prog, "uGamma"), Number(p.gamma) || 1.0);
  gl.uniform1f(gl.getUniformLocation(prog, "uSharpen"), p.pure_sharpen != null ? Number(p.pure_sharpen) : 2.5);
  gl.uniform1f(gl.getUniformLocation(prog, "uSat"), p.sat != null ? Number(p.sat) : 0.9);
  drawQuad(gl);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// ---------------------------------------------------------------------------
// Paint: render_cells. Read the cell map per screen pixel → index the per-cell
// color texture → RGBA8. Rendered at SSAA into a target, then box-downsampled to
// the canvas for crisp anti-aliased Voronoi edges. Exactly ONE net Y-flip (here).
// ---------------------------------------------------------------------------
function paint(gl, ctx, prog, cellMap, colorTex, N, pack, W, H, canvasW, canvasH) {
  const SSAA = 2;
  const ssW = canvasW * SSAA, ssH = canvasH * SSAA;
  const big = getTarget(ctx, "voroSSAA", ssW, ssH, "rgba8");

  // Pass 1: render_cells at SSAA, NO flip (keep bottom-origin so the cell map's
  // pixel order matches). The flip happens in the downsample present.
  bindTarget(gl, big);
  gl.useProgram(prog);
  bindUintTextureUnit(gl, prog, "uCellMap", cellMap, 0);
  bindTextureUnit(gl, prog, "uColor", colorTex, 1);
  gl.uniform2f(gl.getUniformLocation(prog, "uMapRes"), W, H);
  gl.uniform2i(gl.getUniformLocation(prog, "uPack"), pack.w, pack.h);
  gl.uniform1i(gl.getUniformLocation(prog, "uFlip"), 0);
  drawQuad(gl);

  // Pass 2: downsample SSAA → canvas with the single net Y-flip.
  bindTarget(gl, null);
  const dprog = getDownsampleProgram(gl, ctx);
  gl.useProgram(dprog);
  bindTextureUnit(gl, dprog, "uTex", big.texture, 0);
  gl.uniform2f(gl.getUniformLocation(dprog, "uTexel"), 1.0 / ssW, 1.0 / ssH);
  gl.uniform1i(gl.getUniformLocation(dprog, "uSSAA"), SSAA);
  drawQuad(gl);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function getDownsampleProgram(gl, ctx) {
  if (!ctx._voroDown) ctx._voroDown = createProgram(gl, DOWNSAMPLE_FRAG);
  return ctx._voroDown;
}

// bindTextureUnit variant that sets an integer sampler (usampler2D) — identical
// binding, but kept explicit for clarity.
function bindUintTextureUnit(gl, prog, name, texture, unit) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform1i(loc, unit);
}

// ---------------------------------------------------------------------------
// Fallback field (rotation / double_vortex) when field.js is unavailable.
// rg32f, top-left authored, encoded enc = v*0.5+0.5 (decode v = tex*2-1).
// ---------------------------------------------------------------------------
function makeFallbackField(gl, ctx, p) {
  const res = Math.max((p.width | 0) || 0, 512);
  if (!ctx._voroFallbackProg) ctx._voroFallbackProg = createProgram(gl, FALLBACK_FIELD_FRAG);
  const prog = ctx._voroFallbackProg;
  const tgt = getTarget(ctx, "voroFallbackField", res, res, "rg32f");
  bindTarget(gl, tgt);
  gl.useProgram(prog);
  gl.uniform1i(gl.getUniformLocation(prog, "uDoubleVortex"), p.field === "double_vortex" ? 1 : 0);
  drawQuad(gl);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { texture: tgt.texture, w: res, h: res };
}

// ---------------------------------------------------------------------------
// CPU HSV→RGB (matches _hsv_to_rgb_1d) for the hsv noise path.
// ---------------------------------------------------------------------------
function hsvToRgb(h, s, v) {
  const h6 = h * 6.0;
  const i = Math.floor(h6) % 6;
  const f = h6 - Math.floor(h6);
  const pp = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i) {
    case 0: return [v, t, pp];
    case 1: return [q, v, pp];
    case 2: return [pp, v, t];
    case 3: return [pp, q, v];
    case 4: return [t, pp, v];
    default: return [v, pp, q];
  }
}

// ---------------------------------------------------------------------------
// Inline shader sources.
// ---------------------------------------------------------------------------

// Per-cell color. Output: one RGBA8 colour per packed cell texel.
const COLOR_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uLic;     // packed rgba32f per-cell LIC
uniform sampler2D uField;   // rg32f, top-left authored
uniform sampler2D uSeeds;   // rg32f Nx1: seed positions
uniform sampler2D uRand;    // rg32f packed: per-cell uniform random (pure_rgb)
uniform ivec2 uPack;
uniform int   uNumSeeds;
uniform int   uMode;        // 0 rgb, 1 pure_rgb, 2 angle_hsv, 3 colormap
uniform int   uEnhance;     // 0 none, 1 stretch, 2 gamma
uniform vec3  uLicMin;
uniform vec3  uLicMax;
uniform float uGamma;
uniform float uSharpen;
uniform float uSat;

vec3 hsv2rgb(float h, float s, float v) {
  float h6 = h * 6.0;
  float fi = floor(h6);
  int i = int(mod(fi, 6.0));
  float f = h6 - fi;
  float p = v * (1.0 - s);
  float q = v * (1.0 - f * s);
  float t = v * (1.0 - (1.0 - f) * s);
  if (i == 0) return vec3(v, t, p);
  if (i == 1) return vec3(q, v, p);
  if (i == 2) return vec3(p, v, t);
  if (i == 3) return vec3(p, q, v);
  if (i == 4) return vec3(t, p, v);
  return vec3(v, p, q);
}

vec3 enhance(vec3 c) {
  if (uEnhance == 1) { // stretch (per channel)
    vec3 lo = uLicMin, hi = uLicMax;
    vec3 d = max(hi - lo, vec3(1e-12));
    return clamp((c - lo) / d, 0.0, 1.0);
  }
  if (uEnhance == 2) { // gamma
    return clamp(pow(c, vec3(uGamma)), 0.0, 1.0);
  }
  return c;
}

void main() {
  ivec2 ip = ivec2(gl_FragCoord.xy);
  int cell = ip.y * uPack.x + ip.x;
  if (cell >= uNumSeeds) { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }

  vec4 lic = texelFetch(uLic, ip, 0);

  if (uMode == 1) {
    // pure_rgb stochastic, ONE pick per cell (mirrors pure_rgb_cells_stochastic).
    vec3 v = lic.rgb;
    float total = v.r + v.g + v.b;
    vec3 probs = (total > 1e-8) ? v / total : vec3(1.0 / 3.0);
    if (uSharpen != 1.0) probs = pow(probs, vec3(uSharpen));
    float s = probs.r + probs.g + probs.b;
    probs = (s > 0.0) ? probs / s : vec3(1.0 / 3.0);
    float u = texelFetch(uRand, ip, 0).r; // per-cell uniform random
    // categorical pick via cumulative sum: choices = sum(u >= cumprobs).
    float c0 = probs.r;
    float c1 = probs.r + probs.g;
    vec3 col;
    if (u < c0) col = vec3(1.0, 0.0, 0.0);
    else if (u < c1) col = vec3(0.0, 1.0, 0.0);
    else col = vec3(0.0, 0.0, 1.0);
    fragColor = vec4(col, 1.0);
    return;
  }

  if (uMode == 2) {
    // angle_hsv: hue from field angle at seed, value from enhanced LIC gray.
    vec2 seed = texelFetch(uSeeds, ivec2(cell, 0), 0).xy;
    vec2 fuv = vec2(seed.x, 1.0 - seed.y);
    vec2 fv = texture(uField, fuv).xy * 2.0 - 1.0;
    float ang = atan(fv.y, fv.x);
    float hue = (ang + 3.14159265358979) / (2.0 * 3.14159265358979);
    float gray = enhance(vec3((lic.r + lic.g + lic.b) / 3.0)).r;
    fragColor = vec4(hsv2rgb(hue, uSat, clamp(gray, 0.0, 1.0)), 1.0);
    return;
  }

  if (uMode == 3) {
    // colormap: grayscale LIC -> grayscale ramp (LUT not wired in voronoi path;
    // matplotlib LUTs live in color.js. Use the enhanced gray as a stand-in so
    // the path renders; integration may swap a LUT lookup here later).
    float gray = enhance(vec3((lic.r + lic.g + lic.b) / 3.0)).r;
    fragColor = vec4(vec3(gray), 1.0);
    return;
  }

  // rgb (default): direct enhanced LIC -> 8-bit.
  fragColor = vec4(enhance(lic.rgb), 1.0);
}`;

// Paint: render_cells. Reads the cell map (R32UI) at this fragment's uv → cell
// index → per-cell color texel. uFlip is 0 here (flip done in downsample).
const PAINT_FRAG = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
in vec2 vUv;
out vec4 fragColor;

uniform usampler2D uCellMap; // R32UI: nearest-seed index+1
uniform sampler2D  uColor;   // packed RGBA8 per-cell color
uniform vec2  uMapRes;       // (W, H) of cell map
uniform ivec2 uPack;
uniform int   uFlip;

void main() {
  vec2 uv = (uFlip == 1) ? vec2(vUv.x, 1.0 - vUv.y) : vUv;
  ivec2 mp = ivec2(clamp(uv, vec2(0.0), vec2(0.999999)) * uMapRes);
  uint id = texelFetch(uCellMap, mp, 0).r;
  if (id == 0u) { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  int cell = int(id) - 1;
  ivec2 cp = ivec2(cell % uPack.x, cell / uPack.x);
  fragColor = texelFetch(uColor, cp, 0);
}`;

// Downsample SSAA → canvas (box filter over SSAA×SSAA) WITH the single net
// Y-flip so row 0 ends up at the top on the canvas (CONTRACT present rule).
const DOWNSAMPLE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTex;
uniform vec2 uTexel; // 1/ssW, 1/ssH
uniform int  uSSAA;
void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y); // net Y-flip
  // This output pixel covers an SSAA×SSAA block of source texels. Its block
  // origin (top-left source texel centre) is uv shifted back by (SSAA-1)/2.
  vec2 origin = uv - (float(uSSAA) - 1.0) * 0.5 * uTexel;
  vec4 acc = vec4(0.0);
  float n = 0.0;
  for (int j = 0; j < 8; j++) {
    if (j >= uSSAA) break;
    for (int i = 0; i < 8; i++) {
      if (i >= uSSAA) break;
      acc += texture(uTex, origin + vec2(float(i), float(j)) * uTexel);
      n += 1.0;
    }
  }
  fragColor = acc / max(n, 1.0);
}`;

// Fallback rotation / double_vortex field, top-left authored, rg32f enc.
const FALLBACK_FIELD_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec2 fragColor;
uniform int uDoubleVortex;
void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y); // top-left authoring
  vec2 v;
  if (uDoubleVortex == 1) {
    vec2 a = uv - vec2(0.3, 0.5);
    vec2 b = uv - vec2(0.7, 0.5);
    vec2 va = vec2(-a.y, a.x);   // s1 = +1
    vec2 vb = vec2(b.y, -b.x);   // s2 = -1
    v = va + vb;
  } else {
    vec2 c = uv - 0.5;
    v = vec2(-c.y, c.x);         // rotation ccw
  }
  float m = length(v);
  v = (m > 0.0) ? v / m : vec2(0.0);
  fragColor = v * 0.5 + 0.5;
}`;
