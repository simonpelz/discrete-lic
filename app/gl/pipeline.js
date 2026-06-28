// pipeline.js — orchestrator for the in-browser LIC generator.
//
// OWNED BY AGENT A. This version wires the real grid path:
//   field → noise → LIC → color → canvas
// The public API (init / renderGrid / renderVoronoi / exportPNG / DEFAULTS) is
// frozen per CONTRACT.md and must not change signatures.
//
// DEFENSIVE INTEGRATION
// ---------------------
// field.js (Agent B) and color.js (Agent C) may not exist yet. We import them
// lazily and fall back to internal stubs so the grid pipeline renders
// end-to-end on its own:
//   * no createFieldTexture  -> internal rotation-field stub (rg32f).
//   * no applyColor          -> internal direct-RGB color pass.
// Swapping in the real modules later is a one-line change (the dynamic import
// simply succeeds and the stub branch is skipped).

import {
  createGL, createProgram, createTarget,
  bindTarget, bindTextureUnit, drawQuad,
} from "./glutil.js";
import { makeNoise } from "./noise.js";
import { makeRng } from "./rng.js";

// The LIC fragment shader lives in lic-grid.frag (canonical source). Browsers
// can't `import` a .frag as a string without a build step, so we fetch it once
// at runtime (relative to this module) and compile lazily on first render. This
// keeps init() synchronous and the shader in a real, lintable .frag file.
const LIC_GRID_FRAG_URL = new URL("./lic-grid.frag", import.meta.url);
let _licGridSrc = null;

async function getLicGridProgram(ctx) {
  if (ctx.programs.licGrid) return ctx.programs.licGrid;
  if (_licGridSrc == null) {
    const res = await fetch(LIC_GRID_FRAG_URL);
    if (!res.ok) throw new Error("Failed to fetch lic-grid.frag: " + res.status);
    _licGridSrc = await res.text();
  }
  ctx.programs.licGrid = createProgram(ctx.gl, _licGridSrc);
  return ctx.programs.licGrid;
}

// --- optional collaborator modules (loaded if present) ----------------------
let _fieldMod = null;   // { createFieldTexture }
let _colorMod = null;   // { applyColor }
let _modsTried = false;

async function loadOptionalModules() {
  if (_modsTried) return;
  _modsTried = true;
  try { _fieldMod = await import("./field.js"); } catch (e) { _fieldMod = null; }
  try { _colorMod = await import("./color.js"); } catch (e) { _colorMod = null; }
}

export const DEFAULTS = {
  field: "rotation", field_expr: "z**2", color: "pure_rgb",
  pixel_mode: "voronoi", noise: "white_rgb",
  steps: 30, step_size: null, kernel: "box", kernel_sigma: 0.3,
  boundary: "nearest", enhance: "stretch", gamma: 1.0,
  voronoi_cells: 800, pure_mode: "stochastic", pure_sharpen: 2.5,
  streamline_stride: 4, colormap: "viridis", sat: 0.9,
  noise_points: 200, noise_block: 8,
  width: 600, height: null, dpi: 96, seed: 42,
  mhd_sample: "MHDSloshing", mhd_vector: "vorticity", mhd_slice_axis: "y",
  mhd_resolution: 512, mhd_width_kpc: 500.0,
  wd_dataset: null, wd_resolution: 512, wd_slice_axis: "theta",
};

const KERNEL_ID = { box: 0, gaussian: 1, raised_cosine: 2 };
const BOUNDARY_ID = { nearest: 0, wrap: 1 };

export function init(canvas) {
  const gl = createGL(canvas);
  const ctx = { gl, canvas, programs: {}, targets: {} };
  // licGrid is compiled lazily on first render (fetched from lic-grid.frag).
  ctx.programs.fieldStub = createProgram(gl, FIELD_STUB_FRAG);
  ctx.programs.present = createProgram(gl, PRESENT_FRAG);
  return ctx;
}

function sizeOf(params) {
  const width = params.width | 0 || DEFAULTS.width;
  const height = (params.height | 0) || width;
  return { width, height };
}

// PUBLIC API ----------------------------------------------------------------

export async function renderGrid(ctx, params) {
  const p = { ...DEFAULTS, ...params };
  const { width, height } = sizeOf(p);
  resizeCanvas(ctx, width, height);
  await loadOptionalModules();

  const { gl } = ctx;

  // 1) FIELD (Agent B) ---------------------------------------------------
  //    rg32f unit-vector texture. Fall back to internal rotation stub.
  let field;
  if (_fieldMod && typeof _fieldMod.createFieldTexture === "function") {
    field = await _fieldMod.createFieldTexture(gl, p); // { texture, w, h }
  } else {
    field = makeStubField(ctx, width, height, p.field);
  }

  // 2) NOISE (Agent A) ---------------------------------------------------
  const rng = makeRng(p.seed);
  const noiseKind = normaliseNoiseKind(p);
  const noise = makeNoise(gl, noiseKind, width, height, rng);

  // 3) LIC (Agent A) -----------------------------------------------------
  const lic = await licGridPass(ctx, field, noise, width, height, p);

  // 4) COLOR (Agent C) → canvas -----------------------------------------
  if (_colorMod && typeof _colorMod.applyColor === "function") {
    // Real color stage renders RGBA8 to the canvas (or a target). It owns the
    // final Y-flip-on-present per CONTRACT. It is async (fetches color.glsl +
    // colormap LUTs before drawing) — MUST be awaited so the draw completes
    // before renderGrid resolves (else exportPNG/screenshots catch a blank).
    await _colorMod.applyColor(gl, ctx, lic, field, p);
  } else {
    presentColorStub(ctx, lic);
  }
}

export async function renderVoronoi(ctx, params) {
  const p = { ...DEFAULTS, ...params };
  const { width, height } = sizeOf(p);
  resizeCanvas(ctx, width, height);
  await loadOptionalModules();
  // Agent E owns voronoi.js. Delegate if present; otherwise fall back to grid.
  try {
    const vmod = await import("./voronoi.js");
    if (vmod && typeof vmod.renderVoronoi === "function") {
      return vmod.renderVoronoi(ctx.gl, ctx, p);
    }
  } catch (e) { /* not available yet */ }
  return renderGrid(ctx, params);
}

export function exportPNG(ctx) {
  return new Promise((resolve) => ctx.canvas.toBlob(resolve, "image/png"));
}

// INTERNAL ------------------------------------------------------------------

function resizeCanvas(ctx, w, h) {
  if (ctx.canvas.width !== w || ctx.canvas.height !== h) {
    ctx.canvas.width = w;
    ctx.canvas.height = h;
  }
}

// pure_rgb color implies pure_rgb noise; otherwise use the requested noise kind.
// We only generate the three grid-mode noise kinds here (white/white_rgb/
// pure_rgb); other kinds from the Python zoo degrade to white_rgb for now.
function normaliseNoiseKind(p) {
  if (p.color === "pure_rgb") return "pure_rgb";
  if (p.noise === "white" || p.noise === "white_rgb" || p.noise === "pure_rgb") {
    return p.noise;
  }
  return "white_rgb";
}

// Allocate (or reuse) a float target of the given size/kind.
function getTarget(ctx, key, w, h, kind) {
  const t = ctx.targets[key];
  if (t && t.w === w && t.h === h && t.kind === kind) return t;
  const nt = createTarget(ctx.gl, w, h, kind);
  ctx.targets[key] = nt;
  return nt;
}

// --- LIC grid pass ---------------------------------------------------------
async function licGridPass(ctx, field, noise, w, h, p) {
  const { gl } = ctx;
  const stepSize = (p.step_size != null) ? p.step_size : (1.0 / Math.max(w, h));
  const steps = Math.max(0, Math.min(64, p.steps | 0));

  const lic = getTarget(ctx, "lic", w, h, "rgba32f");
  const prog = await getLicGridProgram(ctx);

  bindTarget(gl, lic);
  gl.useProgram(prog);
  bindTextureUnit(gl, prog, "uField", field.texture, 0);
  bindTextureUnit(gl, prog, "uNoise", noise.texture, 1);
  gl.uniform2f(gl.getUniformLocation(prog, "uResolution"), w, h);
  gl.uniform1i(gl.getUniformLocation(prog, "uSteps"), steps);
  gl.uniform1f(gl.getUniformLocation(prog, "uStepSize"), stepSize);
  gl.uniform1i(gl.getUniformLocation(prog, "uBoundary"), BOUNDARY_ID[p.boundary] ?? 0);
  gl.uniform1i(gl.getUniformLocation(prog, "uKernel"), KERNEL_ID[p.kernel] ?? 0);
  gl.uniform1f(gl.getUniformLocation(prog, "uKernelSigma"), p.kernel_sigma);
  drawQuad(gl);

  return lic; // { texture, fbo, w, h, kind }
}

// --- internal field stub (rotation) ---------------------------------------
// Produces an rg32f unit-vector field so the grid path is testable before
// Agent B's field.js exists. Matches lic_fields "rotation": v = (-(y-0.5), x-0.5)
// normalised. Returns { texture, w, h } like createFieldTexture.
function makeStubField(ctx, w, h, fieldName) {
  const { gl } = ctx;
  const tgt = getTarget(ctx, "fieldStub", w, h, "rg32f");
  const prog = ctx.programs.fieldStub;
  bindTarget(gl, tgt);
  gl.useProgram(prog);
  drawQuad(gl);
  return { texture: tgt.texture, w, h };
}

// --- internal color stub: present LIC directly to canvas -------------------
// Direct-RGB pass with the final Y-flip-on-present (CONTRACT). Used only when
// the real color.js is absent.
function presentColorStub(ctx, lic) {
  const { gl } = ctx;
  bindTarget(gl, null);
  const prog = ctx.programs.present;
  gl.useProgram(prog);
  bindTextureUnit(gl, prog, "uTex", lic.texture, 0);
  drawQuad(gl);
}

// --- shader sources for the internal stubs ---------------------------------

// rotation field, rg32f, encoded enc = v*0.5+0.5 (CONTRACT). vUv is bottom-left;
// we author in top-left so the LIC sampler (which flips Y) reads it correctly.
const FIELD_STUB_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec2 fragColor;
void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y); // top-left origin authoring
  vec2 c = uv - 0.5;
  vec2 v = vec2(-c.y, c.x);           // rotation: counter-clockwise swirl
  float m = length(v);
  v = (m > 0.0) ? v / m : vec2(0.0);
  fragColor = v * 0.5 + 0.5;
}`;

// Direct present: sample the float LIC texture and write RGBA8 to the canvas,
// flipping Y so row 0 ends up at the top (CONTRACT final-present rule).
const PRESENT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y); // flip Y on final present
  vec3 c = texture(uTex, uv).rgb;
  fragColor = vec4(c, 1.0);
}`;
