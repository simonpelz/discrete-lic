// field.js — build the rg32f UNIT vector-field texture for the LIC pipeline.
//
// OWNED BY AGENT B.
//
//   createFieldTexture(gl, params) -> Promise<{texture, w, h}>
//
// Pure-math fields (uniform/rotation/source/sink/saddle/shear/double_vortex/
// wave/perlin_curl/spiral/complex) are rendered by fields-eval.frag. The baked
// fields (mhd_cluster, wd_merger) load a precomputed asset written by
// scripts/export_field_assets.py and upload it as an rg32f texture.
//
// The texture stores UNIT vectors encoded enc = v*0.5+0.5 (decode v = tex*2-1),
// authored TOP-LEFT (texture row 0 = image top). The pipeline samples it at
// vec2(uv.x, 1-uv.y) per CONTRACT.md.

import {
  createProgram, createTexture, createTarget, bindTarget, drawQuad,
} from "./glutil.js";
import { makeRng } from "./rng.js";
import { exprToGlsl } from "./expr.js";

// Field name -> integer code understood by fields-eval.frag.
const FIELD_CODES = {
  uniform: 0, rotation: 1, source: 2, sink: 2, saddle: 3, shear: 4,
  double_vortex: 5, wave: 6, perlin_curl: 7, spiral: 8, complex: 9,
};

const BAKED_FIELDS = new Set(["mhd_cluster", "wd_merger"]);

// fields-eval.frag is fetched once and cached (with the complex expr injected).
let _fragText = null;
async function loadFrag() {
  if (_fragText !== null) return _fragText;
  const url = new URL("./fields-eval.frag", import.meta.url);
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch fields-eval.frag: " + res.status);
  _fragText = await res.text();
  return _fragText;
}

// Program cache keyed by the injected complex expression (most fields share the
// default expr, so this is usually a single program).
const _progCache = new Map();
async function getProgram(gl, complexGlsl) {
  if (_progCache.has(complexGlsl)) return _progCache.get(complexGlsl);
  const tmpl = await loadFrag();
  // NB: the token appears twice (a comment on ~line 19 and the real cexpr body),
  // so replace ALL occurrences — String.replace(string,…) would only do the
  // first (the comment), leaving the code token intact → '%' compile error.
  const src = tmpl.replaceAll("%%COMPLEX_EXPR%%", complexGlsl);
  const prog = createProgram(gl, src);
  _progCache.set(complexGlsl, prog);
  return prog;
}

function fieldResolution(params) {
  let res = Math.max((params.width | 0) || 0, 512);
  // In voronoi mode the JFA cell map is authored at field resolution, so it must
  // have enough pixels to resolve every seed — otherwise high cell counts
  // collapse into mush. Target ~16 px/cell (= sqrt(N)*4 per axis). Capped so a
  // huge slider value can't blow up memory/time.
  if (params.pixel_mode === "voronoi") {
    const n = (params.voronoi_cells | 0) || 0;
    res = Math.max(res, Math.ceil(Math.sqrt(n) * 4));
  }
  return Math.min(res, 4096);
}

// Mirror perlin_curl's per-octave random draws from lic_fields.py:
//   angles = rng.uniform(0, 2pi, octaves)   (drawn first)
//   phi    = rng.uniform(0, 2pi, octaves)   (drawn second)
// We can't reproduce numpy's bit stream, but we match the *structure* with a
// deterministic seeded PRNG (rng.js). Visual character, not bit-exactness.
function perlinOctaves(scale, octaves, seed) {
  const rng = makeRng((seed | 0) ^ 0x9e3779b9); // decorrelate from noise seed use
  const A = [], kx = [], ky = [], phi = [];
  const angles = [];
  for (let i = 0; i < octaves; i++) angles.push(rng.random() * 2 * Math.PI);
  for (let i = 0; i < octaves; i++) phi.push(rng.random() * 2 * Math.PI);
  for (let i = 0; i < octaves; i++) {
    A.push(1.0 / (i + 1));
    const freq = scale * Math.pow(2.0, i);
    kx.push(freq * Math.cos(angles[i]));
    ky.push(freq * Math.sin(angles[i]));
  }
  return { A, kx, ky, phi };
}

function setUniform2f(gl, prog, name, x, y) {
  const l = gl.getUniformLocation(prog, name);
  if (l) gl.uniform2f(l, x, y);
}
function setUniform1f(gl, prog, name, v) {
  const l = gl.getUniformLocation(prog, name);
  if (l) gl.uniform1f(l, v);
}
function setUniform1i(gl, prog, name, v) {
  const l = gl.getUniformLocation(prog, name);
  if (l) gl.uniform1i(l, v);
}
function setUniform1fv(gl, prog, name, arr) {
  const l = gl.getUniformLocation(prog, name);
  if (l) gl.uniform1fv(l, arr);
}

// ---- baked-asset loading ---------------------------------------------------

const ASSET_CANDIDATES = {
  mhd_cluster: "mhd_cluster",
  wd_merger: "wd_merger",
};

// Try to fetch a baked asset (.json sidecar + .bin). Returns null if absent.
async function tryLoadBaked(name) {
  const base = ASSET_CANDIDATES[name];
  if (!base) return null;
  const jsonUrl = new URL(`../assets/fields/${base}.json`, import.meta.url);
  let meta;
  try {
    const r = await fetch(jsonUrl);
    if (!r.ok) return null;
    meta = await r.json();
  } catch (_) {
    return null;
  }
  const binUrl = new URL(`../assets/fields/${meta.bin || base + ".bin"}`, import.meta.url);
  let buf;
  try {
    const r = await fetch(binUrl);
    if (!r.ok) return null;
    buf = await r.arrayBuffer();
  } catch (_) {
    return null;
  }
  return { meta, buf };
}

// Convert a baked buffer to a Float32Array of UNIT-encoded RG in [0,1].
// The asset stores the UNIT field (already normalised). Encoding may be:
//   "unorm8"  : 2 bytes/texel, value/255 -> [0,1] is already enc = v*0.5+0.5
//   "float16" : 2 halfs/texel, raw unit components in [-1,1] -> encode here
//   "float32" : 2 floats/texel, raw unit components in [-1,1] -> encode here
function decodeBaked(meta, buf) {
  const { resolution } = meta;
  const w = meta.width || resolution;
  const h = meta.height || resolution;
  const n = w * h;
  const out = new Float32Array(n * 2);
  const enc = meta.encoding || "unorm8";
  if (enc === "unorm8") {
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < n; i++) {
      // stored as enc = v*0.5+0.5 quantised to [0,255]
      out[2 * i] = u8[2 * i] / 255.0;
      out[2 * i + 1] = u8[2 * i + 1] / 255.0;
    }
  } else if (enc === "float16") {
    const u16 = new Uint16Array(buf);
    for (let i = 0; i < n; i++) {
      const vx = halfToFloat(u16[2 * i]);
      const vy = halfToFloat(u16[2 * i + 1]);
      out[2 * i] = vx * 0.5 + 0.5;
      out[2 * i + 1] = vy * 0.5 + 0.5;
    }
  } else if (enc === "float32") {
    const f32 = new Float32Array(buf);
    for (let i = 0; i < n; i++) {
      out[2 * i] = f32[2 * i] * 0.5 + 0.5;
      out[2 * i + 1] = f32[2 * i + 1] * 0.5 + 0.5;
    }
  } else {
    throw new Error("Unknown baked encoding: " + enc);
  }
  return { data: out, w, h };
}

function halfToFloat(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

// ---- public API ------------------------------------------------------------

export async function createFieldTexture(gl, params) {
  const fieldName = params.field || "rotation";

  if (BAKED_FIELDS.has(fieldName)) {
    const baked = await tryLoadBaked(fieldName);
    if (baked) {
      const { data, w, h } = decodeBaked(baked.meta, baked.buf);
      // The asset is authored top-left (row 0 = top). The texture's row 0 ends
      // up at GL bottom; the pipeline samples baked assets at vec2(uv.x,1-uv.y),
      // so we upload rows as-is (top row first in the buffer).
      const texture = createTexture(gl, w, h, "rg32f", data, gl.NEAREST);
      return { texture, w, h };
    }
    // No asset present: fall through to a synthetic stand-in so the pipeline
    // still runs. perlin_curl gives turbulent structure reminiscent of the MHD
    // vorticity field. (Integration should swap in the real bake.)
    console.warn(
      `[field.js] baked asset for '${fieldName}' not found; rendering a ` +
      `synthetic perlin_curl placeholder. Run scripts/export_field_assets.py.`
    );
    return renderPureMath(gl, { ...params, field: "perlin_curl", seed: 17, scale: 4.0, octaves: 5 });
  }

  return renderPureMath(gl, params);
}

async function renderPureMath(gl, params) {
  const fieldName = params.field || "rotation";
  const code = FIELD_CODES[fieldName] !== undefined ? FIELD_CODES[fieldName] : 1;
  const res = fieldResolution(params);

  // Resolve complex expression -> GLSL (with safe fallback inside).
  const complexGlsl = (code === 9)
    ? exprToGlsl(params.field_expr || "z**2").glsl
    : "z"; // harmless default when not the complex field

  const prog = await getProgram(gl, complexGlsl);
  const target = createTarget(gl, res, res, "rg32f", gl.NEAREST);

  gl.useProgram(prog);
  setUniform1i(gl, prog, "uField", code);

  // ---- per-field uniforms (defaults mirror lic_fields.py factories) --------
  // uniform: angle 0
  setUniform1f(gl, prog, "uAngle", deg2rad(params.angle_deg ?? 0));
  // rotation: centre (0.5,0.5), ccw (sign +1)
  setUniform2f(gl, prog, "uC1", params.cx ?? 0.5, params.cy ?? 0.5);
  setUniform1f(gl, prog, "uSign", params.clockwise ? -1.0 : 1.0);
  // source/sink: sink = strength -1
  let strength = params.strength ?? 1.0;
  if (fieldName === "sink") strength = -1.0;
  setUniform1f(gl, prog, "uStrength", strength);
  // shear: horizontal default
  setUniform1i(gl, prog, "uShearVert", (params.shear_direction === "vertical") ? 1 : 0);
  // double_vortex: centres (0.3,0.5) & (0.7,0.5), s1=+1, s2=-1
  if (fieldName === "double_vortex") {
    setUniform2f(gl, prog, "uC1", params.x1 ?? 0.3, params.y1 ?? 0.5);
  }
  setUniform2f(gl, prog, "uC2", params.x2 ?? 0.7, params.y2 ?? 0.5);
  setUniform1f(gl, prog, "uS1", params.s1 ?? 1.0);
  setUniform1f(gl, prog, "uS2", params.s2 ?? -1.0);
  // wave: kx=3, ky=2, phase 0
  setUniform2f(gl, prog, "uWaveK", params.kx ?? 3.0, params.ky ?? 2.0);
  setUniform1f(gl, prog, "uWavePhase", params.wave_phase ?? 0.0);
  // spiral: inward 0.5, rot 1.0
  setUniform1f(gl, prog, "uInward", params.inward ?? 0.5);
  setUniform1f(gl, prog, "uRot", params.rot ?? 1.0);

  // perlin_curl octaves
  const octaves = Math.min(params.octaves ?? 4, 8);
  const oc = perlinOctaves(params.scale ?? 4.0, octaves, params.seed ?? 0);
  setUniform1i(gl, prog, "uOctaves", octaves);
  // pad to MAX_OCTAVES (8) for the uniform array uploads
  setUniform1fv(gl, prog, "uAmp", padTo(oc.A, 8));
  setUniform1fv(gl, prog, "uKx", padTo(oc.kx, 8));
  setUniform1fv(gl, prog, "uKy", padTo(oc.ky, 8));
  setUniform1fv(gl, prog, "uPhi", padTo(oc.phi, 8));

  bindTarget(gl, target);
  drawQuad(gl);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  return { texture: target.texture, w: res, h: res };
}

function padTo(arr, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < Math.min(arr.length, n); i++) out[i] = arr[i];
  return out;
}

function deg2rad(d) { return (d * Math.PI) / 180.0; }
