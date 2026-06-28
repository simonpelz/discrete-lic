// glutil.js — shared WebGL2 helpers for the in-browser LIC pipeline.
//
// OWNED BY STEP 0 (shared infrastructure). Treat as READ-ONLY: every module
// (grid LIC, fields, color, voronoi) imports these helpers. If you think you
// need to change a signature here, coordinate — don't fork it.
//
// Conventions
// -----------
// * WebGL2 only. Float render targets via EXT_color_buffer_float.
// * All passes are full-screen-quad fragment shaders. Vertex shader is fixed
//   (VERT_QUAD): it emits a varying `vUv` in [0,1], origin bottom-left in GL.
//   The pipeline flips Y on final present so row 0 = top (matches lic_core.py).
// * Textures are sampled with `vUv` in [0,1]. Field/noise/LIC intermediates are
//   float textures; the final color pass outputs RGBA8 to the canvas.

export const VERT_QUAD = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

export function createGL(canvas) {
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");
  // Needed to render into float textures (LIC accumulation, JFA, etc.)
  const extCBF = gl.getExtension("EXT_color_buffer_float");
  if (!extCBF) throw new Error("EXT_color_buffer_float unsupported — cannot render to float targets.");
  gl.getExtension("OES_texture_float_linear"); // optional: linear filtering on float textures
  return gl;
}

export function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("Shader compile error:\n" + log + "\n--- source ---\n" + numberLines(src));
  }
  return sh;
}

// Build a program from a fragment shader source, reusing VERT_QUAD.
export function createProgram(gl, fragSrc, vertSrc = VERT_QUAD) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.bindAttribLocation(prog, 0, "aPos");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("Program link error:\n" + gl.getProgramInfoLog(prog));
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}

// A single full-screen triangle-pair VAO, created once and reused.
let _quadVao = null;
export function fullscreenQuad(gl) {
  if (_quadVao) return _quadVao;
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  // two triangles covering clip space [-1,1]^2
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 1, -1, -1, 1,
    -1, 1, 1, -1, 1, 1,
  ]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  _quadVao = vao;
  return vao;
}

export function drawQuad(gl) {
  gl.bindVertexArray(fullscreenQuad(gl));
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}

// Internal-format presets. Use "rgba32f" for general float intermediates,
// "rg32f" for vector fields, "r32f" for scalar (grayscale LIC).
const FORMATS = {
  rgba32f: { internal: 0x8814 /*RGBA32F*/, format: 0x1908 /*RGBA*/, type: 0x1406 /*FLOAT*/, channels: 4 },
  rg32f:   { internal: 0x8230 /*RG32F*/,   format: 0x8227 /*RG*/,   type: 0x1406, channels: 2 },
  r32f:    { internal: 0x822E /*R32F*/,    format: 0x1903 /*RED*/,  type: 0x1406, channels: 1 },
  rgba8:   { internal: 0x8058 /*RGBA8*/,   format: 0x1908 /*RGBA*/, type: 0x1401 /*UNSIGNED_BYTE*/, channels: 4 },
};

// Create a texture. `data` may be a typed array (Float32Array / Uint8Array) or null.
export function createTexture(gl, w, h, kind = "rgba32f", data = null, filter = gl.NEAREST) {
  const f = FORMATS[kind];
  if (!f) throw new Error("Unknown texture kind: " + kind);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, f.internal, w, h, 0, f.format, f.type, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  tex._kind = kind; tex._w = w; tex._h = h;
  return tex;
}

// Wrap a texture in a framebuffer for render-to-texture.
export function createFBO(gl, texture) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error("Framebuffer incomplete: 0x" + status.toString(16));
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return fbo;
}

// Convenience: allocate matching {texture, fbo} target.
export function createTarget(gl, w, h, kind = "rgba32f", filter = gl.NEAREST) {
  const texture = createTexture(gl, w, h, kind, null, filter);
  const fbo = createFBO(gl, texture);
  return { texture, fbo, w, h, kind };
}

// Bind a target (or null for the canvas) and set the viewport.
export function bindTarget(gl, target) {
  if (target) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, target.w, target.h);
  } else {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
  }
}

// Bind a texture to a unit and set the sampler uniform by name.
export function bindTextureUnit(gl, prog, name, texture, unit) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  const loc = gl.getUniformLocation(prog, name);
  if (loc !== null) gl.uniform1i(loc, unit);
}

// Read back a float target into a Float32Array (used by enhance min/max,
// fidelity diffing, and Voronoi cell-index readback).
export function readFloatPixels(gl, target) {
  const f = FORMATS[target.kind];
  const out = new Float32Array(target.w * target.h * f.channels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
  gl.readPixels(0, 0, target.w, target.h, f.format, gl.FLOAT, out);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return out;
}

function numberLines(src) {
  return src.split("\n").map((l, i) => `${String(i + 1).padStart(3)}: ${l}`).join("\n");
}
