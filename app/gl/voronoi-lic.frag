#version 300 es
// voronoi-lic.frag — per-cell Line Integral Convolution (compute_voronoi_lic).
//
// OWNED BY AGENT E. One fragment == one Voronoi cell. The cell texels are
// packed into a PACKW x PACKH grid (PACKW*PACKH >= N); a fragment's linear cell
// index is  cell = iy*PACKW + ix. Inactive padding texels (cell >= N) output 0.
//
// Each cell traces a streamline from its seed position (sx, sy) in [0,1]
// (col-frac, row-frac), forward and backward along the unit field, taking
// num_steps Euler steps each way (plus the centre sample). At every step the
// position is converted to a pixel and looked up in the cell-map (uCellMap,
// R32UI, index+1) to fetch THAT cell's noise from uNoise — piecewise-constant,
// no interpolation, exactly like the numpy oracle. Kernel-weighted accumulate,
// divide by total weight.
//
// FIELD orientation: uField is authored top-left (row 0 = image top); we sample
// it at vec2(cx, 1-cy) per CONTRACT, decode v = tex*2-1. The cell map / seeds /
// noise are all in the bottom-origin frame established by the JFA pass, so cy
// here is row-frac from the BOTTOM; we convert to top-frac only when reading the
// top-left-authored field. Output of this pass carries the same bottom-origin
// indexing; the final paint flips Y once.

precision highp float;
precision highp int;
precision highp usampler2D;

uniform sampler2D  uField;    // rg32f, unit vectors, top-left authored
uniform sampler2D  uSeeds;    // rg32f Nx1: seed positions (cx, cy)
uniform usampler2D uCellMap;  // R32UI W x H: nearest-seed index+1 per pixel
uniform sampler2D  uNoise;    // rgba32f Nx1: per-cell noise (one-hot / white / ...)

uniform vec2  uResolution;    // (W, H) of the cell map / field
uniform ivec2 uPack;          // (PACKW, PACKH)
uniform int   uNumSeeds;      // N
uniform int   uSteps;         // num_steps (each direction)
uniform float uStepSize;      // Euler step in [0,1]
uniform int   uKernel;        // 0=box, 1=gaussian, 2=raised_cosine
uniform float uKernelSigma;   // gaussian sigma
uniform int   uBoundary;      // 0=nearest(clamp), 1=wrap

const int MAX_STEPS = 64;

out vec4 outLic;

// Kernel weight at signed offset i (i in [-uSteps, uSteps]); mirrors
// _make_kernel(2*steps+1) where t = linspace(-1,1,n) and centre index = steps.
float kernelWeight(int i) {
  if (uKernel == 0) return 1.0;                       // box
  float n = float(2 * uSteps + 1);
  // t = -1 + 2*(idx)/(n-1), idx = i + uSteps
  float t = (n > 1.0) ? (-1.0 + 2.0 * float(i + uSteps) / (n - 1.0)) : 0.0;
  if (uKernel == 1) {                                  // gaussian
    return exp(-(t * t) / (2.0 * uKernelSigma * uKernelSigma));
  }
  return 0.5 * (1.0 + cos(3.14159265358979 * t));      // raised_cosine
}

// Read the unit field at normalised (cx, cy) [row-frac from BOTTOM]. The field
// asset is authored top-left, so flip Y when sampling it. Nearest lookup (the
// oracle uses vx[row,col] directly, no interpolation).
vec2 fieldAt(vec2 p) {
  vec2 uv = vec2(p.x, 1.0 - p.y);
  return texture(uField, uv).xy * 2.0 - 1.0;
}

// Cell index (0-based) at normalised (cx, cy). Mirrors the oracle's
// row=clip(py*H), col=clip(px*W) nearest lookup. Returns -1 if empty.
int cellAt(vec2 p) {
  int col = int(p.x * uResolution.x);
  int row = int(p.y * uResolution.y);
  col = clamp(col, 0, int(uResolution.x) - 1);
  row = clamp(row, 0, int(uResolution.y) - 1);
  uint id = texelFetch(uCellMap, ivec2(col, row), 0).r;
  return int(id) - 1;
}

vec4 cellNoise(int cell) {
  if (cell < 0) return vec4(0.0);
  return texelFetch(uNoise, ivec2(cell, 0), 0);
}

vec2 boundPos(vec2 p) {
  if (uBoundary == 1) return fract(p);   // wrap
  return clamp(p, 0.0, 1.0);             // nearest
}

void main() {
  ivec2 ip = ivec2(gl_FragCoord.xy);
  int cell = ip.y * uPack.x + ip.x;
  if (cell >= uNumSeeds) { outLic = vec4(0.0); return; }

  vec2 seed = texelFetch(uSeeds, ivec2(cell, 0), 0).xy;

  float totalW = 0.0;
  vec4 accum = vec4(0.0);

  // Centre sample (weight at i=0).
  float wc = kernelWeight(0);
  accum += wc * cellNoise(cellAt(seed));
  totalW += wc;

  // Forward pass (+field).
  vec2 cx = seed;
  for (int i = 1; i <= MAX_STEPS; i++) {
    if (i > uSteps) break;
    vec2 v = fieldAt(cx);
    cx = boundPos(cx + uStepSize * v);
    float w = kernelWeight(i);
    accum += w * cellNoise(cellAt(cx));
    totalW += w;
  }

  // Backward pass (-field).
  cx = seed;
  for (int i = 1; i <= MAX_STEPS; i++) {
    if (i > uSteps) break;
    vec2 v = fieldAt(cx);
    cx = boundPos(cx - uStepSize * v);
    float w = kernelWeight(-i);
    accum += w * cellNoise(cellAt(cx));
    totalW += w;
  }

  outLic = (totalW > 0.0) ? (accum / totalW) : vec4(0.0);
}
