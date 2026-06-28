#version 300 es
// lic-grid.frag — Line Integral Convolution, grid mode.
//
// OWNED BY AGENT A. Faithful GLSL port of compute_lic() in src/lic_core.py.
//
// Per output pixel we integrate a streamline both forward and backward along the
// UNIT vector field (sampled from uField), accumulating kernel-weighted samples
// of the noise texture (uNoise), then divide by the total kernel weight.
//
//   total samples = 2*steps + 1   (center + forward[1..steps] + backward[1..steps])
//   Euler step:    c += step_size * v   (v = unit field at c)
//   weight(center) = w[steps]
//   weight(fwd i)  = w[steps + i]
//   weight(bwd i)  = w[steps - i]
//
// Kernels (see _make_kernel), with t in [-1,1] across the full streamline:
//   box           : 1
//   gaussian      : exp(-t^2 / (2 sigma^2))
//   raised_cosine : 0.5 * (1 + cos(pi t))
//
// COORDINATE SPACE
// ----------------
// We integrate in TOP-LEFT origin uv space (cy=0 at top), matching Python's
// cx/cy. The noise texture is authored top-left and sampled the same way the
// Python code indexes rows (row = cy*(H-1)), so we read it at uv directly with
// an explicit Y handling below. The field texture is a GL (bottom-left) texture
// authored top-left, so per CONTRACT we sample it at (x, 1-y).
//
// We do our own bilinear sampling against (W-1)/(H-1) coordinates to mirror
// scipy.map_coordinates(order=1) exactly (texelFetch + manual lerp), so NEAREST
// filtering on the source textures is fine.

precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uField;   // rg32f, enc = v*0.5+0.5 ; decode v = enc*2-1
uniform sampler2D uNoise;   // rgba32f, values in [0,1]
uniform vec2  uResolution;  // (W, H)
uniform int   uSteps;       // num_steps per direction
uniform float uStepSize;    // Euler step in normalised [0,1] coords
uniform int   uBoundary;    // 0 = nearest (clamp), 1 = wrap
uniform int   uKernel;      // 0 = box, 1 = gaussian, 2 = raised_cosine
uniform float uKernelSigma; // sigma for gaussian

const int MAX_STEPS = 64;   // GLSL ES requires constant loop bounds
const float PI = 3.14159265358979323846;

// ---- boundary handling on normalised coords --------------------------------
vec2 applyBoundary(vec2 c) {
  if (uBoundary == 1) {
    // wrap: fract maps into [0,1)
    return fract(c);
  }
  return clamp(c, 0.0, 1.0);
}

// ---- bilinear sample of a generic texture using (N-1) coords ---------------
// Mirrors scipy.map_coordinates(order=1): position = c*(N-1), then lerp the
// four neighbouring texels. `c` is a top-left-origin normalised coord in [0,1].
// `flipY` selects whether the texture is stored bottom-left (field asset) and
// must be read at (x, 1-y).
vec4 bilinearTopLeft(sampler2D tex, vec2 c, vec2 res, bool flipY) {
  vec2 dim = res - 1.0;
  vec2 pos = c * dim;                 // fractional texel position, top-left origin
  vec2 p0 = floor(pos);
  vec2 f  = pos - p0;
  vec2 p1 = min(p0 + 1.0, dim);
  p0 = max(p0, vec2(0.0));

  // Convert top-left integer rows to texel fetch coords. For a bottom-left GL
  // texture authored top-left, row r (from top) lives at texel (H-1 - r).
  ivec2 maxi = ivec2(res) - 1;
  int x0 = int(p0.x), x1 = int(p1.x);
  int y0t = int(p0.y), y1t = int(p1.y);
  int y0 = y0t, y1 = y1t;
  if (flipY) {
    y0 = maxi.y - y0t;
    y1 = maxi.y - y1t;
  }
  vec4 t00 = texelFetch(tex, ivec2(x0, y0), 0);
  vec4 t10 = texelFetch(tex, ivec2(x1, y0), 0);
  vec4 t01 = texelFetch(tex, ivec2(x0, y1), 0);
  vec4 t11 = texelFetch(tex, ivec2(x1, y1), 0);
  vec4 top = mix(t00, t10, f.x);
  vec4 bot = mix(t01, t11, f.x);
  return mix(top, bot, f.y);
}

// ---- sample the unit field at normalised coord c (top-left origin) ---------
vec2 sampleField(vec2 c) {
  vec4 enc = bilinearTopLeft(uField, c, uResolution, true);
  return enc.xy * 2.0 - 1.0;
}

// ---- sample noise (all 4 channels) at normalised coord c -------------------
// The noise texture is uploaded as a Float32Array authored top-left (row 0 at
// the start of the buffer). WebGL uploads row 0 to the BOTTOM of the texture,
// so it is effectively bottom-left and must be read flipped, same as the field.
vec4 sampleNoise(vec2 c) {
  return bilinearTopLeft(uNoise, c, uResolution, true);
}

// ---- kernel weight at parametric position t in [-1,1] ----------------------
float kernelWeight(float t) {
  if (uKernel == 1) {
    return exp(-(t * t) / (2.0 * uKernelSigma * uKernelSigma));
  } else if (uKernel == 2) {
    return 0.5 * (1.0 + cos(PI * t));
  }
  return 1.0; // box
}

void main() {
  // vUv is bottom-left origin (from VERT_QUAD). Convert to top-left uv so the
  // integration matches Python's row indexing; the pipeline flips Y again on
  // final present, so the net result is correct top-left imagery.
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y);

  // t spans [-1, 1] across the full streamline (2*steps+1 samples); centre at
  // index uSteps. Index i in [0, 2*steps] maps to t = (i - steps) / steps —
  // matching np.linspace(-1, 1, 2*steps+1) used by _make_kernel.
  float invSteps = (uSteps > 0) ? (1.0 / float(uSteps)) : 0.0;

  vec4 accum = vec4(0.0);
  float wsum = 0.0;

  // centre sample (t = 0)
  {
    float w = kernelWeight(0.0);
    accum += w * sampleNoise(uv);
    wsum += w;
  }

  // forward pass: step along +field
  vec2 c = uv;
  for (int i = 1; i <= MAX_STEPS; i++) {
    if (i > uSteps) break;
    vec2 v = sampleField(c);
    c = c + uStepSize * v;
    c = applyBoundary(c);
    float t = float(i) * invSteps;
    float w = kernelWeight(t);
    accum += w * sampleNoise(c);
    wsum += w;
  }

  // backward pass: step along -field
  c = uv;
  for (int i = 1; i <= MAX_STEPS; i++) {
    if (i > uSteps) break;
    vec2 v = sampleField(c);
    c = c - uStepSize * v;
    c = applyBoundary(c);
    float t = -float(i) * invSteps;
    float w = kernelWeight(t);
    accum += w * sampleNoise(c);
    wsum += w;
  }

  vec4 result = (wsum > 0.0) ? (accum / wsum) : accum;
  fragColor = vec4(result.rgb, 1.0);
}
