#version 300 es
// fields-eval.frag — evaluate the pure-math vector fields and write the
// encoded UNIT field into an rg32f texture.
//
// OWNED BY AGENT B. Ports lic_fields.py (uniform, rotation, source, sink,
// saddle, shear, double_vortex, wave, perlin_curl, spiral, complex).
//
// Encoding (per CONTRACT.md): we store unit vectors (vx, vy) as enc = v*0.5+0.5.
// Python evaluate_field() normalises to unit length; stagnation (zero) stays 0.
// Decode elsewhere: v = tex*2 - 1, so zero -> enc 0.5.
//
// Coordinate convention: Python uses x=(col+0.5)/W, y=(row+0.5)/H with row 0 at
// the TOP. The field texture here is authored TOP-LEFT (texture row 0 = top),
// so the pipeline samples it with vec2(uv.x, 1-uv.y) like the baked asset.
// GL renders at vUv (origin bottom-left); to make texture-row-0 = Python-row-0
// (top) we evaluate the field at py = vec2(vUv.x, 1.0 - vUv.y).
//
// The complex-field expression is injected by field.js by replacing the token
// %%COMPLEX_EXPR%% with a GLSL expression in terms of `z` (a vec2). When the
// field is not `complex`, the injected expression is harmless (e.g. `z`).

precision highp float;

in vec2 vUv;
out vec4 fragColor;

// Which field to evaluate (integer code, see field.js FIELD_CODES).
uniform int uField;

// Generic field parameters (centres, strengths, etc.). Defaults mirror
// lic_fields.py factory defaults; field.js sets them.
uniform vec2  uC1;        // centre 1 (cx, cy) — rotation/source/sink/saddle/spiral
uniform vec2  uC2;        // centre 2 — double_vortex second vortex
uniform float uAngle;     // uniform: direction in radians
uniform float uSign;      // rotation: +1 ccw, -1 cw
uniform float uStrength;  // source/sink strength
uniform float uS1;        // double_vortex s1 sign
uniform float uS2;        // double_vortex s2 sign
uniform int   uShearVert; // shear: 0 horizontal, 1 vertical
uniform vec2  uWaveK;     // wave: (kx, ky)
uniform float uWavePhase; // wave: phase
uniform float uInward;    // spiral inward
uniform float uRot;       // spiral rot

// perlin_curl octave data, precomputed JS-side (rng.js-style hashing) and passed
// in. Constant array bound for ES 3.00 loop validity; uOctaves selects active.
const int MAX_OCTAVES = 8;
uniform int   uOctaves;                 // active octave count
uniform float uAmp[MAX_OCTAVES];        // A_i = 1/(i+1)
uniform float uKx[MAX_OCTAVES];         // freq_i * cos(angle_i)
uniform float uKy[MAX_OCTAVES];         // freq_i * sin(angle_i)
uniform float uPhi[MAX_OCTAVES];        // phase_i

// ---- complex helpers (used by the injected expression) --------------------
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 cconj(vec2 a) { return vec2(a.x, -a.y); }
vec2 cdiv(vec2 a, vec2 b) {
  float d = b.x * b.x + b.y * b.y;
  if (d == 0.0) return vec2(0.0);
  return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}
vec2 cexp(vec2 a) { float e = exp(a.x); return vec2(e * cos(a.y), e * sin(a.y)); }
// principal log
vec2 clog(vec2 a) { return vec2(log(length(a) + 1e-30), atan(a.y, a.x)); }
// complex power via exp(b * log(a)); handles non-integer exponents like Python.
vec2 cpow(vec2 a, vec2 b) {
  if (a.x == 0.0 && a.y == 0.0) return vec2(0.0);
  return cexp(cmul(b, clog(a)));
}
vec2 csin(vec2 a) { return vec2(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y)); }
vec2 ccos(vec2 a) { return vec2(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y)); }

// Injected complex expression as a function of z (vec2). field.js replaces the
// token below. Default is identity-safe.
vec2 cexpr(vec2 z) { return %%COMPLEX_EXPR%%; }

// ---- field evaluators (return RAW, unnormalised vector) -------------------

vec2 fieldVec(vec2 p) {
  // p = (x, y) in [0,1], y increasing downward (Python convention).
  float x = p.x;
  float y = p.y;

  // 0 uniform
  if (uField == 0) return vec2(cos(uAngle), sin(uAngle));
  // 1 rotation:  (-sign*(y-cy),  sign*(x-cx))
  if (uField == 1) return vec2(-uSign * (y - uC1.y), uSign * (x - uC1.x));
  // 2 source/sink: strength*(x-cx, y-cy)
  if (uField == 2) return uStrength * (p - uC1);
  // 3 saddle: (x-cx, -(y-cy))
  if (uField == 3) return vec2(x - uC1.x, -(y - uC1.y));
  // 4 shear: horizontal (y,0) or vertical (0,x)
  if (uField == 4) return (uShearVert == 1) ? vec2(0.0, x) : vec2(y, 0.0);
  // 5 double_vortex: sum of two rotations (sign from s1/s2)
  if (uField == 5) {
    float sg1 = (uS1 < 0.0) ? -1.0 : 1.0;
    float sg2 = (uS2 < 0.0) ? -1.0 : 1.0;
    vec2 v1 = vec2(-sg1 * (y - uC1.y), sg1 * (x - uC1.x));
    vec2 v2 = vec2(-sg2 * (y - uC2.y), sg2 * (x - uC2.x));
    return v1 + v2;
  }
  // 6 wave: (sin(2pi*ky*y + phase), sin(2pi*kx*x))
  if (uField == 6) {
    return vec2(sin(6.2831853 * uWaveK.y * y + uWavePhase),
                sin(6.2831853 * uWaveK.x * x));
  }
  // 7 perlin_curl: curl of layered-sine potential
  if (uField == 7) {
    vec2 v = vec2(0.0);
    for (int i = 0; i < MAX_OCTAVES; i++) {
      if (i >= uOctaves) break;
      float ca = cos(uKx[i] * x + uKy[i] * y + uPhi[i]);
      v.x += uAmp[i] * uKy[i] * ca;   // dP/dy
      v.y -= uAmp[i] * uKx[i] * ca;   // -dP/dx
    }
    return v;
  }
  // 8 spiral: (-inward*dx - rot*dy, -inward*dy + rot*dx)
  if (uField == 8) {
    float dx = x - uC1.x;
    float dy = y - uC1.y;
    return vec2(-uInward * dx - uRot * dy, -uInward * dy + uRot * dx);
  }
  // 9 complex: z = x + i*y -> w; vx=Re(w), vy=Im(w)
  if (uField == 9) {
    vec2 w = cexpr(vec2(x, y));
    if (!all(equal(w, w))) return vec2(0.0); // NaN guard
    if (isinf(w.x) || isinf(w.y)) return vec2(0.0);
    return w;
  }
  return vec2(0.0);
}

void main() {
  // Author top-left: texture row 0 = Python row 0 (top). GL is bottom-left,
  // so Python-y = 1 - vUv.y.
  vec2 p = vec2(vUv.x, 1.0 - vUv.y);
  vec2 v = fieldVec(p);
  float m = length(v);
  vec2 unit = (m > 0.0) ? v / m : vec2(0.0);
  vec2 enc = unit * 0.5 + 0.5;
  fragColor = vec4(enc, 0.0, 1.0);
}
