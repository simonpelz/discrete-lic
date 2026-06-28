#version 300 es
// color.glsl — final color stage for the in-browser LIC pipeline (Agent C).
//
// Mirrors src/lic_color.py. One fragment shader, branched by uMode:
//   0 rgb          : clamp LIC rgb -> 8-bit            (rgb_lic_to_image)
//   1 pure_rgb     : stochastic / streamline pure primary
//                    (pure_rgb_stochastic / pure_rgb_streamline)
//   2 angle_hsv    : hue from field angle, value from LIC gray (colorize_by_angle)
//   3 colormap     : grayscale LIC -> 256x1 LUT texture (apply_colormap)
//
// `enhance` (stretch/gamma/none) is applied to the LIC sample FIRST, exactly as
// enhance_contrast() runs before the color strategy in the Python pipeline.
// Stretch uses per-channel min/max passed in as uniforms (CPU readback in JS),
// matching numpy's global per-channel (arr.min()/arr.max()) stretch.
//
// Conventions (see app/CONTRACT.md):
//   * Field texture rg32f stores unit vectors encoded enc = v*0.5 + 0.5.
//   * vUv in [0,1], origin bottom-left in GL; the field/LIC textures share this
//     framebuffer orientation (both rendered by earlier passes), so we sample
//     them with vUv directly. The pipeline performs the single top-left Y-flip
//     for final present elsewhere; here we just draw to the canvas as-is.

precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uLic;       // rgba32f / r32f LIC result in [0,1]
uniform sampler2D uField;     // rg32f unit field, encoded enc = v*0.5+0.5
uniform sampler2D uColormap;  // 256x1 RGB LUT (colormap mode)

uniform int   uMode;          // 0 rgb, 1 pure_rgb, 2 angle_hsv, 3 colormap
uniform int   uPureMode;      // 0 stochastic, 1 streamline (mode 1 only)
uniform int   uEnhance;       // 0 none, 1 stretch, 2 gamma
uniform vec3  uLicMin;        // per-channel min over LIC (stretch)
uniform vec3  uLicMax;        // per-channel max over LIC (stretch)
uniform float uGamma;         // gamma exponent (gamma enhance)
uniform float uSharpen;       // pure_rgb sharpness exponent
uniform float uSat;           // angle_hsv saturation
uniform float uSeed;          // stochastic RNG seed
uniform vec2  uTexel;         // 1.0 / vec2(W, H)  (streamline trace step basis)
uniform int   uSteps;         // streamline trace length (num_steps)
uniform float uStepPx;        // streamline pixel step = step_size * max(W,H)

const float PI = 3.14159265358979323846;

// --- enhance_contrast() applied to a single LIC sample --------------------
vec3 enhance(vec3 c) {
  if (uEnhance == 1) {            // stretch (per channel)
    vec3 lo = uLicMin;
    vec3 hi = uLicMax;
    vec3 range = hi - lo;
    // hi > lo per channel -> rescale; else pass channel through (numpy behaviour)
    vec3 stretched = (c - lo) / max(range, vec3(1e-20));
    vec3 pass = c;
    vec3 outc;
    outc.r = range.r > 0.0 ? clamp(stretched.r, 0.0, 1.0) : pass.r;
    outc.g = range.g > 0.0 ? clamp(stretched.g, 0.0, 1.0) : pass.g;
    outc.b = range.b > 0.0 ? clamp(stretched.b, 0.0, 1.0) : pass.b;
    return outc;
  } else if (uEnhance == 2) {     // gamma
    return clamp(pow(max(c, vec3(0.0)), vec3(uGamma)), 0.0, 1.0);
  }
  return c;                       // none
}

// --- HSV -> RGB, matching _hsv_to_rgb_vec (h,s,v in [0,1]) -----------------
vec3 hsv_to_rgb(float h, float s, float v) {
  float h6 = h * 6.0;
  // i = floor(h6) mod 6 ; numpy uses int(h6) % 6 (h>=0 so int == floor)
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
  return vec3(v, p, q);           // i == 5
}

// --- per-pixel hash RNG in [0,1) (numpy-exact NOT required) ----------------
float hash21(vec2 p) {
  // Cheap, well-distributed hash on fragment coordinate + seed.
  p = fract(p * vec2(0.1031, 0.11369));
  p += dot(p, p.yx + 19.19);
  return fract((p.x + p.y) * p.x);
}

// Decode a unit field vector at uv.
vec2 fieldVec(vec2 uv) {
  return texture(uField, uv).xy * 2.0 - 1.0;
}

// Categorical sample from a 3-way probability distribution (pure_rgb).
vec3 pure_pick(vec3 lic) {
  vec3 v = max(lic, vec3(0.0));
  float total = v.r + v.g + v.b;
  vec3 probs = total > 1e-8 ? v / total : vec3(1.0 / 3.0);
  if (uSharpen != 1.0) probs = pow(probs, vec3(uSharpen));
  float s = probs.r + probs.g + probs.b;
  probs = s > 0.0 ? probs / s : vec3(1.0 / 3.0);

  float u = hash21(gl_FragCoord.xy + vec2(uSeed, uSeed * 1.7));
  // choices = sum(u >= cumprobs): u<c0 ->R, u<c0+c1 ->G, else B
  float c0 = probs.r;
  float c1 = probs.r + probs.g;
  if (u < c0) return vec3(1.0, 0.0, 0.0);
  if (u < c1) return vec3(0.0, 1.0, 0.0);
  return vec3(0.0, 0.0, 1.0);
}

// Streamline approximation: trace a short forward streamline from this pixel,
// accumulate the raw LIC distribution along it, paint the dominant primary.
// (pure_rgb does NOT apply enhance in the Python pipeline — see lic_main.py.)
// See color.js header for how this deviates from pure_rgb_streamline.
vec3 streamline_pick(vec2 uv0) {
  vec3 accum = texture(uLic, uv0).rgb;
  vec2 uv = uv0;
  // constant loop bound for GLSL ES 3.00; clamp by uSteps at runtime.
  const int MAX_STEPS = 64;
  for (int k = 1; k <= MAX_STEPS; k++) {
    if (k > uSteps) break;
    vec2 v = fieldVec(uv);
    uv = clamp(uv + uStepPx * uTexel * v, vec2(0.0), vec2(1.0));
    accum += texture(uLic, uv).rgb;
  }
  // argmax over the three primaries
  if (accum.r >= accum.g && accum.r >= accum.b) return vec3(1.0, 0.0, 0.0);
  if (accum.g >= accum.b) return vec3(0.0, 1.0, 0.0);
  return vec3(0.0, 0.0, 1.0);
}

void main() {
  vec4 licRaw = texture(uLic, vUv);
  vec3 outc;

  if (uMode == 1) {                           // pure_rgb (NO enhance — matches Python)
    if (uPureMode == 1) outc = streamline_pick(vUv);
    else                outc = pure_pick(licRaw.rgb);
    fragColor = vec4(outc, 1.0);
    return;
  }

  // rgb / angle_hsv / colormap all run enhance_contrast first.
  vec3 lic = enhance(licRaw.rgb);

  if (uMode == 0) {                          // rgb
    outc = clamp(lic, 0.0, 1.0);
  } else if (uMode == 2) {                    // angle_hsv
    vec2 v = fieldVec(vUv);
    float angle = atan(v.y, v.x);             // [-PI, PI]
    float hue = (angle + PI) / (2.0 * PI);    // [0, 1]
    float value = clamp(lic.r, 0.0, 1.0);     // LIC gray (R channel)
    outc = hsv_to_rgb(hue, uSat, value);
  } else {                                    // colormap
    float g = clamp(lic.r, 0.0, 1.0);         // grayscale LIC
    // 256x1 LUT; sample at texel centres to mirror cmap(g) over 256 entries.
    float x = (g * 255.0 + 0.5) / 256.0;
    outc = texture(uColormap, vec2(x, 0.5)).rgb;
  }

  fragColor = vec4(outc, 1.0);
}
