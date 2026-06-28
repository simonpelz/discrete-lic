#version 300 es
// jfa.frag — one Jump Flood Algorithm pass for nearest-seed Voronoi tessellation.
//
// OWNED BY AGENT E. Replaces scipy.cKDTree.build_cell_map: it produces, per
// pixel, the INDEX of the nearest Voronoi seed point.
//
// ENCODING
// --------
// The ping-pong "best seed" buffer is an integer texture (R32UI). Each texel
// stores the winning seed index + 1, so 0 is reserved for "no seed yet"
// (uninitialised / empty). The seed POSITIONS live in a separate Nx1 rg32f
// texture (uSeeds); during a jump we fetch the candidate's position from there
// and compare squared distance to this fragment's pixel-centre. This keeps the
// payload (an index, which can exceed 255 — so we MUST use an integer texture)
// independent of the metric (position), exactly as JFA requires.
//
// Coordinate convention: this pass runs with the fixed VERT_QUAD (vUv bottom-
// left). We author the cell map in PIXEL space and DO NOT flip here — the final
// paint pass owns the single net Y-flip. Pixel centre of this fragment:
//   pix = gl_FragCoord.xy  (already at +0.5 centre)
// normalised seed/pixel positions are (col-frac, row-frac) in [0,1], matching
// seeds[:,0]=cx, seeds[:,1]=cy. Here "row-frac" is measured from the BOTTOM
// (GL framebuffer order); seeds are uploaded in that same bottom-origin frame so
// the metric is self-consistent, and the paint pass flips once at the end.

precision highp float;
precision highp int;
precision highp usampler2D;

uniform usampler2D uPrev;   // R32UI: current best seed index+1 per pixel
uniform sampler2D  uSeeds;  // rg32f Nx1: seed positions (cx, cy) in [0,1]
uniform vec2  uResolution;  // (W, H) of the cell map
uniform int   uNumSeeds;    // N
uniform int   uStep;        // jump distance in pixels (W/2, W/4, ... 1)

out uint outId;             // winning seed index+1

vec2 seedPos(int idx) {
  // Fetch seed idx position from the Nx1 seed texture (texel centre).
  return texelFetch(uSeeds, ivec2(idx, 0), 0).xy;
}

void main() {
  ivec2 ip = ivec2(gl_FragCoord.xy);      // this pixel
  vec2 pixCentre = (vec2(ip) + 0.5) / uResolution;  // (col-frac, row-frac) bottom-origin

  uint bestId = 0u;            // 0 == none
  float bestD = 1.0e30;

  // Seed the search with our own current value so progress is monotone.
  uint cur = texelFetch(uPrev, ip, 0).r;
  if (cur != 0u) {
    vec2 sp = seedPos(int(cur) - 1);
    vec2 d = sp - pixCentre;
    bestD = dot(d, d);
    bestId = cur;
  }

  // Examine the 3x3 neighbourhood at the current jump distance.
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = ip + ivec2(dx, dy) * uStep;
      if (q.x < 0 || q.y < 0 ||
          q.x >= int(uResolution.x) || q.y >= int(uResolution.y)) continue;
      uint cand = texelFetch(uPrev, q, 0).r;
      if (cand == 0u) continue;
      vec2 sp = seedPos(int(cand) - 1);
      vec2 d = sp - pixCentre;
      float dist = dot(d, d);
      if (dist < bestD) {
        bestD = dist;
        bestId = cand;
      }
    }
  }

  outId = bestId;
}
