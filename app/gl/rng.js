// rng.js — seeded PRNG for deterministic noise generation.
//
// OWNED BY AGENT A. Bit-exact parity with numpy's Generator is NOT required;
// we only need determinism (same seed → same stream) and a well-distributed
// uniform [0,1). Uses splitmix32 to derive a state from the seed, then
// mulberry32 as the stepping function — both are fast, dependency-free, and
// pass basic distribution sanity checks.
//
// makeRng(seed) -> {
//   random()      -> float in [0, 1)
//   randint(n)    -> integer in [0, n)   (n > 0)
// }
// The returned object is also callable shorthand via .random().

function splitmix32(a) {
  // Mixes a 32-bit seed into a better-distributed 32-bit value.
  a |= 0;
  a = (a + 0x9e3779b9) | 0;
  let t = a ^ (a >>> 16);
  t = Math.imul(t, 0x21f0aaad);
  t = t ^ (t >>> 15);
  t = Math.imul(t, 0x735a2d97);
  return (t ^ (t >>> 15)) >>> 0;
}

export function makeRng(seed) {
  // Derive an initial 32-bit state from the seed (coerce to uint32).
  let s = splitmix32((seed | 0) >>> 0 || 1);

  function random() {
    // mulberry32 step.
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function randint(n) {
    // Unbiased-enough integer in [0, n). n is small here (e.g. 3 for primaries).
    return Math.floor(random() * n);
  }

  return { random, randint };
}
