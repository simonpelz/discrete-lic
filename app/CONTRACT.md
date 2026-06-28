# In-browser LIC — module contract (Step 0, frozen)

This is the interface every parallel workstream codes against. **Do not change a
signature here without coordinating.** Each agent owns disjoint files (below), so
parallel work won't collide. The only shared, already-written files are
`app/gl/glutil.js` and this contract — treat both as read-only.

Reference oracle for *behaviour* (match the visual character, not bit-exactness):
`../src/lic_core.py`, `../src/lic_voronoi.py`, `../src/lic_noise.py`,
`../src/lic_color.py`, `../src/lic_fields.py`.

## Coordinate & data conventions

- `uv` ∈ [0,1]², sampled in fragment shaders via the `vUv` varying.
- Image origin = top-left (row 0 = top), matching the Python pipeline. GL
  textures are bottom-left origin; the pipeline flips Y once on final present.
- **Field texture** (`rg32f`): stores **unit** vectors `(vx, vy)` encoded to
  [0,1] as `enc = v*0.5 + 0.5`. Decode in GLSL: `vec2 v = texture(uField, uv).xy*2.0 - 1.0;`
  Python `evaluate_field()` normalises to unit length and leaves stagnation
  points as zero — do the same (zero stays `enc = 0.5`).
- **Noise texture** (`rgba32f`): values in [0,1]. `white_rgb` = independent per
  channel. `pure_rgb` = one-hot primary in RGB (R=(1,0,0), etc.).
- **LIC texture** (`rgba32f` or `r32f`): LIC result in [0,1], same channel count
  as the noise it consumed.
- Y-flip rule for sampling a field/noise the way Python indexes rows: Python uses
  `row = y*(H-1)` with row 0 at top; in GL sample at `vec2(uv.x, 1.0-uv.y)` when
  reading assets authored top-left (the baked field asset; see field.js).

## `params` object (subset of web/server.py GenerateRequest)

Passed from the UI to `pipeline.renderGrid` / `renderVoronoi`. Names match the
existing `app.js`:

```
field, field_expr, color, pixel_mode, noise, steps, step_size|null,
kernel, kernel_sigma, boundary, enhance, gamma, voronoi_cells, pure_mode,
pure_sharpen, streamline_stride, colormap, sat, noise_points, noise_block,
width, height|null, dpi, seed,
mhd_sample, mhd_vector, mhd_slice_axis, mhd_resolution, mhd_width_kpc,
wd_dataset, wd_resolution, wd_slice_axis
```

Defaults live in `pipeline.DEFAULTS`. Height defaults to width when null.

## Module boundaries (ownership)

### Agent A — grid core (`pipeline.js`, `lic-grid.frag`, `noise.js`, `rng.js`)
- `pipeline.js` is the orchestrator and the ONLY file that imports the others.
  Public API (UI calls these):
  - `init(canvas) -> ctx` — creates GL context, compiles programs.
  - `renderGrid(ctx, params) -> void` — field→noise→LIC→color onto the canvas.
  - `renderVoronoi(ctx, params) -> void` — voronoi path (delegates to voronoi.js).
  - `exportPNG(ctx) -> Promise<Blob>` — current canvas as PNG.
  - `DEFAULTS` — the params defaults object.
- `noise.js`: `makeNoise(gl, kind, w, h, rng) -> {texture}` for `white`,
  `white_rgb`, `pure_rgb`. `rng.js`: seeded PRNG (`makeRng(seed) -> ()=>float`
  in [0,1], plus `randint(n)`). Bit-exact numpy is NOT required.
- `lic-grid.frag`: integrate `2*steps+1` Euler steps (center+fwd+back) along the
  unit field sampled from `uField`, kernel-weighted accumulation of `uNoise`,
  divide by total weight. Kernels: box / gaussian / raised_cosine
  (`_make_kernel`). Boundary: nearest=clamp, wrap=fract. `step_size` default
  `1/max(W,H)`.

### Agent B — fields (`field.js`, `fields-eval.frag`, `expr.js`, assets)
- `field.js`: `createFieldTexture(gl, params) -> Promise<{texture, w, h}>` —
  rg32f unit-vector field per the encoding above. Pure-math fields are rendered
  by `fields-eval.frag` (port the formulas in `lic_fields.py`); `mhd_cluster` /
  `wd_merger` load a baked asset (see below) and normalise.
- `expr.js`: parse `field_expr` (`f(z)`, complex) → GLSL snippet or a JS
  evaluator; fall back gracefully on parse error.
- `scripts/export_field_assets.py`: runs the Python `mhd_cluster` factory
  (project `.venv` has yt) and writes `app/assets/fields/mhd_cluster.bin` +
  `.json` sidecar (resolution, encoding, slice metadata per CLAUDE.md y-slice
  convention). float16 or 8-bit RG.

### Agent C — color (`color.js`, `color.glsl`)
- `color.js`: `applyColor(gl, ctx, licTarget, fieldTarget, params) -> renders
  RGBA8 to the canvas (or a target)`. Implements `pure_rgb_stochastic` &
  `_streamline`, `angle_hsv` (needs field), `rgb`, `colormap`, plus `enhance`
  (stretch/gamma; stretch needs a min/max reduction over the LIC texture — use
  `readFloatPixels` or a reduction pass). Colormaps come from
  `scripts/bake_colormaps.py` → 256×1 PNG lookups in `app/assets/colormaps/`.
- Per-pixel stochastic RNG may use a GLSL hash (numpy-exact not required).

### Agent D — UI (`app/index.html`, `app/main.js`, `app/style.css`)
- Adapt the existing `web/static/index.html`+`app.js`+`style.css`. Replace the
  `fetch('/generate')` POST with `pipeline.renderGrid/renderVoronoi`; download via
  `pipeline.exportPNG`. Keep Gallery tab (load static `gallery.json`). yt-field
  badge. Live re-render on change (debounce voronoi). Codes against the stub
  `pipeline.js` shipped in Step 0 (renders a placeholder until A lands).

### Agent E — voronoi (`voronoi.js`, `jfa.frag`, `voronoi-lic.frag`)
- `voronoi.js`: `renderVoronoi(gl, ctx, params)` — JFA builds the cell-index map
  (replaces `cKDTree`), per-cell LIC (`compute_voronoi_lic`, N texels, default
  `step_size=1/sqrt(N)`), paint each pixel via cell map, crisp edges via SSAA.
  Uses seeds = `rng.random(N,2)`, per-cell noise from `noise.js` semantics, and
  the color stage from `color.js`. Depends on A's pipeline plumbing — start once
  A's interface is stable.
```
