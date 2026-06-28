# In-browser LIC generator (`app/`)

A zero-install, fully client-side WebGL2 reimplementation of the LIC generator.
No Python backend, no `fetch('/generate')` — the entire pipeline
(field → noise → LIC → color) runs on the GPU in the browser, so the published
GitHub Pages site works for anyone with the URL.

This is the browser port of the Python pipeline in `../src/`. The Python CLI
remains the source of truth for print-resolution and yt-backed renders.

## Run locally

ES modules + shader/asset `fetch` require an HTTP server (not `file://`):

```bash
cd discrete-lic
python3 -m http.server 8000
# open http://localhost:8000/app/
```

## Architecture (`app/gl/`)

| File | Role |
|---|---|
| `glutil.js` | shared WebGL2 helpers (programs, float targets, FBOs) |
| `pipeline.js` | orchestrator + public API: `init`, `renderGrid`, `renderVoronoi`, `exportPNG`, `DEFAULTS` |
| `lic-grid.frag` | grid-mode LIC shader (port of `compute_lic`) |
| `noise.js`, `rng.js` | seeded noise textures (white / white_rgb / pure_rgb) |
| `field.js`, `fields-eval.frag`, `expr.js` | vector fields (pure-math + complex `f(z)`) and baked-field loader |
| `color.js`, `color.glsl` | color stage (pure_rgb / angle_hsv / colormap / rgb + enhance) |
| `voronoi.js`, `jfa.frag`, `voronoi-lic.frag` | Voronoi-pixel mode via Jump Flood (the artwork's mode) |

The frozen module contract is in `../app/CONTRACT.md`.

**Fidelity bar:** visual equivalence, not bit-exactness. The seeded RNG is a JS
PRNG (not numpy PCG64) and math is float32, so per-pixel patterns differ from the
CLI while the *character* (fibre length, mosaic density, RGB balance) matches.
All physical prints come from the Python CLI.

## Verify against the Python reference

`app/fidelity/` renders the same parameter sets with the Python CLI and the
browser, then diffs (MAD + SSIM).

```bash
# 1. Python reference images (pure-math cases; system python):
python3 app/fidelity/render_refs.py
#    add the yt artwork case (needs the venv + MHDSloshing sample):
VENV=../.venv/bin/python python3 app/fidelity/render_refs.py --with-yt

# 2. Browser renders + diff (needs Playwright + a working WebGL2 browser):
pip install playwright && python3 -m playwright install chromium
python3 app/fidelity/verify_browser.py
#    screenshots land in app/fidelity/out/, diffs print to stdout.
```

> Note: `verify_browser.py` needs a GPU/SwiftShader-capable browser. A headless
> environment without a GL stack (no Mesa/Vulkan loader, no Xvfb) will report
> "WebGL2 is not available" — run it on a normal desktop with Chrome.

## Open items

- **Real `mhd_cluster` field asset**: `app/assets/fields/mhd_cluster.bin` is
  currently a **synthetic placeholder** (`"synthetic": true` in the `.json`). The
  real bake downloads the ~1.5 GB yt MHDSloshing sample:
  ```bash
  rm app/assets/fields/mhd_cluster.*
  ../.venv/bin/python scripts/export_field_assets.py
  ```
- **Y-flip orientation**: each module keeps one net Y-flip; confirm grid and
  voronoi outputs are not vertically mirrored vs the Python reference, and fix at
  the single flip point if so (don't add a second).
- **Deploy**: point GitHub Pages at `app/` (or link it from the root gallery
  `index.html`).
