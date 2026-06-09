# Feature ideas & open items

## Pending verification

- [ ] **Generator endpoint end-to-end** — start the server, POST to `/generate` with
  `{"field":"rotation","color":"pure_rgb","pixel_mode":"voronoi"}` and confirm a PNG comes back.
  The server starts and the SPA loads (verified), but the generate round-trip wasn't tested
  because background processes don't survive between sandbox shell invocations.
  Manual test:
  ```bash
  curl -X POST http://127.0.0.1:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"field":"rotation","color":"pure_rgb","pixel_mode":"voronoi","voronoi_cells":400,"width":300}' \
    --output /tmp/test_gen.png && open /tmp/test_gen.png
  ```

## Web UI

- [ ] **"Copy CLI command" button** — show the exact `python src/lic_main.py …` invocation
  matching the current generator settings; lets people reproduce in the CLI.
- [ ] **Slow-mode warning** — grid mode at 800×800 takes ~30 s; show an estimated time or
  a "this will be slow" banner when pixel-mode=grid is selected.
- [ ] **Expose advanced controls** — `--kernel-sigma`, `--boundary` (wrap/nearest),
  `--sat`, `--enhance`, `--gamma` are all wired server-side but hidden in the UI.
  Could live behind an "Advanced" disclosure triangle.
- [ ] **Random button** — pick a random field + color combination and fill in the form.
- [ ] **Height independence** — currently height defaults to width; expose them as independent
  inputs (the server already accepts independent height).

## Color modes & fields

- [ ] **`pure_cmyk` mode** — same idea but with cyan, magenta, yellow as the three primaries.
  Subtractive-color counterpart to pure_rgb; visually quite different.
- [ ] **`pure_rygb` or custom primaries** — let the user define 2–4 primaries as hex colors.
- [ ] **Animated export** — generate N frames sweeping `--seed` or a field parameter;
  output as GIF or APNG. Even just 12 frames of a rotating `perlin_curl` seed would be
  striking.
- [ ] **`--color streamline_rgb`** — draw actual visible streamlines (thin curves) in R/G/B
  on a black background, rather than filling every pixel.

## Gallery & README

- [ ] **Add yt-backed gallery image** — once yt is installed, generate one `mhd_cluster`
  example for the gallery (it was the original source image for the KAB piece and is the
  most dramatic). Requires running `generate_gallery.py` with yt installed.
- [ ] **README: add `pure_rgb_perlin_grid.png`** as an explicit comparison to show grid
  vs Voronoi mode side-by-side.

## Packaging (low priority)

- [ ] **`pyproject.toml`** with `[project.scripts] lic = "lic_main:main"` so `pip install -e .`
  gives a `lic` command. Keep the "clone and run" path as primary.
- [ ] **GitHub Actions** — on push to main, run `generate_gallery.py` and commit updated
  gallery images. Keeps gallery fresh if anyone adds a new example definition.
