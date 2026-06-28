#!/usr/bin/env python3
"""Headless-browser fidelity check for the WebGL LIC pipeline.

Serves the repo, drives the real pipeline via harness.html in headless Chromium
(SwiftShader), screenshots the canvas for each refs/*.json case, saves it to
app/fidelity/out/, and diffs against the Python CLI reference refs/<name>.png.

Run WITHOUT the sandbox so the in-process server and the browser share a network
namespace:
    python3 app/fidelity/verify_browser.py
"""
from __future__ import annotations

import base64
import http.server
import json
import socketserver
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]              # discrete-lic/
REFS = HERE / "refs"
OUT = HERE / "out"
PORT = 8137

GL_ARGS = [
    "--use-gl=angle", "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
]


def serve():
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(REPO), **k)
        def log_message(self, *a):  # quiet
            pass
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", PORT), H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def diff(a: Image.Image, b: Image.Image):
    """Return (mean_abs_diff_0_255, ssim_or_None) on a common size."""
    if a.size != b.size:
        b = b.resize(a.size)
    aa = np.asarray(a.convert("RGB"), np.float64)
    bb = np.asarray(b.convert("RGB"), np.float64)
    mad = float(np.abs(aa - bb).mean())
    ssim = None
    try:
        from skimage.metrics import structural_similarity as ss
        ssim = float(ss(aa, bb, channel_axis=2, data_range=255))
    except Exception:
        pass
    return mad, ssim


def main() -> int:
    OUT.mkdir(exist_ok=True)
    cases = sorted(REFS.glob("*.json"))
    if not cases:
        print("No refs/*.json — run render_refs.py first.")
        return 1

    httpd = serve()
    results = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=GL_ARGS)
            page = browser.new_page(viewport={"width": 640, "height": 640})
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{PORT}/app/fidelity/harness.html")
            page.wait_for_function("window.__harnessReady === true", timeout=15000)

            init_err = page.evaluate("window.__initError")
            print("WebGL init:", "OK" if not init_err else f"FAILED: {init_err}")
            if init_err:
                return 2

            for jf in cases:
                params = json.loads(jf.read_text())
                name = jf.stem
                res = page.evaluate("(p) => window.licRender(p)", params)
                if res.get("error"):
                    print(f"  {name}: RENDER ERROR: {res['error'][:300]}")
                    results.append((name, None, None, "render-error"))
                    continue
                png = base64.b64decode(res["dataUrl"].split(",", 1)[1])
                out_png = OUT / f"{name}.png"
                out_png.write_bytes(png)
                ref_png = REFS / f"{name}.png"
                if ref_png.exists():
                    mad, ssim = diff(Image.open(out_png), Image.open(ref_png))
                    s = f"MAD={mad:6.2f}/255" + (f"  SSIM={ssim:.3f}" if ssim is not None else "")
                    print(f"  {name}: {s}")
                    results.append((name, mad, ssim, "ok"))
                else:
                    print(f"  {name}: rendered (no ref to diff)")
                    results.append((name, None, None, "no-ref"))
            browser.close()
            if errors:
                print("\nConsole/page errors:")
                for e in errors[:20]:
                    print("  !", e[:300])
    finally:
        httpd.shutdown()

    print(f"\nScreenshots in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
