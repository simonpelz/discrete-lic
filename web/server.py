"""FastAPI web server for discrete-lic.

Start with:
    uvicorn web.server:app --reload
from the project root, or:
    cd web && uvicorn server:app --reload
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# Add src/ to path so we can import the LIC modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lic_fields import FIELDS  # noqa: E402

GALLERY_DIR = Path(__file__).parent / "gallery"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="discrete-lic", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# Field metadata
# ---------------------------------------------------------------------------

FIELD_DESCRIPTIONS = {
    "uniform": "Constant direction — straight parallel lines",
    "rotation": "Concentric circles around a center point",
    "source": "Radial rays outward from a point",
    "sink": "Radial rays inward to a point",
    "saddle": "Hyperbolic X-shaped flow",
    "shear": "Linear shear flow",
    "double_vortex": "Two counter-rotating vortices",
    "wave": "Sinusoidal woven pattern",
    "perlin_curl": "Turbulent curl-noise field (divergence-free)",
    "spiral": "Logarithmic spiral",
    "complex": "User-defined analytic function f(z) in the complex plane",
    "mhd_cluster": "Magnetohydrodynamic galaxy cluster simulation (requires yt)",
    "wd_merger": "White dwarf merger magnetic field snapshot (requires yt)",
}


@app.get("/fields")
def get_fields():
    fields = []
    for name in list(FIELDS.keys()) + ["complex"]:
        fields.append({
            "name": name,
            "description": FIELD_DESCRIPTIONS.get(name, ""),
            "requires_yt": name in ("mhd_cluster", "wd_merger"),
        })
    return JSONResponse(fields)


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

@app.get("/gallery")
def get_gallery():
    meta_path = GALLERY_DIR / "gallery.json"
    if not meta_path.exists():
        return JSONResponse([])
    with open(meta_path) as f:
        return JSONResponse(json.load(f))


@app.get("/gallery/{filename}")
def get_gallery_image(filename: str):
    # Prevent path traversal
    path = GALLERY_DIR / Path(filename).name
    if not path.exists() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    field: str = "rotation"
    field_expr: str = "z**2"
    color: str = "pure_rgb"
    pixel_mode: str = "voronoi"
    noise: str = "white_rgb"
    steps: int = 30
    kernel: str = "box"
    kernel_sigma: float = 0.3
    boundary: str = "nearest"
    enhance: str = "stretch"
    gamma: float = 1.0
    voronoi_cells: int = 800
    pure_mode: str = "stochastic"
    pure_sharpen: float = 2.5
    streamline_stride: int = 4
    colormap: str = "viridis"
    sat: float = 0.9
    noise_points: int = 200
    noise_block: int = 8
    width: int = 600
    height: Optional[int] = None
    dpi: int = 96
    seed: int = 42
    # yt-backed fields
    mhd_sample: str = "MHDSloshing"
    mhd_vector: str = "vorticity"
    mhd_slice_axis: str = "y"
    mhd_resolution: int = 512
    mhd_width_kpc: float = 500.0
    wd_dataset: Optional[str] = None
    wd_resolution: int = 512
    wd_slice_axis: str = "theta"
    step_size: Optional[float] = None

    @field_validator("width")
    @classmethod
    def cap_width(cls, v):
        return min(v, 1024)

    @field_validator("height")
    @classmethod
    def cap_height(cls, v):
        if v is not None:
            return min(v, 1024)
        return v


@app.post("/generate")
async def generate_image(req: GenerateRequest):
    import asyncio
    from functools import partial

    # Import here so a missing dependency gives a clean error at request time
    try:
        from lic_main import generate
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Import error: {e}")

    args = SimpleNamespace(**req.model_dump())

    loop = asyncio.get_event_loop()
    try:
        pil_img = await loop.run_in_executor(None, partial(generate, args))
    except (ValueError, ImportError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------------------------------------------------------------------
# Static files + SPA root
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
