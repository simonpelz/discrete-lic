"""Format-agnostic output: saves a PIL Image as PNG, TIFF, JPEG, or PDF.

Format is inferred from the file extension. For PDF, the image is embedded
into a single-page document sized to match the image at the given DPI.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


_TIFF = {".tif", ".tiff"}
_JPEG = {".jpg", ".jpeg"}
_PNG  = {".png"}
_PDF  = {".pdf"}
_ALL  = _TIFF | _JPEG | _PNG | _PDF


def save_image(image: Image.Image, path: Path, dpi: int = 150) -> None:
    """Save a PIL RGB Image to path. Format inferred from suffix.

    Supported suffixes: .png .tif .tiff .jpg .jpeg .pdf
    """
    suffix = path.suffix.lower()
    if suffix in _PNG:
        image.save(str(path), format="PNG", dpi=(dpi, dpi))
    elif suffix in _TIFF:
        image.save(str(path), format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    elif suffix in _JPEG:
        image.save(str(path), format="JPEG", quality=95, dpi=(dpi, dpi))
    elif suffix in _PDF:
        _save_as_pdf(image, path, dpi)
    else:
        raise ValueError(
            f"Unsupported output format: {suffix!r}. "
            f"Use one of: {', '.join(sorted(_ALL))}"
        )


def _save_as_pdf(image: Image.Image, path: Path, dpi: int) -> None:
    """Embed image into a single-page PDF sized to fit the image at the given DPI."""
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rl_canvas

    W_px, H_px = image.size
    # Convert pixels → PDF points (1 point = 1/72 inch)
    page_w = W_px / dpi * 72.0
    page_h = H_px / dpi * 72.0

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    c = rl_canvas.Canvas(str(path), pagesize=(page_w, page_h))
    c.drawImage(ImageReader(buf), 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()
