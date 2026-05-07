"""Image-document ingestion utilities for VerdictAI L1.

Handles single-image "documents" — typical when a field officer
photographs a certificate with their phone instead of scanning it.
The file is treated as a one-page scanned document so the full
OCR + preprocessing pipeline runs exactly as it does for a scanned
PDF.

Supported input formats:
    .png, .jpg, .jpeg, .tif, .tiff, .bmp

The function ``copy_image_as_page`` normalises every supported input
to a PNG next to a canonical filename, which keeps downstream OpenCV
behaviour predictable regardless of the source codec.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
}


def parse_image(file_path: str) -> dict:
    """Treat a single image file as a one-page scanned document.

    Returns a shape matching :func:`parse_pdf` so L1 can dispatch
    uniformly. ``is_scanned=True`` forces the OCR path downstream.

    Args:
        file_path: Path to the image file (PNG, JPG, TIFF, BMP).

    Returns:
        On success::

            {
              "page_count": 1,
              "pages": [{"page_number": 1, "text": "", "has_tables": False}],
              "is_scanned": True,
            }

        On failure (file missing or unsupported extension)::

            {"error": True, "message": str, "page_count": 0,
             "pages": [], "is_scanned": True}
    """
    if not os.path.exists(file_path):
        return {
            "error": True,
            "message": f"File not found: {file_path}",
            "page_count": 0,
            "pages": [],
            "is_scanned": True,
        }

    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return {
            "error": True,
            "message": (
                f"Unsupported image extension '{ext}'. "
                f"Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            ),
            "page_count": 0,
            "pages": [],
            "is_scanned": True,
        }

    return {
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "text": "",
                "has_tables": False,
            }
        ],
        "is_scanned": True,
    }


def copy_image_as_page(image_path: str, output_dir: str) -> list[str]:
    """Copy/convert the input image to a canonical PNG page file.

    The output filename matches the pattern used by
    :func:`extract_page_images` for PDFs (``{stem}_page_1.png``) so
    everything downstream sees the same shape.

    Supported input formats: PNG, JPG, JPEG, TIFF, BMP. Non-PNG inputs
    are re-encoded via Pillow to guarantee OpenCV compatibility (some
    TIFFs in particular use compression schemes OpenCV can't read).

    Args:
        image_path: Path to the source image.
        output_dir: Directory where the page PNG will be written.
                    Created if it does not exist.

    Returns:
        ``[output_png_path]`` on success, ``[]`` if the source is
        missing, unsupported, or cannot be opened.
    """
    if not os.path.exists(image_path):
        logger.error("copy_image_as_page: source not found: %s", image_path)
        return []

    ext = Path(image_path).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.error(
            "copy_image_as_page: unsupported extension '%s' for %s",
            ext, image_path,
        )
        return []

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output directory %s: %s", output_dir, exc)
        return []

    stem = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{stem}_page_1.png")

    # PNG → direct copy; other formats → re-encode via Pillow.
    if ext == ".png":
        try:
            shutil.copyfile(image_path, output_path)
            return [output_path]
        except OSError as exc:
            logger.error(
                "Failed to copy %s → %s: %s", image_path, output_path, exc
            )
            return []

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        logger.error("Pillow not installed: %s", exc)
        return []

    try:
        with Image.open(image_path) as img:
            # Some TIFFs are multi-frame; take the first frame so we
            # always produce exactly one page image.
            if getattr(img, "is_animated", False):
                try:
                    img.seek(0)
                except EOFError:
                    pass

            # Normalise mode so OpenCV's imread doesn't choke on exotic
            # palettes or alpha-only images.
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            img.save(output_path, format="PNG", optimize=True)
        return [output_path]
    except Exception as exc:
        logger.error(
            "Pillow failed to convert %s → PNG: %s: %s",
            image_path, type(exc).__name__, exc,
        )
        return []
