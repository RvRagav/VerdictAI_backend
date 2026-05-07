"""PDF parsing utilities for VerdictAI Document Intelligence layer (L1).

Real PDF processing using pdfplumber (embedded text + tables) and
pdf2image/poppler (page rasterisation).

- parse_pdf: Extract page count, embedded text per page, detect tables,
             classify scanned vs digital PDFs based on text density.
- extract_page_images: Rasterise each page to a high-DPI PNG on disk
                       using poppler (pdftoppm) via pdf2image.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

logger = logging.getLogger(__name__)

# Threshold (characters per page, averaged) below which a PDF is treated
# as "scanned" and routed through the OCR pipeline. A fully digital PDF
# typically has hundreds-to-thousands of characters per page; a blank
# or pure-image PDF has near zero.
_SCANNED_TEXT_THRESHOLD = 50


def parse_pdf(file_path: str) -> dict:
    """Parse a PDF, extract text/tables per page, and classify scanned vs digital.

    Uses pdfplumber to:
      * Count pages
      * Extract embedded text per page
      * Detect whether each page contains tables (has_tables flag)

    Classification rule: if average embedded-text length per page is below
    _SCANNED_TEXT_THRESHOLD, the PDF is flagged is_scanned=True and the
    caller should route it through the OCR pipeline.

    Args:
        file_path: Path to the PDF file.

    Returns:
        On success:
            {
              "page_count": int,
              "pages": [{"page_number": int, "text": str, "has_tables": bool}, ...],
              "is_scanned": bool,
            }
        On failure:
            {"error": True, "message": str, "page_count": 0, "pages": [], "is_scanned": False}
    """
    if not os.path.exists(file_path):
        return {
            "error": True,
            "message": f"File not found: {file_path}",
            "page_count": 0,
            "pages": [],
            "is_scanned": False,
        }

    try:
        pages: list[dict] = []
        total_text_length = 0

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                # Embedded text extraction. pdfplumber returns None when a
                # page has no extractable text layer (pure image / scanned).
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # pragma: no cover - rare parser glitch
                    logger.warning(
                        "pdfplumber.extract_text failed on page %d of %s: %s",
                        i + 1,
                        file_path,
                        exc,
                    )
                    text = ""

                # Table detection — find_tables returns [] when none present.
                try:
                    tables = page.find_tables()
                    has_tables = bool(tables)
                except Exception as exc:  # pragma: no cover - rare parser glitch
                    logger.warning(
                        "pdfplumber.find_tables failed on page %d of %s: %s",
                        i + 1,
                        file_path,
                        exc,
                    )
                    has_tables = False

                total_text_length += len(text.strip())
                pages.append(
                    {
                        "page_number": i + 1,
                        "text": text,
                        "has_tables": has_tables,
                    }
                )

        avg_text_per_page = (
            total_text_length / page_count if page_count > 0 else 0
        )
        is_scanned = avg_text_per_page < _SCANNED_TEXT_THRESHOLD

        return {
            "page_count": page_count,
            "pages": pages,
            "is_scanned": is_scanned,
        }

    except Exception as exc:
        return {
            "error": True,
            "message": (
                f"Failed to parse PDF '{file_path}': "
                f"{type(exc).__name__}: {exc}"
            ),
            "page_count": 0,
            "pages": [],
            "is_scanned": False,
        }


def extract_page_images(
    file_path: str,
    output_dir: str,
    dpi: int = 300,
) -> list[str]:
    """Rasterise every page of a PDF to a PNG file on disk.

    Uses pdf2image (poppler / pdftoppm under the hood) at the given DPI.
    300 DPI is the standard used elsewhere in the pipeline and what
    Tesseract is calibrated against for government-document OCR.

    Args:
        file_path: Path to the PDF file.
        output_dir: Directory to write PNG images. Created if missing.
        dpi: Rasterisation DPI. Defaults to 300.

    Returns:
        List of filesystem paths to the written PNGs, one per page,
        ordered by page number. Returns [] on any failure (missing
        file, poppler not installed, corrupt PDF, I/O error).
    """
    if not os.path.exists(file_path):
        return []

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output directory %s: %s", output_dir, exc)
        return []

    base_name = Path(file_path).stem

    try:
        images = convert_from_path(file_path, dpi=dpi, fmt="png")
    except (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError,
    ) as exc:
        logger.error(
            "pdf2image failed on %s: %s: %s",
            file_path,
            type(exc).__name__,
            exc,
        )
        return []
    except Exception as exc:
        logger.error(
            "Unexpected error rasterising %s: %s: %s",
            file_path,
            type(exc).__name__,
            exc,
        )
        return []

    image_paths: list[str] = []
    for i, image in enumerate(images):
        image_filename = f"{base_name}_page_{i + 1}.png"
        image_path = os.path.join(output_dir, image_filename)
        try:
            # optimize=True reduces file size noticeably with no quality loss
            image.save(image_path, format="PNG", optimize=True)
            image_paths.append(image_path)
        except OSError as exc:
            logger.error("Failed to write %s: %s", image_path, exc)
            # Continue — partial success is more useful than total failure.

    return image_paths
