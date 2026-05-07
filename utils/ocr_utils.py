"""OCR utilities for VerdictAI Document Intelligence layer (L1).

Real OCR using Tesseract 5 via pytesseract. Extracts per-word bounding
boxes and confidence scores from pre-processed page images.

- extract_text_from_image: Tesseract image_to_data → word objects.
- compute_page_confidence:  Length-weighted mean of per-word confidences.
- is_degraded_page:         Routing threshold — confidence < 0.50.
"""

from __future__ import annotations

import logging
import os
import uuid

import pytesseract
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Tesseract engine config. --oem 3 = default LSTM, --psm 6 = assume a
# single uniform block of text (best default for multi-paragraph A4
# scans). Callers can override by calling pytesseract directly.
_TESSERACT_CONFIG = "--oem 3 --psm 6"

# Reported engine identifier threaded through to stored word_objects.
_ENGINE_ID = "tesseract-5.5"


def extract_text_from_image(
    image_path: str,
    languages: str = "eng",
) -> dict:
    """Run Tesseract OCR on an image and return per-word bboxes + confidence.

    Uses pytesseract.image_to_data with DICT output so every recognised
    word carries its left/top/width/height and 0–100 confidence — which
    we normalise to [0, 1.0]. Tesseract returns confidence = -1 for
    skipped regions; those rows are dropped.

    Args:
        image_path: Path to the (ideally pre-processed) page image.
        languages: Tesseract language code(s) joined with '+' for
                   multilingual pages. Examples: "eng", "eng+hin",
                   "eng+tam", "eng+kan". Defaults to "eng".

    Returns:
        {
            "words": [
                {
                    "id": str,
                    "text_content": str,
                    "x_min": float, "y_min": float,
                    "x_max": float, "y_max": float,
                    "confidence": float,   # 0.0–1.0
                    "source_engine": "tesseract",
                },
                ...
            ],
            "raw_text": str,               # space-joined reconstruction
            "engine": "tesseract-5.5",
            "is_stub": False,
            "word_count": int,
            "mean_confidence": float,
        }

        On any error (missing file, unreadable image, Tesseract not on
        PATH) returns the same shape with an "error" field populated
        and empty words/raw_text.
    """
    if not os.path.exists(image_path):
        return _error_result(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
        image.load()  # force-decode now so errors surface here
    except (UnidentifiedImageError, OSError) as exc:
        return _error_result(f"Cannot open image '{image_path}': {exc}")

    try:
        data = pytesseract.image_to_data(
            image,
            lang=languages,
            output_type=pytesseract.Output.DICT,
            config=_TESSERACT_CONFIG,
        )
    except pytesseract.TesseractNotFoundError as exc:
        logger.error("Tesseract binary not found on PATH: %s", exc)
        return _error_result(f"Tesseract binary not available: {exc}")
    except pytesseract.TesseractError as exc:
        logger.error("Tesseract OCR failed on %s: %s", image_path, exc)
        return _error_result(f"Tesseract OCR failed: {exc}")
    except Exception as exc:
        logger.error("Unexpected OCR error on %s: %s", image_path, exc)
        return _error_result(f"OCR error: {type(exc).__name__}: {exc}")

    words: list[dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = data["text"][i].strip() if data["text"][i] else ""
        try:
            conf_raw = float(data["conf"][i])
        except (TypeError, ValueError):
            continue

        if not text or conf_raw < 0:
            # Tesseract returns -1 for layout-only rows (block/par/line).
            continue

        left = float(data["left"][i])
        top = float(data["top"][i])
        width = float(data["width"][i])
        height = float(data["height"][i])

        words.append(
            {
                "id": str(uuid.uuid4()),
                "text_content": text,
                "x_min": left,
                "y_min": top,
                "x_max": left + width,
                "y_max": top + height,
                "confidence": round(conf_raw / 100.0, 4),
                "source_engine": "tesseract",
            }
        )

    raw_text = " ".join(w["text_content"] for w in words)
    mean_conf = (
        sum(w["confidence"] for w in words) / len(words) if words else 0.0
    )

    return {
        "words": words,
        "raw_text": raw_text,
        "engine": _ENGINE_ID,
        "is_stub": False,
        "word_count": len(words),
        "mean_confidence": round(mean_conf, 4),
    }


def compute_page_confidence(words: list[dict]) -> float:
    """Length-weighted mean of per-word confidences.

    Longer words contribute proportionally more to the page confidence,
    which reduces the influence of short high-confidence artefacts
    (e.g. a stray '.' at 1.0) on the page-level routing decision.

    Args:
        words: List of word dicts with 'text_content' and 'confidence'.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for empty input.
    """
    if not words:
        return 0.0

    total_weighted = 0.0
    total_length = 0
    for word in words:
        text = word.get("text_content", "")
        confidence = word.get("confidence", 0.0)
        word_len = len(text)
        total_weighted += word_len * confidence
        total_length += word_len

    if total_length == 0:
        return 0.0
    return total_weighted / total_length


def is_degraded_page(confidence: float) -> bool:
    """Page-level degradation flag used for HITL routing.

    Pages below 0.50 are considered unreliable and will be routed to
    human review rather than auto-committed.
    """
    return confidence < 0.50


# ─── Helpers ─────────────────────────────────────────────────────────────


def _error_result(message: str) -> dict:
    """Shape-preserving error payload for OCR failures."""
    return {
        "words": [],
        "raw_text": "",
        "engine": _ENGINE_ID,
        "is_stub": False,
        "error": message,
        "word_count": 0,
        "mean_confidence": 0.0,
    }
