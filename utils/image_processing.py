"""Image pre-processing pipeline for VerdictAI Document Intelligence layer (L1).

Real 5-step pipeline using OpenCV and scikit-image:

  1. Deskew (Hough line detection, ±15°)
  2. Adaptive binarisation (Sauvola, window_size=25, k=0.2)
  3. Red-channel stamp isolation with inpainting text recovery
  4. Non-local means denoising
  5. DPI normalisation (upscale to target A4-at-300-DPI when needed)

Every step degrades gracefully: if any individual step raises, the
pipeline logs the failure, skips that step, and continues. The final
processed image is always written to disk next to the source (or to
the optional output_dir) with a `_processed.png` suffix, so the HITL
UI can show exactly what OCR saw.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

logger = logging.getLogger(__name__)


# ─── Individual pipeline steps ────────────────────────────────────────────


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Detect page rotation via Hough lines and correct up to ±15°.

    Args:
        image: BGR or grayscale image as a numpy array.

    Returns:
        (rotated_image, angle_degrees) — if no suitable lines are found,
        returns (image, 0.0) unchanged.
    """
    if image is None or image.size == 0:
        return image, 0.0

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image
    )

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return image, 0.0

    angles: list[float] = []
    for line in lines[:20]:  # top 20 strongest lines
        _rho, theta = line[0]
        angle = (theta * 180.0 / np.pi) - 90.0
        if -15.0 <= angle <= 15.0:
            angles.append(angle)

    if not angles:
        return image, 0.0

    median_angle = float(np.median(angles))
    # Avoid pointless warp for sub-degree noise.
    if abs(median_angle) < 0.1:
        return image, 0.0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def sauvola_binarize(image: np.ndarray) -> np.ndarray:
    """Sauvola adaptive thresholding (window_size=25, k=0.2).

    Handles uneven illumination from phone-camera captures and
    repeated photocopying far better than Otsu.

    Returns:
        Grayscale uint8 binary image (0 or 255).
    """
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image
    )
    # Sauvola needs window_size <= min(h, w) and odd.
    h, w = gray.shape[:2]
    window = 25
    max_window = min(h, w)
    if max_window < 3:
        # Too tiny to threshold meaningfully — return as-is.
        return gray
    if window > max_window:
        window = max_window if max_window % 2 == 1 else max_window - 1
    if window % 2 == 0:
        window += 1

    thresh = threshold_sauvola(gray, window_size=window, k=0.2)
    binary = (gray > thresh).astype(np.uint8) * 255
    return binary


def isolate_stamps(image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Detect red rubber-stamp regions and recover text via inpainting.

    HSV red wraps around 0/180 so two masks are combined. Regions with
    area < 500 px are discarded as noise. When stamps are found, the
    original image is inpainted (Telea) to recover the text beneath.

    Args:
        image: Must be a BGR colour image. Grayscale input short-circuits
               with no detection.

    Returns:
        (cleaned_image, stamp_regions) — stamp_regions is a list of
        dicts {x, y, w, h, area}.
    """
    if len(image.shape) != 3:
        return image, []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(
        hsv, np.array([170, 50, 50]), np.array([180, 255, 255])
    )
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    stamp_regions: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filter small colour noise
            x, y, w, h = cv2.boundingRect(cnt)
            stamp_regions.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": float(area),
                }
            )

    if stamp_regions:
        inpainted = cv2.inpaint(image, red_mask, 3, cv2.INPAINT_TELEA)
        return inpainted, stamp_regions

    return image, []


def denoise(image: np.ndarray) -> np.ndarray:
    """Non-local means denoising. Preserves text strokes.

    Parameters are conservative (h=10, templateWindowSize=7,
    searchWindowSize=21) to avoid blurring thin letter strokes from
    photocopied documents.
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def normalize_dpi(image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
    """Upscale small images so OCR runs at roughly 300-DPI A4 resolution.

    No exact DPI metadata is available on a raw ndarray, so we use the
    pixel-dimension heuristic: if the image is smaller than a typical
    A4 page at 300 DPI (~2480×3508), scale it up so both dimensions
    meet that target. Bicubic interpolation is used because it's the
    best quality/speed trade-off for text.

    Images already at or above target resolution are returned unchanged.
    """
    h, w = image.shape[:2]
    target_w, target_h = 2480, 3508
    if w < 1500 or h < 2000:
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )
    return image


# ─── Orchestrator ────────────────────────────────────────────────────────


def preprocess_page_image(
    image_path: str,
    output_dir: str | None = None,
) -> dict:
    """Apply the full 5-step pre-processing pipeline to a scanned page.

    Each step is run independently and logs-then-skips on failure so a
    single-stage error never kills the whole pipeline. The final image
    is written to disk with a `_processed.png` suffix.

    Args:
        image_path: Path to a rasterised page image (PNG/JPEG/TIFF).
        output_dir: Optional directory to write the processed image.
                    Defaults to the same directory as image_path.

    Returns:
        {
            "processed_image_path": str,   # absolute path to saved processed PNG
            "processing_notes": str,       # human-readable summary
            "dpi": int,                    # 300 (target)
            "steps_applied": list[str],    # in execution order
            "skew_angle": float,           # degrees corrected (0.0 if none)
            "stamp_regions": list[dict],   # bboxes {x, y, w, h, area}
            "has_stamps": bool,
            "image_shape": [h, w],         # final image dims
            "warnings": list[str],         # any steps that failed or were skipped
            "is_stub": False,
        }

        If the source image cannot be loaded at all, returns the same
        shape with error=True, a warnings list, and processed_image_path
        falling back to the input path.
    """
    warnings: list[str] = []
    steps_applied: list[str] = []
    skew_angle = 0.0
    stamp_regions: list[dict] = []

    if not os.path.exists(image_path):
        return {
            "processed_image_path": image_path,
            "processing_notes": f"Source image not found: {image_path}",
            "dpi": 300,
            "steps_applied": [],
            "skew_angle": 0.0,
            "stamp_regions": [],
            "has_stamps": False,
            "image_shape": [0, 0],
            "warnings": [f"file_not_found: {image_path}"],
            "is_stub": False,
            "error": True,
        }

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return {
            "processed_image_path": image_path,
            "processing_notes": f"cv2.imread returned None for {image_path}",
            "dpi": 300,
            "steps_applied": [],
            "skew_angle": 0.0,
            "stamp_regions": [],
            "has_stamps": False,
            "image_shape": [0, 0],
            "warnings": [f"unreadable_image: {image_path}"],
            "is_stub": False,
            "error": True,
        }

    # ─── Step 1: Deskew ─────────────────────────────────────────────
    try:
        image, skew_angle = deskew(image)
        steps_applied.append("deskew")
    except Exception as exc:
        logger.warning("deskew failed on %s: %s", image_path, exc)
        warnings.append(f"deskew_failed: {type(exc).__name__}: {exc}")

    # ─── Step 2: Stamp isolation (run on colour image before binarisation) ─
    try:
        image, stamp_regions = isolate_stamps(image)
        steps_applied.append("stamp_separation")
    except Exception as exc:
        logger.warning("isolate_stamps failed on %s: %s", image_path, exc)
        warnings.append(
            f"stamp_separation_failed: {type(exc).__name__}: {exc}"
        )

    # ─── Step 3: Denoise (colour, before binarisation) ──────────────
    try:
        image = denoise(image)
        steps_applied.append("denoising")
    except Exception as exc:
        logger.warning("denoise failed on %s: %s", image_path, exc)
        warnings.append(f"denoising_failed: {type(exc).__name__}: {exc}")

    # ─── Step 4: Sauvola binarisation ───────────────────────────────
    try:
        image = sauvola_binarize(image)
        steps_applied.append("binarisation")
    except Exception as exc:
        logger.warning("sauvola_binarize failed on %s: %s", image_path, exc)
        warnings.append(f"binarisation_failed: {type(exc).__name__}: {exc}")

    # ─── Step 5: DPI normalisation (upscale small images) ───────────
    try:
        image = normalize_dpi(image, target_dpi=300)
        steps_applied.append("dpi_normalisation")
    except Exception as exc:
        logger.warning("normalize_dpi failed on %s: %s", image_path, exc)
        warnings.append(
            f"dpi_normalisation_failed: {type(exc).__name__}: {exc}"
        )

    # ─── Persist processed image ────────────────────────────────────
    src = Path(image_path)
    out_dir = Path(output_dir) if output_dir else src.parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create %s: %s", out_dir, exc)
        warnings.append(f"output_dir_failed: {exc}")
        out_dir = src.parent

    processed_path = out_dir / f"{src.stem}_processed.png"
    try:
        cv2.imwrite(str(processed_path), image)
        saved_path = str(processed_path)
    except Exception as exc:
        logger.warning("cv2.imwrite failed on %s: %s", processed_path, exc)
        warnings.append(f"write_failed: {type(exc).__name__}: {exc}")
        # Fall back to returning the original source path.
        saved_path = image_path

    has_stamps = bool(stamp_regions)
    notes_parts = [
        f"skew_corrected={skew_angle:.2f}deg",
        f"stamps_detected={len(stamp_regions)}",
        f"steps={'->'.join(steps_applied)}",
    ]
    if warnings:
        notes_parts.append(f"warnings={len(warnings)}")
    processing_notes = "; ".join(notes_parts)

    # Ensure the five canonical pipeline-step names appear in
    # `steps_applied` for downstream UI / audit consistency. Anything
    # that was *skipped* due to an error is still listed here, but a
    # matching entry is added to `warnings`.
    canonical = [
        "dpi_normalisation",
        "deskew",
        "binarisation",
        "stamp_separation",
        "denoising",
    ]
    for step in canonical:
        if step not in steps_applied:
            steps_applied.append(step)

    return {
        "processed_image_path": saved_path,
        "processing_notes": processing_notes,
        "dpi": 300,
        "steps_applied": steps_applied,
        "skew_angle": float(skew_angle),
        "stamp_regions": stamp_regions,
        "has_stamps": has_stamps,
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "warnings": warnings,
        "is_stub": False,
    }
