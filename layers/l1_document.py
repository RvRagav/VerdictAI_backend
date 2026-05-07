"""Layer 1: Document Intelligence for VerdictAI.

Orchestrates the document processing pipeline:
Document upload → parse → pre-process → OCR → store normalised objects.

Supported input formats:
    PDF  (.pdf)                          — pdfplumber + pdf2image + OCR
    DOCX (.docx)                         — python-docx, digital text path
    Image (.png .jpg .jpeg .tif .tiff .bmp) — Pillow + OpenCV + OCR

The public function :func:`process_document` dispatches on the file
extension and produces a uniform output contract regardless of input
format: documents + pages + word_objects rows, plus
``document_received`` and ``ocr_completed`` audit events.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from layers.l5_audit import append_audit_event
from utils.hash_utils import compute_file_hash
from utils.image_processing import preprocess_page_image
from utils.ocr_utils import (
    compute_page_confidence,
    extract_text_from_image,
    is_degraded_page,
)
from utils.pdf_utils import extract_page_images, parse_pdf
from utils.docx_utils import parse_docx
from utils.image_utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    copy_image_as_page,
    parse_image,
)


# ─── Format detection ────────────────────────────────────────────────────


def _detect_format(file_path: str) -> str:
    """Return one of 'pdf', 'docx', 'image', or 'unknown'."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


# ─── Main entry point ────────────────────────────────────────────────────


def process_document(
    conn: sqlite3.Connection,
    tender_id: str,
    file_path: str,
    doc_type: str,
    bidder_id: str | None = None,
) -> dict:
    """Process a document (PDF, DOCX, or image) through the full L1 pipeline.

    Dispatches by file extension. All three branches produce the same
    output shape and store rows in documents / pages / word_objects.

    Args:
        conn: Active SQLite connection (caller manages transaction).
        tender_id: Tender this document belongs to.
        file_path: Path to the source file on disk.
        doc_type: One of "nit", "corrigendum", "bidder_submission",
                  "certificate".
        bidder_id: Optional bidder ID for bidder-scoped documents.

    Returns:
        Dict with keys: id, tender_id, bidder_id, doc_type, filename,
        file_path, sha256_hash, page_count, avg_ocr_confidence,
        processing_status, pages.
    """
    document_id = str(uuid.uuid4())
    filename = os.path.basename(file_path)
    upload_timestamp = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )

    # SHA-256 is computed for every format — raises FileNotFoundError
    # if the path is bogus, which matches historical PDF behaviour.
    file_hash = compute_file_hash(file_path)

    fmt = _detect_format(file_path)

    if fmt == "pdf":
        parsed = parse_pdf(file_path)
    elif fmt == "docx":
        parsed = parse_docx(file_path)
    elif fmt == "image":
        parsed = parse_image(file_path)
    else:
        parsed = {
            "error": True,
            "message": (
                f"Unsupported file format for '{filename}'. "
                "Supported: PDF, DOCX, PNG, JPG, JPEG, TIFF, BMP."
            ),
            "page_count": 0,
            "pages": [],
            "is_scanned": False,
        }

    # ── Parse failure → record as error and return early ──
    if parsed.get("error"):
        conn.execute(
            """INSERT INTO documents (id, tender_id, bidder_id, doc_type, filename, file_path,
               sha256_hash, page_count, avg_ocr_confidence, upload_timestamp, processing_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (document_id, tender_id, bidder_id, doc_type, filename, file_path,
             file_hash, 0, 0.0, upload_timestamp, "error"),
        )
        return {
            "id": document_id,
            "tender_id": tender_id,
            "bidder_id": bidder_id,
            "doc_type": doc_type,
            "filename": filename,
            "file_path": file_path,
            "sha256_hash": file_hash,
            "page_count": 0,
            "avg_ocr_confidence": 0.0,
            "processing_status": "error",
            "error_message": parsed.get("message", "Unknown error"),
            "pages": [],
        }

    page_count = parsed["page_count"]
    is_scanned = parsed["is_scanned"]

    # ── Store initial document record (processing) ──
    conn.execute(
        """INSERT INTO documents (id, tender_id, bidder_id, doc_type, filename, file_path,
           sha256_hash, page_count, avg_ocr_confidence, upload_timestamp, processing_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (document_id, tender_id, bidder_id, doc_type, filename, file_path,
         file_hash, page_count, 0.0, upload_timestamp, "processing"),
    )

    # ── document_received audit event ──
    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="document_received",
        event_data={
            "document_id": document_id,
            "filename": filename,
            "doc_type": doc_type,
            "format": fmt,
            "sha256_hash": file_hash,
            "page_count": page_count,
            "is_scanned": is_scanned,
        },
        actor="system",
    )

    # ── Extract page images (format-specific) ──
    # Output dir is scoped to the document so multiple uploads of the
    # same file don't collide on disk.
    output_dir = os.path.join(os.path.dirname(file_path), "pages", document_id)

    if fmt == "pdf":
        image_paths = extract_page_images(file_path, output_dir)
    elif fmt == "image":
        image_paths = copy_image_as_page(file_path, output_dir)
    else:
        # DOCX has no rasterised images — we skip OCR entirely.
        image_paths = []

    # ── Process each page ──
    page_confidences: list[float] = []
    stored_pages: list[dict] = []

    for i in range(page_count):
        page_id = str(uuid.uuid4())
        page_number = i + 1

        image_path = image_paths[i] if i < len(image_paths) else ""

        embedded_text = (
            parsed["pages"][i]["text"] if i < len(parsed["pages"]) else ""
        )

        if fmt == "docx":
            # DOCX: digital text, no preprocessing / OCR.
            raw_text = embedded_text
            words = _text_to_word_objects(page_id, embedded_text)
            page_confidence = 0.95
            processed_image_path = ""
            dpi = 0
            processing_notes = "docx_digital_text"
        else:
            # Pre-process the page image (PDF raster or image upload).
            preprocess_result = preprocess_page_image(image_path) \
                if image_path else {
                    "processed_image_path": image_path,
                    "processing_notes": "no_image_extracted",
                    "dpi": 300,
                }
            processed_image_path = preprocess_result["processed_image_path"]
            processing_notes = preprocess_result["processing_notes"]
            dpi = preprocess_result["dpi"]

            if embedded_text.strip() and not is_scanned:
                # Digital PDF path — embedded text, synthesise word bboxes.
                raw_text = embedded_text
                words = _text_to_word_objects(page_id, embedded_text)
                page_confidence = 0.95
            else:
                # OCR path — scanned PDF or uploaded image.
                ocr_result = extract_text_from_image(processed_image_path) \
                    if processed_image_path else {
                        "raw_text": "",
                        "words": [],
                    }
                raw_text = ocr_result.get("raw_text", "")
                words = ocr_result.get("words", [])
                for w in words:
                    w["page_id"] = page_id
                page_confidence = compute_page_confidence(words)

        page_confidences.append(page_confidence)

        # Store page record
        conn.execute(
            """INSERT INTO pages (id, document_id, page_number, image_path,
               ocr_confidence, raw_text, dpi, processing_notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (page_id, document_id, page_number, processed_image_path,
             page_confidence, raw_text, dpi, processing_notes),
        )

        # Store word objects
        for word in words:
            word_id = word.get("id", str(uuid.uuid4()))
            conn.execute(
                """INSERT INTO word_objects (id, page_id, text_content,
                   x_min, y_min, x_max, y_max, confidence, source_engine)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (word_id, page_id, word["text_content"],
                 word["x_min"], word["y_min"], word["x_max"], word["y_max"],
                 word["confidence"],
                 word.get("source_engine", "tesseract")),
            )

        stored_pages.append({
            "id": page_id,
            "page_number": page_number,
            "image_path": processed_image_path,
            "ocr_confidence": page_confidence,
            "raw_text": raw_text,
            "dpi": dpi,
            "processing_notes": processing_notes,
            "is_degraded": is_degraded_page(page_confidence),
            "word_count": len(words),
        })

    avg_ocr_confidence = (
        sum(page_confidences) / len(page_confidences)
        if page_confidences else 0.0
    )

    conn.execute(
        """UPDATE documents SET avg_ocr_confidence = ?, processing_status = ?
           WHERE id = ?""",
        (avg_ocr_confidence, "complete", document_id),
    )

    # ── ocr_completed audit event (logged for every format, even DOCX
    # where it really means "text_extracted" — we keep the event name
    # stable so downstream audit consumers don't need a new case). ──
    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="ocr_completed",
        event_data={
            "document_id": document_id,
            "format": fmt,
            "page_count": page_count,
            "avg_ocr_confidence": round(avg_ocr_confidence, 4),
            "is_scanned": is_scanned,
            "degraded_pages": [
                p["page_number"] for p in stored_pages if p["is_degraded"]
            ],
        },
        actor="system",
    )

    return {
        "id": document_id,
        "tender_id": tender_id,
        "bidder_id": bidder_id,
        "doc_type": doc_type,
        "filename": filename,
        "file_path": file_path,
        "sha256_hash": file_hash,
        "page_count": page_count,
        "avg_ocr_confidence": round(avg_ocr_confidence, 4),
        "processing_status": "complete",
        "pages": stored_pages,
    }


def _text_to_word_objects(page_id: str, text: str) -> list[dict]:
    """Convert embedded text to word objects with synthetic bounding boxes.

    Used when a document has directly extractable text (digital PDF or
    DOCX), so we create word objects at high confidence with
    approximate positions. The x/y/w/h values are a deterministic
    layout grid — they're not geometrically accurate but they keep the
    word_objects schema contract (bbox + confidence) intact.
    """
    words: list[dict] = []
    x_cursor = 50.0
    y_cursor = 50.0
    line_height = 20.0
    page_width = 2480.0

    for token in text.split():
        if not token.strip():
            continue

        word_width = len(token) * 8.0
        word_height = 14.0

        if x_cursor + word_width > page_width - 50:
            x_cursor = 50.0
            y_cursor += line_height

        words.append({
            "id": str(uuid.uuid4()),
            "page_id": page_id,
            "text_content": token,
            "x_min": round(x_cursor, 1),
            "y_min": round(y_cursor, 1),
            "x_max": round(x_cursor + word_width, 1),
            "y_max": round(y_cursor + word_height, 1),
            "confidence": 0.95,
            "source_engine": "embedded",
        })

        x_cursor += word_width + 10.0

    return words
