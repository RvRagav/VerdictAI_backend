"""Document API endpoints for VerdictAI.

Handles PDF document upload, retrieval, and page/word data access.

Endpoints:
- POST /documents/upload
- GET /documents/{id}
- GET /documents/{id}/pages
- GET /documents/{id}/pages/{page_num}/words

Requirements: 1.1, 16.1
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.database.connection import get_db

router = APIRouter(prefix="/documents", tags=["documents"])


# Extensions that L1 can ingest. Kept here (rather than imported from L1)
# so this module has no heavy imports at module load.
ACCEPTED_EXTENSIONS: tuple[str, ...] = (
    ".pdf", ".docx",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
)


def _has_accepted_extension(filename: str) -> bool:
    name = filename.lower()
    return any(name.endswith(ext) for ext in ACCEPTED_EXTENSIONS)


def _get_conn():
    """Dependency that provides a database connection."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    tender_id: str = Form(...),
    doc_type: str = Form(...),
    bidder_id: str = Form(None),
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Upload a document (PDF, DOCX, or image).

    Accepts multipart file upload with metadata. Computes SHA-256 hash
    for deduplication - re-uploading the same file returns the existing record.
    """
    if not file.filename or not _has_accepted_extension(file.filename):
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "INVALID_FILE_TYPE",
                "message": (
                    "Accepted file types: "
                    + ", ".join(ACCEPTED_EXTENSIONS)
                ),
            }
        })

    valid_doc_types = ("nit", "corrigendum", "bidder_submission", "certificate")
    if doc_type not in valid_doc_types:
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "INVALID_DOC_TYPE",
                "message": f"doc_type must be one of: {', '.join(valid_doc_types)}",
            }
        })

    # Read file content
    content = await file.read()
    sha256_hash = hashlib.sha256(content).hexdigest()

    # Check for duplicate by hash (idempotent upload)
    conn.row_factory = sqlite3.Row
    existing = conn.execute(
        "SELECT * FROM documents WHERE sha256_hash = ? AND tender_id = ?",
        (sha256_hash, tender_id),
    ).fetchone()

    if existing:
        return {
            "id": existing["id"],
            "filename": existing["filename"],
            "page_count": existing["page_count"],
            "status": existing["processing_status"],
            "sha256_hash": existing["sha256_hash"],
            "message": "Document already uploaded (deduplicated by hash)",
        }

    # Save file to disk
    uploads_dir = Path("uploads") / tender_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / f"{sha256_hash}_{file.filename}"
    file_path.write_bytes(content)

    # Create document record
    doc_id = str(uuid.uuid4())
    upload_timestamp = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO documents
            (id, tender_id, bidder_id, doc_type, filename, file_path,
             sha256_hash, page_count, avg_ocr_confidence, upload_timestamp,
             processing_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id, tender_id, bidder_id, doc_type, file.filename,
            str(file_path), sha256_hash, 0, 0.0, upload_timestamp, "pending",
        ),
    )
    conn.commit()

    return {
        "id": doc_id,
        "filename": file.filename,
        "page_count": 0,
        "status": "pending",
        "sha256_hash": sha256_hash,
    }


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get document metadata by ID."""
    conn.row_factory = sqlite3.Row
    doc = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (document_id,)
    ).fetchone()

    if not doc:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "DOCUMENT_NOT_FOUND",
                "message": f"Document {document_id} not found",
            }
        })

    return {
        "id": doc["id"],
        "tender_id": doc["tender_id"],
        "bidder_id": doc["bidder_id"],
        "doc_type": doc["doc_type"],
        "filename": doc["filename"],
        "sha256_hash": doc["sha256_hash"],
        "page_count": doc["page_count"],
        "avg_ocr_confidence": doc["avg_ocr_confidence"],
        "status": doc["processing_status"],
        "upload_timestamp": doc["upload_timestamp"],
    }


@router.get("/{document_id}/pages")
async def get_document_pages(
    document_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """List pages with OCR data for a document."""
    conn.row_factory = sqlite3.Row

    # Verify document exists
    doc = conn.execute(
        "SELECT id FROM documents WHERE id = ?", (document_id,)
    ).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "DOCUMENT_NOT_FOUND",
                "message": f"Document {document_id} not found",
            }
        })

    pages = conn.execute(
        """SELECT p.*, COUNT(w.id) as word_count
           FROM pages p
           LEFT JOIN word_objects w ON w.page_id = p.id
           WHERE p.document_id = ?
           GROUP BY p.id
           ORDER BY p.page_number""",
        (document_id,),
    ).fetchall()

    return [
        {
            "page_number": page["page_number"],
            "ocr_confidence": page["ocr_confidence"],
            "image_path": page["image_path"],
            "word_count": page["word_count"],
            "dpi": page["dpi"],
        }
        for page in pages
    ]


@router.get("/{document_id}/pages/{page_num}/words")
async def get_page_words(
    document_id: str,
    page_num: int,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get word objects with bounding boxes for a specific page."""
    conn.row_factory = sqlite3.Row

    # Find the page
    page = conn.execute(
        "SELECT id FROM pages WHERE document_id = ? AND page_number = ?",
        (document_id, page_num),
    ).fetchone()

    if not page:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "PAGE_NOT_FOUND",
                "message": f"Page {page_num} not found for document {document_id}",
            }
        })

    words = conn.execute(
        """SELECT text_content, x_min, y_min, x_max, y_max, confidence
           FROM word_objects WHERE page_id = ?
           ORDER BY y_min, x_min""",
        (page["id"],),
    ).fetchall()

    return [
        {
            "text": word["text_content"],
            "bbox": {
                "x_min": word["x_min"],
                "y_min": word["y_min"],
                "x_max": word["x_max"],
                "y_max": word["y_max"],
            },
            "confidence": word["confidence"],
        }
        for word in words
    ]
