"""Pydantic models for Document Intelligence layer (L1).

Covers Document, Page, and WordObject models representing
parsed PDF documents, their pages, and OCR-extracted words.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class WordObject(BaseModel):
    """A single word extracted by OCR with bounding box and confidence."""

    id: str
    page_id: str
    text_content: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    source_engine: str = "tesseract"


class Page(BaseModel):
    """A single page within a document, with OCR results."""

    id: str
    document_id: str
    page_number: int
    image_path: str
    ocr_confidence: float
    raw_text: str
    dpi: int
    processing_notes: Optional[str] = None


class Document(BaseModel):
    """A PDF document uploaded for processing."""

    id: str
    tender_id: str
    bidder_id: Optional[str] = None
    doc_type: Literal["nit", "corrigendum", "bidder_submission", "certificate"]
    filename: str
    file_path: str
    sha256_hash: str
    page_count: int
    avg_ocr_confidence: float
    upload_timestamp: str
    processing_status: Literal["pending", "processing", "complete", "error"]
