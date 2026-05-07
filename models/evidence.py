"""Pydantic models for Evidence Extraction layer (L3).

Covers Evidence and EntityMatchResult models representing
extracted evidence and entity name matching results.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class Evidence(BaseModel):
    """Structured evidence extracted for a (bidder, criterion) pair."""

    id: str
    tender_id: str
    bidder_id: str
    criterion_id: str
    extracted_value: Optional[dict] = None
    source_document_id: str
    source_page_number: int
    source_bbox: dict
    ocr_confidence: float
    extraction_confidence: float
    entity_match_flag: bool


class EntityMatchResult(BaseModel):
    """Result of fuzzy entity name matching between registered and extracted names."""

    registered_name: str
    extracted_name: str
    similarity_score: float
    is_match: bool
    mismatch_type: Optional[Literal["parent_company", "abbreviation", "different_entity"]] = None
    requires_review: bool
