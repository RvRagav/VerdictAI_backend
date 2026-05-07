"""Pydantic models for ETS Builder layer (L2).

Covers Criterion model and CriterionType enum representing
extracted eligibility criteria from tender documents.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CriterionType(str, Enum):
    """Classification of criterion evaluation logic."""

    numeric_threshold = "numeric_threshold"
    categorical_presence = "categorical_presence"
    temporal_recency = "temporal_recency"
    composite = "composite"
    qualitative_assessment = "qualitative_assessment"


class Criterion(BaseModel):
    """An eligibility criterion extracted from the Effective Tender Specification."""

    id: str
    tender_id: str
    criterion_text: str
    criterion_type: CriterionType
    threshold_value: Optional[dict] = None
    gfr_override_permitted: bool
    gfr_rule_number: Optional[str] = None
    source_document_id: str
    source_clause_ref: str
    amendment_history: list = Field(default_factory=list)
    is_mandatory: bool
    acceptable_evidence_types: list = Field(default_factory=list)
    measurement_period: Optional[str] = None
    status: Literal["extracted", "reviewed", "approved"]
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
