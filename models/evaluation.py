"""Pydantic models for Evaluation Engine layer (L4).

Covers Verdict enum, Route enum, RoutingDecision, and Evaluation models
representing the evaluation pipeline outputs and routing logic.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Evaluation verdict for a (bidder, criterion) pair."""

    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class Route(str, Enum):
    """Routing disposition for an evaluation case."""

    auto_commit = "auto_commit"
    hitl_review = "hitl_review"
    mandatory_review = "mandatory_review"


class RoutingDecision(BaseModel):
    """Output of the Confidence Router determining case disposition."""

    route: Route
    confidence: float
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    gfr_override_permitted: bool
    is_mandatory_criterion: bool


class Evaluation(BaseModel):
    """A complete evaluation record for a (bidder, criterion) pair."""

    id: str
    tender_id: str
    bidder_id: str
    criterion_id: str
    verdict: Verdict
    confidence: float
    evaluation_method: str
    route: Route
    routing_reason: Optional[str] = None
    extracted_value: Optional[dict] = None
    source_document_id: Optional[str] = None
    source_page_number: Optional[int] = None
    source_bbox: Optional[dict] = None
    ocr_confidence: Optional[float] = None
    extraction_confidence: Optional[float] = None
    entity_match_flag: bool = False
    officer_decision: Optional[Literal["confirmed", "overridden"]] = None
    officer_id: Optional[str] = None
    officer_reason: Optional[str] = None
    officer_decision_timestamp: Optional[str] = None
    second_officer_id: Optional[str] = None
    second_officer_timestamp: Optional[str] = None
    status: Literal["pending", "auto_committed", "pending_review", "resolved"]
    created_at: str
    resolved_at: Optional[str] = None
