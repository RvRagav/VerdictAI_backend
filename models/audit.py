"""Pydantic models for Audit Ledger layer (L5).

Covers AuditEventType enum and AuditEvent model representing
the immutable append-only audit trail.
"""

from enum import Enum

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    """Types of events recorded in the audit ledger."""

    document_received = "document_received"
    ocr_completed = "ocr_completed"
    corrigendum_linked = "corrigendum_linked"
    schema_approved = "schema_approved"
    debarment_checked = "debarment_checked"
    evidence_extracted = "evidence_extracted"
    verdict_computed = "verdict_computed"
    case_routed = "case_routed"
    officer_decision = "officer_decision"
    report_generated = "report_generated"
    override_attempted = "override_attempted"


class AuditEvent(BaseModel):
    """A single immutable entry in the audit ledger with hash chain linkage."""

    id: int
    tender_id: str
    event_type: AuditEventType
    event_data: dict = Field(default_factory=dict)
    actor: str
    timestamp: str
    prev_hash: str
    entry_hash: str
