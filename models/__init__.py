"""Pydantic data models for VerdictAI domain objects.

Exports all models and enums for use across the application.
"""

from backend.models.audit import AuditEvent, AuditEventType
from backend.models.cpm import CPMEntry
from backend.models.criterion import Criterion, CriterionType
from backend.models.document import Document, Page, WordObject
from backend.models.evaluation import Evaluation, Route, RoutingDecision, Verdict
from backend.models.evidence import EntityMatchResult, Evidence

__all__ = [
    # Document Intelligence (L1)
    "Document",
    "Page",
    "WordObject",
    # ETS Builder (L2)
    "Criterion",
    "CriterionType",
    # Evidence Extraction (L3)
    "Evidence",
    "EntityMatchResult",
    # Evaluation Engine (L4)
    "Verdict",
    "Route",
    "RoutingDecision",
    "Evaluation",
    # Audit Ledger (L5)
    "AuditEvent",
    "AuditEventType",
    # CPM
    "CPMEntry",
]
