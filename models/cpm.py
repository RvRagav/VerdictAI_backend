"""Pydantic models for Criterion Precedent Memory (CPM).

Covers CPMEntry model representing stored officer interpretation
precedents for institutional memory and reuse.
"""

from pydantic import BaseModel


class CPMEntry(BaseModel):
    """A precedent entry in the Criterion Precedent Memory store."""

    id: str
    criterion_text: str
    resolved_interpretation: str
    department: str
    tender_category: str
    verdict: str
    officer_action: str
    officer_id: str
    tender_id: str
    criterion_id: str
    created_at: str
