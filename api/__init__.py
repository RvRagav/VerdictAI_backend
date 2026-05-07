"""FastAPI route handlers for the VerdictAI REST API.

Exports all API routers for inclusion in the main application.
"""

from backend.api.audit import router as audit_router
from backend.api.cpm import router as cpm_router
from backend.api.documents import router as documents_router
from backend.api.evaluation import router as evaluation_router
from backend.api.hitl import router as hitl_router
from backend.api.tenders import router as tenders_router

__all__ = [
    "audit_router",
    "cpm_router",
    "documents_router",
    "evaluation_router",
    "hitl_router",
    "tenders_router",
]
