"""FastAPI route handlers for the VerdictAI REST API.

Exports all API routers for inclusion in the main application.
"""

from api.audit import router as audit_router
from api.cpm import router as cpm_router
from api.documents import router as documents_router
from api.evaluation import router as evaluation_router
from api.hitl import router as hitl_router
from api.tenders import router as tenders_router

__all__ = [
    "audit_router",
    "cpm_router",
    "documents_router",
    "evaluation_router",
    "hitl_router",
    "tenders_router",
]
