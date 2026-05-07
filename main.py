"""VerdictAI - Explainable AI Procurement Intelligence Platform.

FastAPI application entry point with global error handling and
all API routers registered under /api/v1 prefix.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import (
    audit_router,
    cpm_router,
    documents_router,
    evaluation_router,
    hitl_router,
    tenders_router,
)
from config import settings


async def init_database():
    """Initialize the SQLite database on startup.

    Creates tables and seeds demo data if running for the first time.
    """
    from database.connection import init_db
    init_db(settings.db_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    await init_database()
    yield


app = FastAPI(
    title="VerdictAI API",
    version="0.1.0",
    description="Explainable AI Procurement Intelligence Platform API",
    lifespan=lifespan,
)

# CORS middleware — dev ports + any production frontend URL configured
# via FRONTEND_ORIGIN env var (Vercel deployment).
import os as _os

_prod_origins = [
    o.strip()
    for o in _os.getenv("FRONTEND_ORIGIN", "").split(",")
    if o.strip()
]
_allow_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    *_prod_origins,
    "https://verdict-ai-frontend-ebon.vercel.app/"
]
# If no production origin set, allow all Vercel preview URLs via regex.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Global Error Handlers ───────────────────────────────────────────────────


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions with consistent error format."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": str(uuid.uuid4()),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with consistent error format."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": str(uuid.uuid4()),
            }
        },
    )


# ─── Health Check ────────────────────────────────────────────────────────────


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


# ─── Router Includes ─────────────────────────────────────────────────────────

app.include_router(documents_router, prefix="/api/v1")
app.include_router(tenders_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(hitl_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")
app.include_router(cpm_router, prefix="/api/v1")
