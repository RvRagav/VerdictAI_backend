"""Audit and Report API endpoints for VerdictAI.

Handles audit trail retrieval, report generation, report download,
and reproducibility verification.

Endpoints:
- GET /tenders/{id}/audit
- POST /tenders/{id}/report
- GET /reports/{id}/download
- POST /tenders/{id}/reproduce

Requirements: 12.1, 13.1, 18.1, 18.2, 18.3, 16.1
"""

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.database.connection import get_db
from backend.layers.l5_audit import get_audit_trail
from backend.services.report_service import generate_report, get_report_download_path
from backend.services.reproducibility import (
    reproduce_evaluation as _reproduce_evaluation_service,
    verify_reproducibility,
)

router = APIRouter(tags=["audit"])


# ─── Request Models ──────────────────────────────────────────────────────────


class ReportRequest(BaseModel):
    officer_id: str


class ReproduceRequest(BaseModel):
    report_id: str | None = None


# ─── Dependency ──────────────────────────────────────────────────────────────


def _get_conn():
    """Dependency that provides a database connection."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/tenders/{tender_id}/audit")
async def get_audit(
    tender_id: str,
    event_type: str | None = Query(None, description="Filter by event type"),
    from_date: str | None = Query(None, alias="from", description="ISO 8601 start date"),
    to_date: str | None = Query(None, alias="to", description="ISO 8601 end date"),
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get audit trail for a tender with optional filtering.

    Supports filtering by event_type and date range.
    """
    conn.row_factory = sqlite3.Row

    # Verify tender exists
    tender = conn.execute(
        "SELECT id FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    if not tender:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "TENDER_NOT_FOUND",
                "message": f"Tender {tender_id} not found",
            }
        })

    trail = get_audit_trail(
        conn=conn,
        tender_id=tender_id,
        event_type=event_type,
        from_date=from_date,
        to_date=to_date,
    )

    # Return full audit trail with hash chain details so the UI can
    # display tamper-evident prev/entry hashes.
    return [
        {
            "id": event["id"],
            "event_type": event["event_type"],
            "actor": event["actor"],
            "timestamp": event["timestamp"],
            "event_data": event["event_data"],
            "data_summary": _summarise_event_data(event["event_data"]),
            "prev_hash": event.get("prev_hash"),
            "entry_hash": event.get("entry_hash"),
        }
        for event in trail
    ]


@router.post("/tenders/{tender_id}/report", status_code=201)
async def create_report(
    tender_id: str,
    request: ReportRequest,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Generate an evaluation report PDF for a tender.

    Computes SHA-256 hash of the audit trail and generates a report.
    """
    conn.row_factory = sqlite3.Row

    # Verify tender exists
    tender = conn.execute(
        "SELECT id FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    if not tender:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "TENDER_NOT_FOUND",
                "message": f"Tender {tender_id} not found",
            }
        })

    try:
        report = generate_report(
            conn=conn,
            tender_id=tender_id,
            officer_id=request.officer_id,
        )
        conn.commit()
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "REPORT_GENERATION_ERROR",
                "message": str(e),
            }
        })

    return {
        "report_id": report["report_id"],
        "download_url": f"/api/v1/reports/{report['report_id']}/download",
        "sha256_hash": report["sha256_hash"],
        "generated_at": report["generated_at"],
    }


@router.get("/reports/{report_id}/download")
async def download_report(report_id: str):
    """Download a generated report file.

    Returns PDF if available, otherwise JSON.
    """
    download_path = get_report_download_path(report_id)

    if not download_path:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "REPORT_NOT_FOUND",
                "message": f"Report {report_id} not found",
            }
        })

    # Determine media type
    if download_path.suffix == ".pdf":
        media_type = "application/pdf"
    else:
        media_type = "application/json"

    return FileResponse(
        path=str(download_path),
        media_type=media_type,
        filename=f"verdict_ai_report_{report_id}{download_path.suffix}",
    )


@router.post("/tenders/{tender_id}/reproduce")
async def trigger_reproduce(
    tender_id: str,
    request: ReproduceRequest,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Trigger reproduction verification for a completed evaluation.

    Verifies that the evaluation can be reproduced from stored inputs:
    1. Document SHA-256 hashes match
    2. LLM Stub version is consistent
    3. Audit trail hash chain is intact
    4. All required inputs are stored
    5. Evaluation records are consistent with audit trail

    Returns match status and any differences found.
    """
    conn.row_factory = sqlite3.Row

    # Verify tender exists
    tender = conn.execute(
        "SELECT id FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    if not tender:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "TENDER_NOT_FOUND",
                "message": f"Tender {tender_id} not found",
            }
        })

    try:
        result = _reproduce_evaluation_service(
            conn=conn,
            tender_id=tender_id,
            original_report_id=request.report_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail={
            "error": {
                "code": "REPRODUCIBILITY_ERROR",
                "message": str(e),
            }
        })

    return result


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _summarise_event_data(event_data: dict | str) -> str:
    """Create a brief summary of event data for list views."""
    if isinstance(event_data, str):
        return event_data[:100]

    if not event_data:
        return ""

    # Extract key fields for summary
    parts = []
    if "bidder_id" in event_data:
        parts.append(f"bidder={event_data['bidder_id'][:8]}...")
    if "criterion_id" in event_data:
        parts.append(f"criterion={event_data['criterion_id'][:8]}...")
    if "verdict" in event_data:
        parts.append(f"verdict={event_data['verdict']}")
    if "decision" in event_data:
        parts.append(f"decision={event_data['decision']}")
    if "officer_id" in event_data:
        parts.append(f"officer={event_data['officer_id']}")
    if "is_debarred" in event_data:
        parts.append(f"debarred={event_data['is_debarred']}")
    if "report_id" in event_data:
        parts.append(f"report={event_data['report_id'][:8]}...")

    return "; ".join(parts) if parts else str(event_data)[:100]
