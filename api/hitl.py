"""HITL Review API endpoints for VerdictAI.

Handles the Human-in-the-Loop review queue, card data retrieval,
officer decisions, and second-officer confirmation.

Endpoints:
- GET /tenders/{id}/hitl/queue
- GET /hitl/{evaluation_id}/card
- POST /hitl/{evaluation_id}/decide
- POST /hitl/{evaluation_id}/second-officer

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 16.1
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.database.connection import get_db
from backend.services.hitl_service import (
    get_hitl_card,
    get_hitl_queue,
    process_decision,
    process_second_officer,
)

router = APIRouter(tags=["hitl"])


# ─── Request Models ──────────────────────────────────────────────────────────


class DecisionRequest(BaseModel):
    decision: str  # "confirm" or "override"
    officer_id: str
    reason: str | None = None
    reason_text: str | None = None


class SecondOfficerRequest(BaseModel):
    officer_id: str
    decision: str  # "approve" or "reject"


# ─── Dependency ──────────────────────────────────────────────────────────────


def _get_conn():
    """Dependency that provides a database connection."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/tenders/{tender_id}/hitl/queue")
async def get_queue(
    tender_id: str,
    route: str | None = Query(None, description="Filter by route: hitl_review or mandatory_review"),
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get pending HITL cases for a tender.

    Returns evaluations ordered by priority: mandatory_review first,
    then by confidence ascending.
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

    # Validate route filter
    if route and route not in ("hitl_review", "mandatory_review"):
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "INVALID_ROUTE_FILTER",
                "message": "route must be 'hitl_review' or 'mandatory_review'",
            }
        })

    queue = get_hitl_queue(conn, tender_id, route_filter=route)
    return queue


@router.get("/hitl/{evaluation_id}/card")
async def get_card(
    evaluation_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get full HITL review card data for an evaluation.

    Returns the 5-component card: criterion details, evidence with bbox,
    system analysis, CPM precedents, and decision options.
    """
    try:
        card = get_hitl_card(conn, evaluation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "EVALUATION_NOT_FOUND",
                "message": str(e),
            }
        })

    return card


@router.post("/hitl/{evaluation_id}/decide")
async def submit_decision(
    evaluation_id: str,
    request: DecisionRequest,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Submit an officer decision on a pending evaluation.

    Validates the decision, enforces GFR rules, records the decision,
    and stores a CPM precedent.
    """
    # Validate decision value
    if request.decision not in ("confirm", "override"):
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "INVALID_DECISION",
                "message": "decision must be 'confirm' or 'override'",
            }
        })

    try:
        result = process_decision(
            conn=conn,
            evaluation_id=evaluation_id,
            decision=request.decision,
            officer_id=request.officer_id,
            reason=request.reason,
            reason_text=request.reason_text,
        )
        conn.commit()
    except ValueError as e:
        error_msg = str(e)

        # Map specific errors to appropriate HTTP codes
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={
                "error": {
                    "code": "EVALUATION_NOT_FOUND",
                    "message": error_msg,
                }
            })
        elif "not pending" in error_msg:
            raise HTTPException(status_code=409, detail={
                "error": {
                    "code": "EVALUATION_NOT_PENDING",
                    "message": error_msg,
                }
            })
        elif "Override not permitted" in error_msg:
            raise HTTPException(status_code=409, detail={
                "error": {
                    "code": "OVERRIDE_NOT_PERMITTED",
                    "message": error_msg,
                }
            })
        elif "structured reason" in error_msg:
            raise HTTPException(status_code=422, detail={
                "error": {
                    "code": "REASON_REQUIRED",
                    "message": error_msg,
                }
            })
        else:
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "DECISION_ERROR",
                    "message": error_msg,
                }
            })

    return {
        "status": result.get("status", "resolved"),
        "evaluation_id": evaluation_id,
        "decision": request.decision,
    }


@router.post("/hitl/{evaluation_id}/second-officer")
async def submit_second_officer(
    evaluation_id: str,
    request: SecondOfficerRequest,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Submit second-officer confirmation for a pending override.

    Required when an override is on a GFR-adjacent criterion.
    """
    if request.decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "INVALID_DECISION",
                "message": "decision must be 'approve' or 'reject'",
            }
        })

    try:
        result = process_second_officer(
            conn=conn,
            evaluation_id=evaluation_id,
            officer_id=request.officer_id,
            decision=request.decision,
        )
        conn.commit()
    except ValueError as e:
        error_msg = str(e)

        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={
                "error": {
                    "code": "EVALUATION_NOT_FOUND",
                    "message": error_msg,
                }
            })
        elif "does not require" in error_msg:
            raise HTTPException(status_code=409, detail={
                "error": {
                    "code": "SECOND_OFFICER_NOT_REQUIRED",
                    "message": error_msg,
                }
            })
        elif "must be different" in error_msg:
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "SAME_OFFICER",
                    "message": error_msg,
                }
            })
        else:
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "SECOND_OFFICER_ERROR",
                    "message": error_msg,
                }
            })

    return result
