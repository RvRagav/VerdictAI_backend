"""Evaluation API endpoints for VerdictAI.

Handles evaluation triggering, status retrieval, summary stats,
debarment checks, and bidder debarment status.

Endpoints:
- POST /tenders/{id}/evaluate
- GET /tenders/{id}/evaluations
- GET /evaluations/{id}
- GET /tenders/{id}/summary
- POST /tenders/{id}/debarment-check
- GET /bidders/{id}/debarment

Requirements: 5.1, 7.6, 16.1
"""

import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from database.connection import get_db
from layers.l4_evaluation import evaluate_all_bidders
from layers.l5_audit import append_audit_event
from services.debarment_service import check_debarment

router = APIRouter(tags=["evaluation"])


def _get_conn():
    """Dependency that provides a database connection."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


@router.post("/tenders/{tender_id}/evaluate")
async def trigger_evaluation(
    tender_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Trigger evaluation for all bidders in a tender.

    Enforces state guards: schema must be approved and debarment must
    be checked (tender in SCHEMA_APPROVED or later state).
    """
    conn.row_factory = sqlite3.Row

    tender = conn.execute(
        "SELECT * FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()

    if not tender:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "TENDER_NOT_FOUND",
                "message": f"Tender {tender_id} not found",
            }
        })

    # State guard: must be SCHEMA_APPROVED or DEBARMENT_CHECK or EVALUATING
    allowed_states = ("SCHEMA_APPROVED", "DEBARMENT_CHECK")
    if tender["status"] not in allowed_states:
        if tender["status"] in ("DOCUMENTS_UPLOADED", "PROCESSING_OCR", "OCR_COMPLETE",
                                 "EXTRACTING_CRITERIA", "SCHEMA_PENDING_REVIEW"):
            raise HTTPException(status_code=409, detail={
                "error": {
                    "code": "SCHEMA_NOT_APPROVED",
                    "message": "Cannot start evaluation: criterion schema has not been approved",
                    "details": {
                        "tender_id": tender_id,
                        "current_status": tender["status"],
                    },
                }
            })
        raise HTTPException(status_code=409, detail={
            "error": {
                "code": "INVALID_STATE_TRANSITION",
                "message": f"Cannot evaluate tender in state '{tender['status']}'",
                "details": {
                    "current_status": tender["status"],
                    "allowed_states": list(allowed_states),
                },
            }
        })

    # Check bidders exist
    bidder_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM bidders WHERE tender_id = ? AND status != 'excluded'",
        (tender_id,),
    ).fetchone()["cnt"]

    if bidder_count == 0:
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "NO_BIDDERS",
                "message": "No bidders available for evaluation",
            }
        })

    # Update tender status to EVALUATING
    conn.execute(
        "UPDATE tenders SET status = 'EVALUATING', updated_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), tender_id),
    )

    # Run evaluation
    results = evaluate_all_bidders(conn, tender_id)

    # Determine final state
    pending_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM evaluations WHERE tender_id = ? AND status IN ('pending_review', 'pending_second_officer')",
        (tender_id,),
    ).fetchone()["cnt"]

    if pending_count > 0:
        new_status = "HITL_PENDING"
    else:
        new_status = "EVALUATION_COMPLETE"

    conn.execute(
        "UPDATE tenders SET status = ?, updated_at = ? WHERE id = ?",
        (new_status, datetime.now(timezone.utc).isoformat(), tender_id),
    )
    conn.commit()

    return {
        "status": "evaluating",
        "bidder_count": bidder_count,
        "results_summary": {
            "total_evaluations": sum(len(r.get("evaluations", [])) for r in results),
            "debarment_flagged": sum(1 for r in results if r["status"] == "debarment_flagged"),
        },
    }


@router.get("/tenders/{tender_id}/evaluations")
async def get_evaluations(
    tender_id: str,
    bidder_id: str | None = Query(None),
    status: str | None = Query(None),
    route: str | None = Query(None),
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get all evaluations for a tender with optional filters."""
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

    query = """
        SELECT e.*, b.company_name as bidder_name, c.criterion_text
        FROM evaluations e
        JOIN bidders b ON e.bidder_id = b.id
        JOIN criteria c ON e.criterion_id = c.id
        WHERE e.tender_id = ?
    """
    params: list = [tender_id]

    if bidder_id:
        query += " AND e.bidder_id = ?"
        params.append(bidder_id)
    if status:
        query += " AND e.status = ?"
        params.append(status)
    if route:
        query += " AND e.route = ?"
        params.append(route)

    query += " ORDER BY e.created_at DESC"

    rows = conn.execute(query, params).fetchall()

    return [
        {
            "id": row["id"],
            "bidder_id": row["bidder_id"],
            "bidder_name": row["bidder_name"],
            "criterion_id": row["criterion_id"],
            "criterion_text": row["criterion_text"],
            "verdict": row["verdict"],
            "confidence": row["confidence"],
            "route": row["route"],
            "status": row["status"],
            "officer_decision": row["officer_decision"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


@router.get("/evaluations/{evaluation_id}")
async def get_evaluation_detail(
    evaluation_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get single evaluation detail with full evidence and reasoning."""
    conn.row_factory = sqlite3.Row

    evaluation = conn.execute(
        "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
    ).fetchone()

    if not evaluation:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "EVALUATION_NOT_FOUND",
                "message": f"Evaluation {evaluation_id} not found",
            }
        })

    return {
        "id": evaluation["id"],
        "tender_id": evaluation["tender_id"],
        "bidder_id": evaluation["bidder_id"],
        "criterion_id": evaluation["criterion_id"],
        "verdict": evaluation["verdict"],
        "confidence": evaluation["confidence"],
        "evaluation_method": evaluation["evaluation_method"],
        "route": evaluation["route"],
        "routing_reason": evaluation["routing_reason"],
        "extracted_value": json.loads(evaluation["extracted_value"]) if evaluation["extracted_value"] else None,
        "source_document_id": evaluation["source_document_id"],
        "source_page_number": evaluation["source_page_number"],
        "source_bbox": json.loads(evaluation["source_bbox"]) if evaluation["source_bbox"] else None,
        "ocr_confidence": evaluation["ocr_confidence"],
        "extraction_confidence": evaluation["extraction_confidence"],
        "entity_match_flag": bool(evaluation["entity_match_flag"]),
        "officer_decision": evaluation["officer_decision"],
        "officer_id": evaluation["officer_id"],
        "officer_reason": evaluation["officer_reason"],
        "officer_decision_timestamp": evaluation["officer_decision_timestamp"],
        "second_officer_id": evaluation["second_officer_id"],
        "second_officer_timestamp": evaluation["second_officer_timestamp"],
        "status": evaluation["status"],
        "created_at": evaluation["created_at"],
        "resolved_at": evaluation["resolved_at"],
    }


@router.get("/tenders/{tender_id}/summary")
async def get_evaluation_summary(
    tender_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get evaluation summary statistics for a tender."""
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

    # Overall counts
    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM evaluations WHERE tender_id = ?",
        (tender_id,),
    ).fetchone()["cnt"]

    auto_committed = conn.execute(
        "SELECT COUNT(*) as cnt FROM evaluations WHERE tender_id = ? AND status = 'auto_committed'",
        (tender_id,),
    ).fetchone()["cnt"]

    pending_review = conn.execute(
        "SELECT COUNT(*) as cnt FROM evaluations WHERE tender_id = ? AND status IN ('pending_review', 'pending_second_officer')",
        (tender_id,),
    ).fetchone()["cnt"]

    completed = conn.execute(
        "SELECT COUNT(*) as cnt FROM evaluations WHERE tender_id = ? AND status = 'resolved'",
        (tender_id,),
    ).fetchone()["cnt"]

    # Per-bidder breakdown
    by_bidder = conn.execute(
        """SELECT b.id, b.company_name,
                  COUNT(e.id) as total,
                  SUM(CASE WHEN e.verdict = 'PASS' THEN 1 ELSE 0 END) as pass_count,
                  SUM(CASE WHEN e.verdict = 'FAIL' THEN 1 ELSE 0 END) as fail_count,
                  SUM(CASE WHEN e.verdict = 'REVIEW' THEN 1 ELSE 0 END) as review_count
           FROM bidders b
           LEFT JOIN evaluations e ON e.bidder_id = b.id AND e.tender_id = ?
           WHERE b.tender_id = ?
           GROUP BY b.id""",
        (tender_id, tender_id),
    ).fetchall()

    return {
        "total": total,
        "auto_committed": auto_committed,
        "pending_review": pending_review,
        "completed": completed,
        "by_bidder": [
            {
                "bidder_id": row["id"],
                "company_name": row["company_name"],
                "total": row["total"],
                "pass_count": row["pass_count"],
                "fail_count": row["fail_count"],
                "review_count": row["review_count"],
            }
            for row in by_bidder
        ],
    }


@router.post("/tenders/{tender_id}/debarment-check")
async def run_debarment_check(
    tender_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Run debarment check for all bidders in a tender."""
    conn.row_factory = sqlite3.Row

    tender = conn.execute(
        "SELECT * FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()

    if not tender:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "TENDER_NOT_FOUND",
                "message": f"Tender {tender_id} not found",
            }
        })

    # Fetch all bidders
    bidders = conn.execute(
        "SELECT * FROM bidders WHERE tender_id = ?", (tender_id,)
    ).fetchall()

    flagged = []
    clear = []

    for bidder in bidders:
        result = check_debarment(
            conn=conn,
            company_name=bidder["company_name"],
            pan_number=bidder["pan_number"],
        )

        # Update bidder debarment status
        debarment_status = "flagged" if result["is_debarred"] else "clear"
        conn.execute(
            "UPDATE bidders SET debarment_status = ?, debarment_check_timestamp = ? WHERE id = ?",
            (debarment_status, datetime.now(timezone.utc).isoformat(), bidder["id"]),
        )

        # Log audit event
        append_audit_event(
            conn=conn,
            tender_id=tender_id,
            event_type="debarment_checked",
            event_data={
                "bidder_id": bidder["id"],
                "company_name": bidder["company_name"],
                "is_debarred": result["is_debarred"],
                "check_method": result["check_method"],
            },
            actor="system",
        )

        if result["is_debarred"]:
            flagged.append({
                "bidder_id": bidder["id"],
                "company_name": bidder["company_name"],
                "matches": result["matches"],
            })
        else:
            clear.append({
                "bidder_id": bidder["id"],
                "company_name": bidder["company_name"],
            })

    conn.commit()

    return {
        "checked": len(bidders),
        "flagged": flagged,
        "clear": clear,
    }


@router.get("/bidders/{bidder_id}/debarment")
async def get_bidder_debarment(
    bidder_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get debarment status for a specific bidder."""
    conn.row_factory = sqlite3.Row

    bidder = conn.execute(
        "SELECT * FROM bidders WHERE id = ?", (bidder_id,)
    ).fetchone()

    if not bidder:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "BIDDER_NOT_FOUND",
                "message": f"Bidder {bidder_id} not found",
            }
        })

    result = {
        "bidder_id": bidder["id"],
        "company_name": bidder["company_name"],
        "status": bidder["debarment_status"],
        "check_timestamp": bidder["debarment_check_timestamp"],
    }

    # If flagged, get match details
    if bidder["debarment_status"] == "flagged":
        debarment_result = check_debarment(
            conn=conn,
            company_name=bidder["company_name"],
            pan_number=bidder["pan_number"],
        )
        result["matched_entity"] = debarment_result["matches"][0] if debarment_result["matches"] else None
        result["match_details"] = debarment_result["matches"]

    return result
