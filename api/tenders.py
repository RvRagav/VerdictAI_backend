"""Tender and ETS/Schema Review API endpoints for VerdictAI.

Handles tender CRUD, processing triggers, criteria management,
schema approval, and corrigendum diffs.

Endpoints:
- POST /tenders
- GET /tenders
- GET /tenders/{id}
- POST /tenders/{id}/process
- GET /tenders/{id}/status
- GET /tenders/{id}/criteria
- PUT /tenders/{id}/criteria/{cid}
- GET /tenders/{id}/criteria/{cid}/diff
- POST /tenders/{id}/schema/approve
- GET /tenders/{id}/criteria/{cid}/cpm

Requirements: 4.1, 4.2, 4.3, 4.5, 16.1
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.database.connection import get_db
from backend.layers.l5_audit import append_audit_event
from backend.services.cpm_service import search_cpm_precedents

router = APIRouter(prefix="/tenders", tags=["tenders"])


# ─── Request Models ──────────────────────────────────────────────────────────


class CreateTenderRequest(BaseModel):
    title: str
    department: str
    category: str


class CreateBidderRequest(BaseModel):
    company_name: str
    pan_number: str | None = None
    registration_number: str | None = None


class UpdateCriterionRequest(BaseModel):
    criterion_text: str | None = None
    threshold_value: str | None = None
    criterion_type: str | None = None


class SchemaApproveRequest(BaseModel):
    officer_id: str


# ─── Dependency ──────────────────────────────────────────────────────────────


def _get_conn():
    """Dependency that provides a database connection."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


# ─── Valid State Transitions ─────────────────────────────────────────────────

VALID_TRANSITIONS = {
    "DOCUMENTS_UPLOADED": ["PROCESSING_OCR"],
    "PROCESSING_OCR": ["OCR_COMPLETE"],
    "OCR_COMPLETE": ["EXTRACTING_CRITERIA"],
    "EXTRACTING_CRITERIA": ["SCHEMA_PENDING_REVIEW"],
    "SCHEMA_PENDING_REVIEW": ["SCHEMA_APPROVED"],
    "SCHEMA_APPROVED": ["DEBARMENT_CHECK"],
    "DEBARMENT_CHECK": ["DEBARMENT_FLAGGED", "EVALUATING"],
    "DEBARMENT_FLAGGED": ["EVALUATING", "BIDDER_EXCLUDED"],
    "EVALUATING": ["VERDICTS_COMPUTED"],
    "VERDICTS_COMPUTED": ["HITL_PENDING", "EVALUATION_COMPLETE"],
    "HITL_PENDING": ["EVALUATION_COMPLETE"],
    "EVALUATION_COMPLETE": ["REPORT_GENERATED"],
}


def _validate_state_transition(current_status: str, target_status: str) -> bool:
    """Check if a state transition is valid."""
    allowed = VALID_TRANSITIONS.get(current_status, [])
    return target_status in allowed


# ─── Tender CRUD Endpoints ───────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_tender(
    request: CreateTenderRequest,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Create a new tender evaluation."""
    tender_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO tenders (id, title, department, category, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tender_id, request.title, request.department, request.category,
         "DOCUMENTS_UPLOADED", created_at, created_at),
    )
    conn.commit()

    return {
        "id": tender_id,
        "title": request.title,
        "status": "DOCUMENTS_UPLOADED",
        "created_at": created_at,
    }


@router.get("/{tender_id}/bidders")
async def list_bidders(
    tender_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """List all bidders for a tender."""
    conn.row_factory = sqlite3.Row
    tender = conn.execute(
        "SELECT id FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    if not tender:
        raise HTTPException(status_code=404, detail={"error": {
            "code": "TENDER_NOT_FOUND",
            "message": f"Tender {tender_id} not found",
        }})

    rows = conn.execute(
        "SELECT * FROM bidders WHERE tender_id = ? ORDER BY company_name",
        (tender_id,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "tender_id": r["tender_id"],
            "company_name": r["company_name"],
            "pan_number": r["pan_number"],
            "registration_number": r["registration_number"],
            "status": r["status"],
            "debarment_status": r["debarment_status"],
        }
        for r in rows
    ]


@router.post("/{tender_id}/bidders", status_code=201)
async def create_bidder(
    tender_id: str,
    request: CreateBidderRequest,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Register a new bidder under a tender."""
    conn.row_factory = sqlite3.Row
    tender = conn.execute(
        "SELECT id FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    if not tender:
        raise HTTPException(status_code=404, detail={"error": {
            "code": "TENDER_NOT_FOUND",
            "message": f"Tender {tender_id} not found",
        }})

    bidder_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO bidders (id, tender_id, company_name, pan_number,
           registration_number, status, debarment_status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            bidder_id, tender_id,
            request.company_name,
            request.pan_number,
            request.registration_number,
            "pending", "clear",
        ),
    )
    conn.commit()

    return {
        "id": bidder_id,
        "tender_id": tender_id,
        "company_name": request.company_name,
        "pan_number": request.pan_number,
        "registration_number": request.registration_number,
        "status": "pending",
        "debarment_status": "clear",
    }


@router.get("")
async def list_tenders(
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """List all tenders."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tenders ORDER BY created_at DESC"
    ).fetchall()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "department": row["department"],
            "category": row["category"],
            "status": row["status"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


@router.get("/{tender_id}")
async def get_tender(
    tender_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get tender details including documents and bidders."""
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

    # Fetch associated documents
    documents = conn.execute(
        "SELECT id, filename, doc_type, processing_status FROM documents WHERE tender_id = ?",
        (tender_id,),
    ).fetchall()

    # Fetch associated bidders
    bidders = conn.execute(
        "SELECT id, company_name, status, debarment_status FROM bidders WHERE tender_id = ?",
        (tender_id,),
    ).fetchall()

    return {
        "id": tender["id"],
        "title": tender["title"],
        "department": tender["department"],
        "category": tender["category"],
        "status": tender["status"],
        "ets_version": tender["ets_version"],
        "created_at": tender["created_at"],
        "updated_at": tender["updated_at"],
        "documents": [dict(d) for d in documents],
        "bidders": [dict(b) for b in bidders],
    }


@router.post("/{tender_id}/process")
async def trigger_processing(
    tender_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Run L1 OCR + L2 UNION criterion extraction for the whole tender.

    Validates state machine: tender must be in DOCUMENTS_UPLOADED state.
    Executes synchronously for a clean demo experience — every uploaded
    document is OCR'd, then the NIT goes through union extraction,
    and any corrigendum is applied.
    """
    from backend.layers.l1_document import process_document
    from backend.layers.l2_ets_builder import extract_criteria, apply_corrigendum

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

    if tender["status"] not in ("DOCUMENTS_UPLOADED", "PROCESSING_OCR"):
        raise HTTPException(status_code=409, detail={
            "error": {
                "code": "INVALID_STATE_TRANSITION",
                "message": f"Cannot process tender in state '{tender['status']}'.",
                "details": {"current_status": tender["status"]},
            }
        })

    # Fetch all documents that still need processing
    docs = conn.execute(
        "SELECT * FROM documents WHERE tender_id = ? "
        "AND processing_status IN ('pending', 'processing') "
        "ORDER BY upload_timestamp ASC",
        (tender_id,),
    ).fetchall()

    if not docs:
        # Already processed — check if at least one document exists
        any_doc = conn.execute(
            "SELECT COUNT(*) as c FROM documents WHERE tender_id = ?",
            (tender_id,),
        ).fetchone()["c"]
        if any_doc == 0:
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "NO_DOCUMENTS",
                    "message": "At least one document must be uploaded before processing",
                }
            })

    # Move to PROCESSING_OCR
    conn.execute(
        "UPDATE tenders SET status = 'PROCESSING_OCR', updated_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), tender_id),
    )
    conn.commit()

    # Run L1 on every pending document (re-processes in place: we delete
    # the placeholder row created by /documents/upload, then the L1
    # pipeline inserts a fresh one with OCR word_objects + audit events).
    processed_docs: list[dict] = []
    for d in docs:
        doc_id = d["id"]
        file_path = d["file_path"]
        doc_type = d["doc_type"]
        bidder_id = d["bidder_id"]

        # Drop the placeholder row — process_document creates a new one
        # (we keep the file on disk via the same path).
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

        try:
            result = process_document(
                conn=conn,
                tender_id=tender_id,
                file_path=file_path,
                doc_type=doc_type,
                bidder_id=bidder_id,
            )
            conn.commit()
            processed_docs.append(result)
        except Exception as exc:
            # Re-insert a placeholder error row so the UI can show the problem
            conn.execute(
                """INSERT INTO documents (id, tender_id, bidder_id, doc_type,
                       filename, file_path, sha256_hash, page_count,
                       avg_ocr_confidence, upload_timestamp, processing_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, tender_id, bidder_id, doc_type, d["filename"], file_path,
                 d["sha256_hash"] or "", 0, 0.0, d["upload_timestamp"], "error"),
            )
            conn.commit()
            raise HTTPException(status_code=500, detail={
                "error": {
                    "code": "DOCUMENT_PROCESSING_FAILED",
                    "message": f"Failed to process {d['filename']}: {exc}",
                }
            })

    # OCR_COMPLETE
    conn.execute(
        "UPDATE tenders SET status = 'OCR_COMPLETE', updated_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), tender_id),
    )
    conn.commit()

    # Run L2 on the NIT — pick the first NIT document for this tender.
    nit_doc = conn.execute(
        "SELECT * FROM documents WHERE tender_id = ? AND doc_type = 'nit' "
        "ORDER BY upload_timestamp ASC LIMIT 1",
        (tender_id,),
    ).fetchone()

    criteria_count = 0
    if nit_doc:
        # Only run if no criteria yet (idempotent re-process)
        existing = conn.execute(
            "SELECT COUNT(*) as c FROM criteria WHERE tender_id = ?",
            (tender_id,),
        ).fetchone()["c"]
        if existing == 0:
            conn.execute(
                "UPDATE tenders SET status = 'EXTRACTING_CRITERIA', "
                "updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), tender_id),
            )
            conn.commit()
            criteria = extract_criteria(conn, tender_id, nit_doc["id"])
            criteria_count = len(criteria)
            conn.commit()
        else:
            criteria_count = existing

    # Apply every corrigendum
    corr_docs = conn.execute(
        "SELECT * FROM documents WHERE tender_id = ? AND doc_type = 'corrigendum' "
        "ORDER BY upload_timestamp ASC",
        (tender_id,),
    ).fetchall()
    for cd in corr_docs:
        try:
            apply_corrigendum(conn, tender_id, cd["id"])
            conn.commit()
        except Exception:
            # Corrigendum failures are non-fatal — officer will review
            conn.rollback()

    # Final status
    final_status = "SCHEMA_PENDING_REVIEW" if criteria_count > 0 else "OCR_COMPLETE"
    conn.execute(
        "UPDATE tenders SET status = ?, updated_at = ? WHERE id = ?",
        (final_status, datetime.now(timezone.utc).isoformat(), tender_id),
    )
    conn.commit()

    return {
        "status": "complete",
        "tender_id": tender_id,
        "processed_documents": len(processed_docs),
        "criteria_extracted": criteria_count,
        "corrigenda_applied": len(corr_docs),
        "final_tender_status": final_status,
    }


@router.get("/{tender_id}/status")
async def get_tender_status(
    tender_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Poll processing status for a tender."""
    conn.row_factory = sqlite3.Row

    tender = conn.execute(
        "SELECT id, status, updated_at FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()

    if not tender:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "TENDER_NOT_FOUND",
                "message": f"Tender {tender_id} not found",
            }
        })

    # Compute progress based on state
    state_progress = {
        "DOCUMENTS_UPLOADED": 0,
        "PROCESSING_OCR": 15,
        "OCR_COMPLETE": 25,
        "EXTRACTING_CRITERIA": 35,
        "SCHEMA_PENDING_REVIEW": 45,
        "SCHEMA_APPROVED": 55,
        "DEBARMENT_CHECK": 60,
        "DEBARMENT_FLAGGED": 65,
        "EVALUATING": 70,
        "VERDICTS_COMPUTED": 85,
        "HITL_PENDING": 90,
        "EVALUATION_COMPLETE": 95,
        "REPORT_GENERATED": 100,
    }

    return {
        "status": tender["status"],
        "progress_pct": state_progress.get(tender["status"], 0),
        "current_step": tender["status"],
        "updated_at": tender["updated_at"],
    }


# ─── ETS / Schema Review Endpoints ──────────────────────────────────────────


@router.get("/{tender_id}/criteria")
async def get_criteria(
    tender_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get extracted criteria for a tender."""
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

    criteria = conn.execute(
        "SELECT * FROM criteria WHERE tender_id = ? ORDER BY rowid",
        (tender_id,),
    ).fetchall()

    return [
        {
            "id": c["id"],
            "criterion_text": c["criterion_text"],
            "criterion_type": c["criterion_type"],
            "threshold_value": c["threshold_value"],
            "gfr_override_permitted": bool(c["gfr_override_permitted"]),
            "gfr_rule_number": c["gfr_rule_number"],
            "source_clause_ref": c["source_clause_ref"],
            "is_mandatory": bool(c["is_mandatory"]),
            "amendment_history": json.loads(c["amendment_history"]) if c["amendment_history"] else [],
            "status": c["status"],
            "acceptable_evidence_types": json.loads(c["acceptable_evidence_types"]) if c["acceptable_evidence_types"] else [],
            "measurement_period": c["measurement_period"],
        }
        for c in criteria
    ]


@router.put("/{tender_id}/criteria/{criterion_id}")
async def update_criterion(
    tender_id: str,
    criterion_id: str,
    request: UpdateCriterionRequest,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Edit a criterion during schema review.

    Only allowed when tender is in SCHEMA_PENDING_REVIEW state.
    """
    conn.row_factory = sqlite3.Row

    # Verify tender state
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

    if tender["status"] != "SCHEMA_PENDING_REVIEW":
        raise HTTPException(status_code=409, detail={
            "error": {
                "code": "INVALID_STATE_TRANSITION",
                "message": "Criteria can only be edited during schema review",
                "details": {
                    "current_status": tender["status"],
                    "required_status": "SCHEMA_PENDING_REVIEW",
                },
            }
        })

    # Fetch criterion
    criterion = conn.execute(
        "SELECT * FROM criteria WHERE id = ? AND tender_id = ?",
        (criterion_id, tender_id),
    ).fetchone()

    if not criterion:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "CRITERION_NOT_FOUND",
                "message": f"Criterion {criterion_id} not found in tender {tender_id}",
            }
        })

    # Apply updates
    updates = {}
    if request.criterion_text is not None:
        updates["criterion_text"] = request.criterion_text
    if request.threshold_value is not None:
        updates["threshold_value"] = request.threshold_value
    if request.criterion_type is not None:
        valid_types = (
            "numeric_threshold", "categorical_presence",
            "temporal_recency", "composite", "qualitative_assessment",
        )
        if request.criterion_type not in valid_types:
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "INVALID_CRITERION_TYPE",
                    "message": f"criterion_type must be one of: {', '.join(valid_types)}",
                }
            })
        updates["criterion_type"] = request.criterion_type

    if not updates:
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "NO_UPDATES",
                "message": "At least one field must be provided for update",
            }
        })

    # Build and execute update query
    set_clauses = ", ".join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [criterion_id]
    conn.execute(
        f"UPDATE criteria SET {set_clauses} WHERE id = ?",
        values,
    )
    conn.commit()

    # Return updated criterion
    updated = conn.execute(
        "SELECT * FROM criteria WHERE id = ?", (criterion_id,)
    ).fetchone()

    return {
        "id": updated["id"],
        "criterion_text": updated["criterion_text"],
        "criterion_type": updated["criterion_type"],
        "threshold_value": updated["threshold_value"],
        "gfr_override_permitted": bool(updated["gfr_override_permitted"]),
        "is_mandatory": bool(updated["is_mandatory"]),
        "status": updated["status"],
    }


@router.get("/{tender_id}/criteria/{criterion_id}/diff")
async def get_criterion_diff(
    tender_id: str,
    criterion_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get corrigendum diff for a criterion showing original vs amended values."""
    conn.row_factory = sqlite3.Row

    criterion = conn.execute(
        "SELECT * FROM criteria WHERE id = ? AND tender_id = ?",
        (criterion_id, tender_id),
    ).fetchone()

    if not criterion:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "CRITERION_NOT_FOUND",
                "message": f"Criterion {criterion_id} not found in tender {tender_id}",
            }
        })

    amendment_history = json.loads(criterion["amendment_history"]) if criterion["amendment_history"] else []

    if not amendment_history or len(amendment_history) < 2:
        return {
            "original": criterion["criterion_text"],
            "amended": None,
            "corrigendum_id": None,
            "amendment_date": None,
            "has_amendments": False,
        }

    return {
        "original": amendment_history[0] if amendment_history else criterion["criterion_text"],
        "amended": amendment_history[-1] if len(amendment_history) > 1 else None,
        "corrigendum_id": criterion["source_document_id"],
        "amendment_date": criterion["approved_at"],
        "has_amendments": True,
        "amendment_history": amendment_history,
    }


@router.post("/{tender_id}/schema/approve")
async def approve_schema(
    tender_id: str,
    request: SchemaApproveRequest,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Approve the criterion schema (gate).

    Enforces state machine: tender must be in SCHEMA_PENDING_REVIEW state.
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

    if tender["status"] != "SCHEMA_PENDING_REVIEW":
        raise HTTPException(status_code=409, detail={
            "error": {
                "code": "SCHEMA_NOT_APPROVED",
                "message": f"Cannot approve schema in state '{tender['status']}'. "
                           f"Expected 'SCHEMA_PENDING_REVIEW'.",
                "details": {
                    "current_status": tender["status"],
                    "required_status": "SCHEMA_PENDING_REVIEW",
                },
            }
        })

    approved_at = datetime.now(timezone.utc).isoformat()

    # Update all criteria to approved
    conn.execute(
        "UPDATE criteria SET status = 'approved', approved_by = ?, approved_at = ? WHERE tender_id = ?",
        (request.officer_id, approved_at, tender_id),
    )

    # Update tender status
    conn.execute(
        "UPDATE tenders SET status = 'SCHEMA_APPROVED', updated_at = ? WHERE id = ?",
        (approved_at, tender_id),
    )

    # Log audit event
    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="schema_approved",
        event_data={
            "officer_id": request.officer_id,
            "criteria_count": conn.execute(
                "SELECT COUNT(*) as cnt FROM criteria WHERE tender_id = ?",
                (tender_id,),
            ).fetchone()["cnt"],
        },
        actor=request.officer_id,
    )

    conn.commit()

    return {
        "status": "approved",
        "approved_at": approved_at,
        "officer_id": request.officer_id,
    }


@router.get("/{tender_id}/criteria/{criterion_id}/cpm")
async def get_criterion_cpm(
    tender_id: str,
    criterion_id: str,
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get CPM precedents for a specific criterion."""
    conn.row_factory = sqlite3.Row

    # Fetch criterion
    criterion = conn.execute(
        "SELECT * FROM criteria WHERE id = ? AND tender_id = ?",
        (criterion_id, tender_id),
    ).fetchone()

    if not criterion:
        raise HTTPException(status_code=404, detail={
            "error": {
                "code": "CRITERION_NOT_FOUND",
                "message": f"Criterion {criterion_id} not found in tender {tender_id}",
            }
        })

    # Fetch tender for department/category
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

    # Search CPM
    precedents = search_cpm_precedents(
        conn=conn,
        criterion_text=criterion["criterion_text"],
        department=tender["department"] or "",
        category=tender["category"] or "",
        limit=3,
    )

    return precedents
