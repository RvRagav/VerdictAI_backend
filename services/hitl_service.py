"""Human-in-the-Loop (HITL) decision processing service for VerdictAI.

Handles officer decisions (confirm/override), GFR enforcement,
second-officer confirmation flows, HITL queue management, and
HITL card data assembly.

Requirements: 8.5, 8.6, 8.7, 9.1, 9.2, 10.2, 10.3, 10.4, 10.5, 11.1
"""

import json
import sqlite3
from datetime import datetime, timezone

from layers.l5_audit import append_audit_event
from services.cpm_service import search_cpm_precedents, store_precedent


def process_decision(
    conn: sqlite3.Connection,
    evaluation_id: str,
    decision: str,
    officer_id: str,
    reason: str | None = None,
    reason_text: str | None = None,
) -> dict:
    """Process an officer decision on a pending evaluation.

    Validates the evaluation exists and is pending_review, enforces GFR
    override rules, records the decision, logs audit events, and stores
    a CPM precedent entry.

    Args:
        conn: Active SQLite connection.
        evaluation_id: The evaluation to decide on.
        decision: Either "confirm" or "override".
        officer_id: The deciding officer's identifier.
        reason: Structured reason code for overrides (required for override).
        reason_text: Optional free-text note accompanying the reason.

    Returns:
        Dict representing the updated evaluation record.

    Raises:
        ValueError: If evaluation not found, not pending, override reason
            missing, or GFR override not permitted.
    """
    conn.row_factory = sqlite3.Row

    # Fetch evaluation
    evaluation = conn.execute(
        "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
    ).fetchone()

    if not evaluation:
        raise ValueError(f"Evaluation {evaluation_id} not found")

    if evaluation["status"] != "pending_review":
        raise ValueError(
            f"Evaluation {evaluation_id} is not pending review "
            f"(current status: {evaluation['status']})"
        )

    # Validate decision value
    if decision not in ("confirm", "override"):
        raise ValueError(f"Invalid decision: {decision}. Must be 'confirm' or 'override'.")

    # For overrides, validate reason is provided (Property 17)
    if decision == "override":
        if not reason or not reason.strip():
            raise ValueError("Override requires a structured reason")

        # Fetch criterion for GFR check
        criterion = conn.execute(
            "SELECT * FROM criteria WHERE id = ?", (evaluation["criterion_id"],)
        ).fetchone()

        if criterion:
            gfr_override_permitted = bool(criterion["gfr_override_permitted"])

            # GFR enforcement (Property 6): reject override if not permitted and verdict is FAIL
            if not gfr_override_permitted and evaluation["verdict"] == "FAIL":
                raise ValueError(
                    f"Override not permitted: GFR rule "
                    f"{criterion['gfr_rule_number'] or 'applicable'} "
                    f"prevents override of FAIL verdict on this criterion"
                )

            # Check if second-officer confirmation is needed for GFR-adjacent overrides
            if _requires_second_officer(criterion, evaluation):
                # Mark as needing second officer but record the initial decision
                conn.execute(
                    """UPDATE evaluations SET
                        officer_decision = ?,
                        officer_id = ?,
                        officer_reason = ?,
                        officer_decision_timestamp = ?,
                        status = 'pending_second_officer'
                    WHERE id = ?""",
                    (
                        decision,
                        officer_id,
                        json.dumps({"reason": reason, "text": reason_text}),
                        datetime.now(timezone.utc).isoformat(),
                        evaluation_id,
                    ),
                )

                # Log audit event
                append_audit_event(
                    conn=conn,
                    tender_id=evaluation["tender_id"],
                    event_type="officer_decision",
                    event_data={
                        "evaluation_id": evaluation_id,
                        "decision": decision,
                        "officer_id": officer_id,
                        "reason": reason,
                        "reason_text": reason_text,
                        "requires_second_officer": True,
                    },
                    actor=officer_id,
                )

                return _build_evaluation_dict(conn, evaluation_id)

    # Record the decision
    timestamp = datetime.now(timezone.utc).isoformat()
    officer_reason_json = json.dumps({"reason": reason, "text": reason_text}) if reason else None

    conn.execute(
        """UPDATE evaluations SET
            officer_decision = ?,
            officer_id = ?,
            officer_reason = ?,
            officer_decision_timestamp = ?,
            status = 'resolved',
            resolved_at = ?
        WHERE id = ?""",
        (
            decision,
            officer_id,
            officer_reason_json,
            timestamp,
            timestamp,
            evaluation_id,
        ),
    )

    # Log audit event
    append_audit_event(
        conn=conn,
        tender_id=evaluation["tender_id"],
        event_type="officer_decision",
        event_data={
            "evaluation_id": evaluation_id,
            "decision": decision,
            "officer_id": officer_id,
            "reason": reason,
            "reason_text": reason_text,
            "requires_second_officer": False,
        },
        actor=officer_id,
    )

    # Store CPM precedent entry
    _store_cpm_precedent(conn, evaluation, decision, officer_id, reason)

    return _build_evaluation_dict(conn, evaluation_id)


def process_second_officer(
    conn: sqlite3.Connection,
    evaluation_id: str,
    officer_id: str,
    decision: str,
) -> dict:
    """Process second-officer confirmation for a pending override.

    Validates that second-officer confirmation is needed, records the
    confirmation, and finalises the evaluation.

    Args:
        conn: Active SQLite connection.
        evaluation_id: The evaluation requiring second-officer confirmation.
        officer_id: The second officer's identifier.
        decision: Either "approve" or "reject".

    Returns:
        Dict with confirmation details.

    Raises:
        ValueError: If evaluation doesn't need second-officer confirmation,
            or if the same officer tries to confirm their own decision.
    """
    conn.row_factory = sqlite3.Row

    evaluation = conn.execute(
        "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
    ).fetchone()

    if not evaluation:
        raise ValueError(f"Evaluation {evaluation_id} not found")

    if evaluation["status"] != "pending_second_officer":
        raise ValueError(
            f"Evaluation {evaluation_id} does not require second-officer confirmation "
            f"(current status: {evaluation['status']})"
        )

    # Prevent same officer from confirming their own decision
    if evaluation["officer_id"] == officer_id:
        raise ValueError("Second officer must be different from the first officer")

    timestamp = datetime.now(timezone.utc).isoformat()

    if decision == "approve":
        # Finalise the override
        conn.execute(
            """UPDATE evaluations SET
                second_officer_id = ?,
                second_officer_timestamp = ?,
                status = 'resolved',
                resolved_at = ?
            WHERE id = ?""",
            (officer_id, timestamp, timestamp, evaluation_id),
        )
    elif decision == "reject":
        # Revert to pending_review - the override is rejected
        conn.execute(
            """UPDATE evaluations SET
                second_officer_id = ?,
                second_officer_timestamp = ?,
                officer_decision = NULL,
                officer_id = NULL,
                officer_reason = NULL,
                officer_decision_timestamp = NULL,
                status = 'pending_review'
            WHERE id = ?""",
            (officer_id, timestamp, evaluation_id),
        )
    else:
        raise ValueError(f"Invalid decision: {decision}. Must be 'approve' or 'reject'.")

    # Log audit event
    append_audit_event(
        conn=conn,
        tender_id=evaluation["tender_id"],
        event_type="officer_decision",
        event_data={
            "evaluation_id": evaluation_id,
            "second_officer_id": officer_id,
            "second_officer_decision": decision,
            "original_officer_id": evaluation["officer_id"],
        },
        actor=officer_id,
    )

    return {
        "evaluation_id": evaluation_id,
        "second_officer_id": officer_id,
        "second_officer_decision": decision,
        "second_officer_timestamp": timestamp,
        "status": "resolved" if decision == "approve" else "pending_review",
    }


def get_hitl_queue(
    conn: sqlite3.Connection,
    tender_id: str,
    route_filter: str | None = None,
) -> list[dict]:
    """Fetch pending evaluations for a tender's HITL queue.

    Returns evaluations ordered by priority: mandatory_review first,
    then by confidence ascending (lowest confidence = highest priority).

    Args:
        conn: Active SQLite connection.
        tender_id: The tender to fetch queue for.
        route_filter: Optional filter by route ("hitl_review" or "mandatory_review").

    Returns:
        List of evaluation dicts ordered by priority.
    """
    conn.row_factory = sqlite3.Row

    query = """
        SELECT e.*, c.criterion_text, c.criterion_type, c.is_mandatory,
               b.company_name as bidder_name
        FROM evaluations e
        JOIN criteria c ON e.criterion_id = c.id
        JOIN bidders b ON e.bidder_id = b.id
        WHERE e.tender_id = ?
          AND e.status IN ('pending_review', 'pending_second_officer')
    """
    params: list = [tender_id]

    if route_filter:
        query += " AND e.route = ?"
        params.append(route_filter)

    # Order: mandatory_review first, then by confidence ascending
    query += """
        ORDER BY
            CASE e.route
                WHEN 'mandatory_review' THEN 0
                WHEN 'hitl_review' THEN 1
                ELSE 2
            END,
            e.confidence ASC
    """

    rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        results.append({
            "evaluation_id": row["id"],
            "tender_id": row["tender_id"],
            "bidder_id": row["bidder_id"],
            "bidder_name": row["bidder_name"],
            "criterion_id": row["criterion_id"],
            "criterion_text": row["criterion_text"],
            "criterion_type": row["criterion_type"],
            "verdict": row["verdict"],
            "confidence": row["confidence"],
            "route": row["route"],
            "routing_reason": row["routing_reason"],
            "is_mandatory": bool(row["is_mandatory"]),
            "status": row["status"],
            "created_at": row["created_at"],
        })

    return results


def get_hitl_card(
    conn: sqlite3.Connection,
    evaluation_id: str,
) -> dict:
    """Return full HITL card data for a single evaluation.

    Assembles all data needed for the 5-component HITL review card:
    criterion details, evidence with bbox, system analysis, CPM
    precedents, and decision options.

    Args:
        conn: Active SQLite connection.
        evaluation_id: The evaluation to build the card for.

    Returns:
        Dict with keys: criterion, evidence, analysis, cpm_precedents,
        decision_options.

    Raises:
        ValueError: If evaluation not found.
    """
    conn.row_factory = sqlite3.Row

    # Fetch evaluation
    evaluation = conn.execute(
        "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
    ).fetchone()

    if not evaluation:
        raise ValueError(f"Evaluation {evaluation_id} not found")

    # Fetch criterion details
    criterion = conn.execute(
        "SELECT * FROM criteria WHERE id = ?", (evaluation["criterion_id"],)
    ).fetchone()

    # Fetch bidder details
    bidder = conn.execute(
        "SELECT * FROM bidders WHERE id = ?", (evaluation["bidder_id"],)
    ).fetchone()

    # Fetch tender details
    tender = conn.execute(
        "SELECT * FROM tenders WHERE id = ?", (evaluation["tender_id"],)
    ).fetchone()

    # Build evidence section with bbox
    # extracted_value is stored as JSON that wraps both the raw value
    # and the officer-grade explanation: {"__value__": ..., "__explanation__": ...}.
    # Legacy rows stored just the value; handle both.
    raw_extracted = evaluation["extracted_value"]
    parsed_extracted: dict | list | str | int | float | None = None
    explanation_payload: dict | None = None
    if raw_extracted:
        try:
            parsed_extracted = json.loads(raw_extracted)
        except (json.JSONDecodeError, TypeError):
            parsed_extracted = raw_extracted
    if isinstance(parsed_extracted, dict) and "__explanation__" in parsed_extracted:
        explanation_payload = parsed_extracted.get("__explanation__")
        parsed_extracted = parsed_extracted.get("__value__")

    evidence = {
        "source_document_id": evaluation["source_document_id"],
        "source_page_number": evaluation["source_page_number"],
        "source_bbox": json.loads(evaluation["source_bbox"]) if evaluation["source_bbox"] else None,
        "extracted_value": parsed_extracted,
        "ocr_confidence": evaluation["ocr_confidence"],
        "extraction_confidence": evaluation["extraction_confidence"],
        "entity_match_flag": bool(evaluation["entity_match_flag"]),
    }

    # Build system analysis section. The routing_reason is written by
    # L4 as the explanation headline, so it's already a human sentence.
    analysis = {
        "verdict": evaluation["verdict"],
        "confidence": evaluation["confidence"],
        "evaluation_method": evaluation["evaluation_method"],
        "route": evaluation["route"],
        "routing_reason": evaluation["routing_reason"],
        "explanation": explanation_payload,
    }

    # Fetch CPM precedents
    cpm_precedents = []
    if criterion and tender:
        cpm_precedents = search_cpm_precedents(
            conn=conn,
            criterion_text=criterion["criterion_text"],
            department=tender["department"] or "",
            category=tender["category"] or "",
            limit=3,
        )

    # Build decision options
    gfr_override_permitted = bool(criterion["gfr_override_permitted"]) if criterion else True
    decision_options = {
        "can_confirm": True,
        "can_override": gfr_override_permitted or evaluation["verdict"] != "FAIL",
        "gfr_override_permitted": gfr_override_permitted,
        "gfr_rule_number": criterion["gfr_rule_number"] if criterion else None,
        "requires_second_officer": _requires_second_officer(criterion, evaluation) if criterion else False,
    }

    return {
        "evaluation_id": evaluation_id,
        "criterion": {
            "id": criterion["id"] if criterion else None,
            "text": criterion["criterion_text"] if criterion else None,
            "type": criterion["criterion_type"] if criterion else None,
            "threshold_value": criterion["threshold_value"] if criterion else None,
            "is_mandatory": bool(criterion["is_mandatory"]) if criterion else False,
            "gfr_override_permitted": gfr_override_permitted,
            "gfr_rule_number": criterion["gfr_rule_number"] if criterion else None,
        },
        "bidder": {
            "id": bidder["id"] if bidder else None,
            "company_name": bidder["company_name"] if bidder else None,
        },
        "evidence": evidence,
        "analysis": analysis,
        "cpm_precedents": cpm_precedents,
        "decision_options": decision_options,
    }


# ─── Private Helpers ─────────────────────────────────────────────────────────


def _requires_second_officer(criterion, evaluation) -> bool:
    """Check if second-officer confirmation is needed.

    Required for GFR-adjacent overrides: when gfr_override_permitted is True
    but the criterion is mandatory and the verdict is FAIL.
    """
    if criterion is None:
        return False

    is_mandatory = bool(criterion["is_mandatory"])
    gfr_override_permitted = bool(criterion["gfr_override_permitted"])

    # Second officer needed for borderline GFR-adjacent criteria:
    # Override is permitted but criterion is mandatory and verdict is FAIL
    if gfr_override_permitted and is_mandatory and evaluation["verdict"] == "FAIL":
        return True

    return False


def _store_cpm_precedent(
    conn: sqlite3.Connection,
    evaluation,
    decision: str,
    officer_id: str,
    reason: str | None,
) -> None:
    """Store a CPM precedent entry from an officer decision."""
    # Fetch criterion and tender for context
    criterion = conn.execute(
        "SELECT * FROM criteria WHERE id = ?", (evaluation["criterion_id"],)
    ).fetchone()

    tender = conn.execute(
        "SELECT * FROM tenders WHERE id = ?", (evaluation["tender_id"],)
    ).fetchone()

    if not criterion or not tender:
        return

    # Build resolved interpretation from decision context
    if decision == "confirm":
        resolved_interpretation = f"Confirmed system verdict: {evaluation['verdict']}"
    else:
        resolved_interpretation = f"Overridden to opposite of {evaluation['verdict']}. Reason: {reason or 'N/A'}"

    store_precedent(
        conn=conn,
        criterion_text=criterion["criterion_text"],
        resolved_interpretation=resolved_interpretation,
        department=tender["department"] or "",
        tender_category=tender["category"] or "",
        verdict=evaluation["verdict"],
        officer_action=decision,
        officer_id=officer_id,
        tender_id=evaluation["tender_id"],
        criterion_id=evaluation["criterion_id"],
    )


def _build_evaluation_dict(conn: sqlite3.Connection, evaluation_id: str) -> dict:
    """Fetch and return the current state of an evaluation as a dict."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
    ).fetchone()

    if not row:
        return {}

    return {
        "id": row["id"],
        "tender_id": row["tender_id"],
        "bidder_id": row["bidder_id"],
        "criterion_id": row["criterion_id"],
        "verdict": row["verdict"],
        "confidence": row["confidence"],
        "evaluation_method": row["evaluation_method"],
        "route": row["route"],
        "routing_reason": row["routing_reason"],
        "extracted_value": row["extracted_value"],
        "source_document_id": row["source_document_id"],
        "source_page_number": row["source_page_number"],
        "source_bbox": row["source_bbox"],
        "ocr_confidence": row["ocr_confidence"],
        "extraction_confidence": row["extraction_confidence"],
        "entity_match_flag": bool(row["entity_match_flag"]),
        "officer_decision": row["officer_decision"],
        "officer_id": row["officer_id"],
        "officer_reason": row["officer_reason"],
        "officer_decision_timestamp": row["officer_decision_timestamp"],
        "second_officer_id": row["second_officer_id"],
        "second_officer_timestamp": row["second_officer_timestamp"],
        "status": row["status"],
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
    }
