"""Layer 2: ETS Builder for VerdictAI.

Assembles the Effective Tender Specification (ETS) from base NIT documents
and corrigenda. Handles criterion extraction via LLM Stub, type classification,
corrigendum version assembly, amendment history tracking, and the mandatory
Schema Review Gate.

Functions:
- extract_criteria: Extract criteria from document text via LLM Stub
- apply_corrigendum: Apply corrigendum amendments to existing criteria
- detect_missing_corrigendum: Scan text for amendment indicators without file
- build_ets: Assemble the complete ETS from all criteria for a tender
- approve_schema: Officer approval gate before evaluation
- update_criterion: Edit criterion during review
- check_schema_approved: Guard check for evaluation readiness

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 4.1, 4.2, 4.3, 4.4, 4.6
"""

import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from layers.l5_audit import append_audit_event
from services.cpm_service import store_precedent
from services.llm_stub import LLMStub
from services.union_service import (
    extract_criteria_union,
    map_corrigendum_union,
)


# Valid criterion types
VALID_CRITERION_TYPES = {
    "numeric_threshold",
    "categorical_presence",
    "temporal_recency",
    "composite",
    "qualitative_assessment",
}

# Amendment indicator phrases to detect missing corrigenda
AMENDMENT_INDICATORS = [
    "as amended",
    "refer addendum",
    "superseded by",
    "refer corrigendum",
]


def extract_criteria(
    conn: sqlite3.Connection,
    tender_id: str,
    document_id: str,
) -> list[dict]:
    """Extract eligibility criteria using the Union architecture (Rules + LLM).

    Runs the rule-based extractor AND the LLM extractor in parallel, then
    cross-validates. Criteria found by both branches land in the database
    at high confidence; criteria found by only one branch are flagged for
    officer review. The merge carries `_sources` metadata so the HITL UI
    can show which branch(es) detected each criterion.

    Args:
        conn: Active SQLite connection.
        tender_id: The tender this document belongs to.
        document_id: The document to extract criteria from.

    Returns:
        List of criterion dicts as stored in the database.
    """
    conn.row_factory = sqlite3.Row

    # Fetch document text (concatenated page raw_text)
    pages = conn.execute(
        "SELECT raw_text FROM pages WHERE document_id = ? ORDER BY page_number ASC",
        (document_id,),
    ).fetchall()

    document_text = "\n".join(
        row["raw_text"] for row in pages if row["raw_text"]
    )

    # ── UNION EXTRACTION: rules + LLM, cross-validated ──
    union = extract_criteria_union(
        document_text=document_text,
        source_document_id=document_id,
    )
    raw_criteria = union.value  # merged list with _sources metadata

    # Log the invocation for audit (includes both branches + agreement)
    llm = LLMStub()
    request = {
        "prompt_type": "criterion_extraction",
        "context": {
            "document_text": document_text[:500] + "..." if len(document_text) > 500 else document_text,
            "document_id": document_id,
            "tender_id": tender_id,
        },
        "tender_id": tender_id,
    }
    response = {
        "result": {
            "criteria": raw_criteria,
            "rules_count": len(union.rules_value),
            "llm_count": len(union.llm_value) if isinstance(union.llm_value, list) else 0,
            "agreement": union.agreement,
            "agreement_score": union.agreement_score,
        },
        "confidence": union.confidence,
        "reasoning": union.reasoning,
        "is_simulated": False,
        "model_version": union.llm_model_version or "rules-v1.0",
        "prompt_hash": union.llm_prompt_hash or "",
    }
    llm.log_invocation(conn, tender_id, request, response)

    stored_criteria = []
    for raw in raw_criteria:
        # Use the merged criterion's id if provided, else generate
        criterion_id = raw.get("id") or str(uuid.uuid4())

        # Classify type (validate against known types)
        criterion_type = raw.get("criterion_type", "qualitative_assessment")
        if criterion_type not in VALID_CRITERION_TYPES:
            criterion_type = "qualitative_assessment"

        # Build threshold_value JSON
        threshold_value = raw.get("threshold_value")
        threshold_json = json.dumps(threshold_value) if threshold_value else None

        # GFR override flag
        gfr_override_permitted = 1 if raw.get("gfr_override_permitted", True) else 0
        gfr_rule_number = raw.get("gfr_rule_number")

        # Source provenance
        source_clause_ref = raw.get("source_clause_ref", "")

        # Amendment history + _sources metadata (union branches that found this)
        amendment_history = raw.get("amendment_history", [])
        sources_tag = raw.get("_sources", ["rules"])
        amendment_history_json = json.dumps({
            "history": amendment_history,
            "_sources": sources_tag,
            "_agreement_similarity": raw.get("_agreement_similarity"),
            "needs_review": raw.get("needs_review", False),
            "review_reason": raw.get("review_reason"),
        })

        # Mandatory flag
        is_mandatory = 1 if raw.get("is_mandatory", False) else 0

        # Insert into criteria table
        conn.execute(
            """
            INSERT INTO criteria
                (id, tender_id, criterion_text, criterion_type, threshold_value,
                 gfr_override_permitted, gfr_rule_number, source_document_id,
                 source_clause_ref, amendment_history, is_mandatory,
                 acceptable_evidence_types, measurement_period, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                criterion_id,
                tender_id,
                raw.get("criterion_text", ""),
                criterion_type,
                threshold_json,
                gfr_override_permitted,
                gfr_rule_number,
                document_id,
                source_clause_ref,
                amendment_history_json,
                is_mandatory,
                json.dumps(raw.get("acceptable_evidence_types", [])),
                raw.get("measurement_period"),
                "extracted",
            ),
        )

        criterion_dict = {
            "id": criterion_id,
            "tender_id": tender_id,
            "criterion_text": raw.get("criterion_text", ""),
            "criterion_type": criterion_type,
            "threshold_value": threshold_value,
            "gfr_override_permitted": bool(gfr_override_permitted),
            "gfr_rule_number": gfr_rule_number,
            "source_document_id": document_id,
            "source_clause_ref": source_clause_ref,
            "amendment_history": amendment_history,
            "is_mandatory": bool(is_mandatory),
            "acceptable_evidence_types": raw.get("acceptable_evidence_types", []),
            "measurement_period": raw.get("measurement_period"),
            "status": "extracted",
            "approved_by": None,
            "approved_at": None,
            "_sources": sources_tag,  # for UI display
            "_agreement_similarity": raw.get("_agreement_similarity"),
            "needs_review": raw.get("needs_review", False),
            "review_reason": raw.get("review_reason"),
        }
        stored_criteria.append(criterion_dict)

    conn.commit()

    # Update tender status to SCHEMA_PENDING_REVIEW
    conn.execute(
        "UPDATE tenders SET status = ?, updated_at = ? WHERE id = ?",
        ("SCHEMA_PENDING_REVIEW", datetime.now(timezone.utc).isoformat(), tender_id),
    )
    conn.commit()

    return stored_criteria


def apply_corrigendum(
    conn: sqlite3.Connection,
    tender_id: str,
    corrigendum_doc_id: str,
) -> list[dict]:
    """Apply corrigendum amendments to existing criteria.

    Extracts amendments from the corrigendum document via LLM Stub,
    updates threshold_value with new values, and appends to amendment_history
    (JSON array with original + all amendments).

    Args:
        conn: Active SQLite connection.
        tender_id: The tender being amended.
        corrigendum_doc_id: The corrigendum document ID.

    Returns:
        Updated criteria list after amendments applied.
    """
    conn.row_factory = sqlite3.Row

    # Fetch corrigendum document text
    pages = conn.execute(
        "SELECT raw_text FROM pages WHERE document_id = ? ORDER BY page_number ASC",
        (corrigendum_doc_id,),
    ).fetchall()

    corrigendum_text = "\n".join(
        row["raw_text"] for row in pages if row["raw_text"]
    )

    # Invoke LLM Stub for criterion extraction from corrigendum
    llm = LLMStub()
    request = {
        "prompt_type": "criterion_extraction",
        "context": {
            "document_text": corrigendum_text,
            "document_id": corrigendum_doc_id,
            "tender_id": tender_id,
            "is_corrigendum": True,
        },
        "tender_id": tender_id,
        "scenario_hint": "corrigendum",
    }
    response = llm.invoke(request)
    llm.log_invocation(conn, tender_id, request, response)

    # Get amendments from response
    result = response.get("result", {})
    raw_criteria = result.get("criteria", [])

    # Fetch existing criteria for this tender
    existing_rows = conn.execute(
        "SELECT * FROM criteria WHERE tender_id = ?",
        (tender_id,),
    ).fetchall()

    # Build lookup by clause ref for matching
    existing_by_clause = {}
    for row in existing_rows:
        clause_ref = row["source_clause_ref"]
        if clause_ref:
            existing_by_clause[clause_ref] = dict(row)

    # Apply amendments
    for raw in raw_criteria:
        clause_ref = raw.get("source_clause_ref", "")
        if clause_ref and clause_ref in existing_by_clause:
            existing = existing_by_clause[clause_ref]
            criterion_id = existing["id"]

            # Parse current threshold and amendment history. The stored
            # shape is now `{"history": [...], "_sources": [...], ...}`
            # (new wrapper introduced by the union extractor); older rows
            # may still be bare lists. Handle both.
            current_threshold = json.loads(existing["threshold_value"]) if existing["threshold_value"] else {}
            stored_ah = json.loads(existing["amendment_history"]) if existing["amendment_history"] else []
            if isinstance(stored_ah, dict):
                stored_wrapper = dict(stored_ah)
                current_history = list(stored_wrapper.get("history") or [])
            else:
                stored_wrapper = None
                current_history = list(stored_ah)

            # New threshold from corrigendum
            new_threshold = raw.get("threshold_value", current_threshold)

            # Build amendment entry
            amendment_entry = {
                "original_value": current_threshold,
                "amended_value": new_threshold,
                "corrigendum_ref": f"Corrigendum doc {corrigendum_doc_id}",
                "amendment_reason": raw.get("amendment_reason", "Corrigendum amendment"),
            }
            current_history.append(amendment_entry)

            # Preserve the wrapper shape we read in (if any)
            if stored_wrapper is not None:
                stored_wrapper["history"] = current_history
                ah_json = json.dumps(stored_wrapper)
            else:
                ah_json = json.dumps(current_history)

            # Update criterion in database
            conn.execute(
                """
                UPDATE criteria
                SET threshold_value = ?,
                    amendment_history = ?,
                    source_document_id = ?
                WHERE id = ?
                """,
                (
                    json.dumps(new_threshold),
                    ah_json,
                    corrigendum_doc_id,
                    criterion_id,
                ),
            )

    conn.commit()

    # Log corrigendum_linked audit event
    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="corrigendum_linked",
        event_data={
            "corrigendum_document_id": corrigendum_doc_id,
            "criteria_amended": len(raw_criteria),
        },
        actor="system",
    )

    # Return updated criteria list
    updated_rows = conn.execute(
        "SELECT * FROM criteria WHERE tender_id = ?",
        (tender_id,),
    ).fetchall()

    updated_criteria = []
    for row in updated_rows:
        updated_criteria.append(_row_to_criterion_dict(row))

    return updated_criteria


def detect_missing_corrigendum(text: str) -> list[str]:
    """Scan text for amendment indicators without a corresponding corrigendum file.

    Looks for phrases that suggest amendments exist but may not have been
    uploaded: "as amended", "refer addendum", "superseded by", "refer corrigendum".

    Args:
        text: The document text to scan.

    Returns:
        List of detected indicator phrases found in the text.
    """
    detected = []
    text_lower = text.lower()

    for indicator in AMENDMENT_INDICATORS:
        if indicator in text_lower:
            detected.append(indicator)

    return detected


def build_ets(
    conn: sqlite3.Connection,
    tender_id: str,
) -> dict:
    """Assemble the Effective Tender Specification from all criteria for a tender.

    Collects all criteria, computes a version hash from the criteria content,
    and returns the ETS structure.

    Args:
        conn: Active SQLite connection.
        tender_id: The tender to build ETS for.

    Returns:
        ETS dict with keys: tender_id, criteria, version_hash, status, built_at.
    """
    conn.row_factory = sqlite3.Row

    # Fetch all criteria for this tender
    rows = conn.execute(
        "SELECT * FROM criteria WHERE tender_id = ? ORDER BY source_clause_ref ASC",
        (tender_id,),
    ).fetchall()

    criteria_list = [_row_to_criterion_dict(row) for row in rows]

    # Compute version hash from criteria content
    criteria_content = json.dumps(criteria_list, sort_keys=True, separators=(",", ":"))
    version_hash = hashlib.sha256(criteria_content.encode("utf-8")).hexdigest()

    # Update tender with ETS version
    conn.execute(
        "UPDATE tenders SET ets_version = ?, updated_at = ? WHERE id = ?",
        (version_hash, datetime.now(timezone.utc).isoformat(), tender_id),
    )
    conn.commit()

    # Determine ETS status
    tender_row = conn.execute(
        "SELECT status FROM tenders WHERE id = ?",
        (tender_id,),
    ).fetchone()
    tender_status = tender_row["status"] if tender_row else "UNKNOWN"

    return {
        "tender_id": tender_id,
        "criteria": criteria_list,
        "version_hash": version_hash,
        "status": tender_status,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Schema Review Gate ───────────────────────────────────────────────────────


def approve_schema(
    conn: sqlite3.Connection,
    tender_id: str,
    officer_id: str,
) -> dict:
    """Approve the criterion schema, enabling evaluation to proceed.

    Validates all criteria have been extracted, updates tender status to
    SCHEMA_APPROVED, updates all criteria status to "approved" with
    officer_id and timestamp, and logs the schema_approved audit event.

    Args:
        conn: Active SQLite connection.
        tender_id: The tender whose schema is being approved.
        officer_id: The officer approving the schema.

    Returns:
        Approval confirmation dict with status, officer_id, approved_at,
        and criteria_count.

    Raises:
        ValueError: If no criteria exist for the tender.
    """
    conn.row_factory = sqlite3.Row

    # Validate criteria exist
    criteria_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM criteria WHERE tender_id = ?",
        (tender_id,),
    ).fetchone()["cnt"]

    if criteria_count == 0:
        raise ValueError(
            f"Cannot approve schema for tender {tender_id}: no criteria extracted"
        )

    # Update tender status
    approved_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE tenders SET status = ?, updated_at = ? WHERE id = ?",
        ("SCHEMA_APPROVED", approved_at, tender_id),
    )

    # Update all criteria to approved
    conn.execute(
        """
        UPDATE criteria
        SET status = 'approved', approved_by = ?, approved_at = ?
        WHERE tender_id = ?
        """,
        (officer_id, approved_at, tender_id),
    )

    conn.commit()

    # Log schema_approved audit event
    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="schema_approved",
        event_data={
            "officer_id": officer_id,
            "criteria_count": criteria_count,
        },
        actor=officer_id,
    )

    return {
        "status": "approved",
        "tender_id": tender_id,
        "officer_id": officer_id,
        "approved_at": approved_at,
        "criteria_count": criteria_count,
    }


def update_criterion(
    conn: sqlite3.Connection,
    criterion_id: str,
    updates: dict,
    officer_id: str,
) -> dict:
    """Update criterion fields during schema review.

    If the interpretation (criterion_text) is changed, stores a CPM precedent
    via cpm_service.store_precedent to capture the officer's interpretation.

    Args:
        conn: Active SQLite connection.
        criterion_id: The criterion to update.
        updates: Dict of fields to update. Supported keys:
            criterion_text, criterion_type, threshold_value,
            gfr_override_permitted, is_mandatory, measurement_period.
        officer_id: The officer making the edit.

    Returns:
        Updated criterion dict.

    Raises:
        ValueError: If criterion not found.
    """
    conn.row_factory = sqlite3.Row

    # Fetch existing criterion
    row = conn.execute(
        "SELECT * FROM criteria WHERE id = ?",
        (criterion_id,),
    ).fetchone()

    if not row:
        raise ValueError(f"Criterion {criterion_id} not found")

    original_text = row["criterion_text"]
    tender_id = row["tender_id"]

    # Build SET clause dynamically from allowed fields
    allowed_fields = {
        "criterion_text",
        "criterion_type",
        "threshold_value",
        "gfr_override_permitted",
        "is_mandatory",
        "measurement_period",
    }

    set_parts = []
    params = []
    for field, value in updates.items():
        if field not in allowed_fields:
            continue

        # Serialize JSON fields
        if field == "threshold_value" and isinstance(value, dict):
            value = json.dumps(value)
        elif field == "gfr_override_permitted":
            value = 1 if value else 0
        elif field == "is_mandatory":
            value = 1 if value else 0

        set_parts.append(f"{field} = ?")
        params.append(value)

    if not set_parts:
        # No valid updates, return current state
        return _row_to_criterion_dict(row)

    # Mark as reviewed
    set_parts.append("status = ?")
    params.append("reviewed")

    # Execute update
    params.append(criterion_id)
    conn.execute(
        f"UPDATE criteria SET {', '.join(set_parts)} WHERE id = ?",
        params,
    )
    conn.commit()

    # If interpretation changed, store CPM precedent
    new_text = updates.get("criterion_text")
    if new_text and new_text != original_text:
        # Fetch tender details for CPM context
        tender_row = conn.execute(
            "SELECT department, category FROM tenders WHERE id = ?",
            (tender_id,),
        ).fetchone()

        if tender_row:
            store_precedent(
                conn=conn,
                criterion_text=original_text,
                resolved_interpretation=new_text,
                department=tender_row["department"],
                tender_category=tender_row["category"],
                verdict="PASS",
                officer_action="confirmed",
                officer_id=officer_id,
                tender_id=tender_id,
                criterion_id=criterion_id,
            )

    # Fetch and return updated criterion
    updated_row = conn.execute(
        "SELECT * FROM criteria WHERE id = ?",
        (criterion_id,),
    ).fetchone()

    return _row_to_criterion_dict(updated_row)


def check_schema_approved(
    conn: sqlite3.Connection,
    tender_id: str,
) -> bool:
    """Check if the tender schema has been approved.

    Returns True if tender status is SCHEMA_APPROVED or any later state
    in the evaluation pipeline. Used as a guard before evaluation.

    Args:
        conn: Active SQLite connection.
        tender_id: The tender to check.

    Returns:
        True if schema is approved (or later state), False otherwise.
    """
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT status FROM tenders WHERE id = ?",
        (tender_id,),
    ).fetchone()

    if not row:
        return False

    # States that indicate schema has been approved (SCHEMA_APPROVED or later)
    approved_states = {
        "SCHEMA_APPROVED",
        "DEBARMENT_CHECK",
        "DEBARMENT_FLAGGED",
        "EVALUATING",
        "VERDICTS_COMPUTED",
        "HITL_PENDING",
        "EVALUATION_COMPLETE",
        "REPORT_GENERATED",
    }

    return row["status"] in approved_states


# ─── Private helpers ──────────────────────────────────────────────────────────


def _row_to_criterion_dict(row: sqlite3.Row) -> dict:
    """Convert a database Row to a criterion dict with parsed JSON fields.

    Args:
        row: A sqlite3.Row from the criteria table.

    Returns:
        Dict representation of the criterion with JSON fields parsed.
    """
    threshold_value = None
    if row["threshold_value"]:
        try:
            threshold_value = json.loads(row["threshold_value"])
        except (json.JSONDecodeError, TypeError):
            threshold_value = row["threshold_value"]

    amendment_history = []
    sources_tag: list = ["rules"]
    agreement_similarity = None
    needs_review = False
    review_reason = None
    if row["amendment_history"]:
        try:
            parsed = json.loads(row["amendment_history"])
        except (json.JSONDecodeError, TypeError):
            parsed = []
        if isinstance(parsed, dict):
            amendment_history = list(parsed.get("history") or [])
            sources_tag = list(parsed.get("_sources") or ["rules"])
            agreement_similarity = parsed.get("_agreement_similarity")
            needs_review = bool(parsed.get("needs_review", False))
            review_reason = parsed.get("review_reason")
        elif isinstance(parsed, list):
            amendment_history = parsed

    acceptable_evidence_types = []
    if row["acceptable_evidence_types"]:
        try:
            acceptable_evidence_types = json.loads(row["acceptable_evidence_types"])
        except (json.JSONDecodeError, TypeError):
            acceptable_evidence_types = []

    return {
        "id": row["id"],
        "tender_id": row["tender_id"],
        "criterion_text": row["criterion_text"],
        "criterion_type": row["criterion_type"],
        "threshold_value": threshold_value,
        "gfr_override_permitted": bool(row["gfr_override_permitted"]),
        "gfr_rule_number": row["gfr_rule_number"],
        "source_document_id": row["source_document_id"],
        "source_clause_ref": row["source_clause_ref"],
        "amendment_history": amendment_history,
        "is_mandatory": bool(row["is_mandatory"]),
        "acceptable_evidence_types": acceptable_evidence_types,
        "measurement_period": row["measurement_period"],
        "status": row["status"],
        "approved_by": row["approved_by"],
        "approved_at": row["approved_at"],
        "_sources": sources_tag,
        "_agreement_similarity": agreement_similarity,
        "needs_review": needs_review,
        "review_reason": review_reason,
    }
