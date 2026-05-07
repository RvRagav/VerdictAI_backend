"""Layer 4: Evaluation Engine for VerdictAI.

Provides the Confidence Router and type-specific evaluation logic.
Computes verdicts, confidence scores, and routing decisions for each
(bidder, criterion) pair.

Functions:
- compute_route: Confidence Router implementing all 6 routing rules
- evaluate_criterion: Evaluate a single (bidder, criterion) pair
- evaluate_all_bidders: Evaluate all bidders for a tender

Requirements: 5.1, 5.2, 5.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 11.5
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from backend.config import settings
from backend.layers.l3_evidence import extract_evidence
from backend.layers.l5_audit import append_audit_event
from backend.services.cpm_service import get_cpm_stats
from backend.services.debarment_service import check_debarment
from backend.services.explanation_service import build_explanation


def compute_route(
    verdict: str,
    confidence: float,
    criterion_type: str,
    flags: list[str],
    is_mandatory: bool,
    gfr_override_permitted: bool,
    cpm_data_count: int,
) -> dict:
    """Confidence Router implementing all 6 routing rules in priority order.

    Routing priority (highest to lowest):
    1. Mandatory FAIL → always mandatory_review
    2. Explicit flags (entity_mismatch, debarment, etc.) → mandatory_review
    3. Low confidence (< floor) → mandatory_review
    4. LLM FAIL verdict → hitl_review (never auto-commit LLM disqualification)
    5. Medium confidence (< auto_commit_threshold) → hitl_review
    6. High confidence (≥ threshold) + deterministic + no flags → auto_commit

    Conservative thresholds are applied when cpm_data_count < 50:
    - Auto-commit ceiling raised to 0.90 (from 0.85)
    - Mandatory review floor raised to 0.60 (from 0.50)

    Args:
        verdict: The evaluation verdict ("PASS", "FAIL", or "REVIEW").
        confidence: The confidence score in [0.0, 1.0].
        criterion_type: One of the 5 criterion types.
        flags: List of flag strings (e.g., "entity_mismatch", "debarment").
        is_mandatory: Whether this is a mandatory criterion.
        gfr_override_permitted: Whether GFR override is allowed.
        cpm_data_count: Number of CPM entries for threshold adjustment.

    Returns:
        Dict matching RoutingDecision model with keys: route, confidence,
        reasons, flags, gfr_override_permitted, is_mandatory_criterion.
    """
    # Determine thresholds based on CPM data availability
    if cpm_data_count < 50:
        auto_commit_threshold = settings.confidence.conservative_auto  # 0.90
        mandatory_floor = settings.confidence.conservative_floor  # 0.60
    else:
        auto_commit_threshold = settings.confidence.auto_commit  # 0.85
        mandatory_floor = settings.confidence.mandatory_floor  # 0.50

    reasons = []

    # Rule 1: Mandatory criterion FAIL → always mandatory review
    if is_mandatory and verdict == "FAIL":
        reasons.append("Mandatory criterion FAIL requires officer confirmation")
        return {
            "route": "mandatory_review",
            "confidence": confidence,
            "reasons": reasons,
            "flags": flags,
            "gfr_override_permitted": gfr_override_permitted,
            "is_mandatory_criterion": is_mandatory,
        }

    # Rule 2: Explicit flags → mandatory review
    if flags:
        reasons.append(f"Flags present: {', '.join(flags)}")
        return {
            "route": "mandatory_review",
            "confidence": confidence,
            "reasons": reasons,
            "flags": flags,
            "gfr_override_permitted": gfr_override_permitted,
            "is_mandatory_criterion": is_mandatory,
        }

    # Rule 3: Low confidence → mandatory review
    if confidence < mandatory_floor:
        reasons.append(
            f"Confidence {confidence:.2f} below mandatory floor {mandatory_floor:.2f}"
        )
        return {
            "route": "mandatory_review",
            "confidence": confidence,
            "reasons": reasons,
            "flags": flags,
            "gfr_override_permitted": gfr_override_permitted,
            "is_mandatory_criterion": is_mandatory,
        }

    # Rule 4: LLM FAIL → never auto-commit
    if criterion_type == "qualitative_assessment" and verdict == "FAIL":
        reasons.append("LLM-based FAIL verdict requires officer review")
        return {
            "route": "hitl_review",
            "confidence": confidence,
            "reasons": reasons,
            "flags": flags,
            "gfr_override_permitted": gfr_override_permitted,
            "is_mandatory_criterion": is_mandatory,
        }

    # Rule 5: Medium confidence → HITL review
    if confidence < auto_commit_threshold:
        reasons.append(
            f"Confidence {confidence:.2f} below auto-commit threshold {auto_commit_threshold:.2f}"
        )
        return {
            "route": "hitl_review",
            "confidence": confidence,
            "reasons": reasons,
            "flags": flags,
            "gfr_override_permitted": gfr_override_permitted,
            "is_mandatory_criterion": is_mandatory,
        }

    # Rule 6: High confidence + deterministic + no flags → auto-commit
    deterministic_types = (
        "numeric_threshold",
        "categorical_presence",
        "temporal_recency",
    )
    if criterion_type in deterministic_types:
        reasons.append("High confidence deterministic evaluation")
        return {
            "route": "auto_commit",
            "confidence": confidence,
            "reasons": reasons,
            "flags": flags,
            "gfr_override_permitted": gfr_override_permitted,
            "is_mandatory_criterion": is_mandatory,
        }

    # Default: HITL for qualitative even at high confidence PASS
    reasons.append("Qualitative assessment requires review")
    return {
        "route": "hitl_review",
        "confidence": confidence,
        "reasons": reasons,
        "flags": flags,
        "gfr_override_permitted": gfr_override_permitted,
        "is_mandatory_criterion": is_mandatory,
    }


def evaluate_criterion(
    conn: sqlite3.Connection,
    tender_id: str,
    bidder_id: str,
    criterion_id: str,
) -> dict:
    """Evaluate a single (bidder, criterion) pair.

    Calls extract_evidence to get evidence, applies type-specific
    evaluation logic, computes confidence, calls compute_route to
    determine routing, stores the evaluation record, and logs audit events.

    Args:
        conn: Active SQLite connection.
        tender_id: The tender being evaluated.
        bidder_id: The bidder being evaluated.
        criterion_id: The criterion to evaluate.

    Returns:
        Dict representing the complete evaluation record.
    """
    conn.row_factory = sqlite3.Row

    # Fetch criterion details
    criterion = conn.execute(
        "SELECT * FROM criteria WHERE id = ?", (criterion_id,)
    ).fetchone()

    if not criterion:
        return _error_evaluation(tender_id, bidder_id, criterion_id, "criterion_not_found")

    criterion_type = criterion["criterion_type"]
    is_mandatory = bool(criterion["is_mandatory"])
    gfr_override_permitted = bool(criterion["gfr_override_permitted"])

    # Extract evidence
    evidence = extract_evidence(conn, tender_id, bidder_id, criterion_id)

    # Apply type-specific evaluation logic
    verdict, eval_confidence, evaluation_method = _evaluate_by_type(
        criterion_type, criterion, evidence
    )

    # Collect flags
    flags = []
    if evidence.get("entity_match_flag"):
        flags.append("entity_mismatch")

    # Get CPM stats for threshold adjustment
    cpm_stats = get_cpm_stats(conn)
    cpm_data_count = cpm_stats.get("total_entries", 0)

    # Compute routing decision
    routing = compute_route(
        verdict=verdict,
        confidence=eval_confidence,
        criterion_type=criterion_type,
        flags=flags,
        is_mandatory=is_mandatory,
        gfr_override_permitted=gfr_override_permitted,
        cpm_data_count=cpm_data_count,
    )

    # Determine evaluation status based on route
    route = routing["route"]
    if route == "auto_commit":
        status = "auto_committed"
    else:
        status = "pending_review"

    # Build officer-grade explanation. This is stored on the evaluation
    # so the HITL card, audit trail, and PDF report all render the
    # same human-readable account of the verdict.
    criterion_dict = dict(criterion) if criterion else {}
    # Parse threshold_value JSON for the explanation builder.
    tv_raw = criterion_dict.get("threshold_value")
    if isinstance(tv_raw, str) and tv_raw:
        try:
            criterion_dict["threshold_value"] = json.loads(tv_raw)
        except (json.JSONDecodeError, TypeError):
            pass
    union_agreement = None
    if isinstance(evidence.get("value"), dict):
        union_info = evidence["value"].get("union") or {}
        union_agreement = union_info.get("agreement")
    explanation = build_explanation(
        verdict=verdict,
        criterion=criterion_dict,
        evidence=evidence,
        route=route,
        union_agreement=union_agreement,
    )

    # Create evaluation record
    evaluation_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Use the explanation's headline as the routing_reason so any legacy
    # consumer reading routing_reason immediately gets a human sentence
    # instead of "Semantic score criterion↔document=0.23".
    routing_reason = explanation["headline"]

    evaluation = {
        "id": evaluation_id,
        "tender_id": tender_id,
        "bidder_id": bidder_id,
        "criterion_id": criterion_id,
        "verdict": verdict,
        "confidence": eval_confidence,
        "evaluation_method": evaluation_method,
        "route": route,
        "routing_reason": routing_reason,
        "extracted_value": _serialise_with_explanation(
            evidence.get("value"), explanation
        ),
        "source_document_id": evidence.get("source_document_id"),
        "source_page_number": evidence.get("page"),
        "source_bbox": json.dumps(evidence.get("bbox")) if evidence.get("bbox") else None,
        "ocr_confidence": evidence.get("ocr_confidence", 0.0),
        "extraction_confidence": evidence.get("extraction_confidence", 0.0),
        "entity_match_flag": 1 if evidence.get("entity_match_flag") else 0,
        "status": status,
        "created_at": created_at,
    }

    # Store evaluation in database
    conn.execute(
        """INSERT INTO evaluations
            (id, tender_id, bidder_id, criterion_id, verdict, confidence,
             evaluation_method, route, routing_reason, extracted_value,
             source_document_id, source_page_number, source_bbox,
             ocr_confidence, extraction_confidence, entity_match_flag,
             status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            evaluation["id"],
            evaluation["tender_id"],
            evaluation["bidder_id"],
            evaluation["criterion_id"],
            evaluation["verdict"],
            evaluation["confidence"],
            evaluation["evaluation_method"],
            evaluation["route"],
            evaluation["routing_reason"],
            evaluation["extracted_value"],
            evaluation["source_document_id"],
            evaluation["source_page_number"],
            evaluation["source_bbox"],
            evaluation["ocr_confidence"],
            evaluation["extraction_confidence"],
            evaluation["entity_match_flag"],
            evaluation["status"],
            evaluation["created_at"],
        ),
    )

    # Log verdict_computed audit event
    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="verdict_computed",
        event_data={
            "evaluation_id": evaluation_id,
            "bidder_id": bidder_id,
            "criterion_id": criterion_id,
            "verdict": verdict,
            "confidence": eval_confidence,
            "evaluation_method": evaluation_method,
        },
        actor="system",
    )

    # Log case_routed audit event
    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="case_routed",
        event_data={
            "evaluation_id": evaluation_id,
            "bidder_id": bidder_id,
            "criterion_id": criterion_id,
            "route": route,
            "reasons": routing["reasons"],
            "flags": flags,
        },
        actor="system",
    )

    # Add routing info to returned evaluation
    evaluation["routing_decision"] = routing
    evaluation["explanation"] = explanation

    return evaluation


def evaluate_all_bidders(
    conn: sqlite3.Connection,
    tender_id: str,
) -> list[dict]:
    """Evaluate all bidders for a tender against all approved criteria.

    Fetches all bidders and approved criteria for the tender, then
    calls evaluate_criterion for each (bidder, criterion) pair.

    Args:
        conn: Active SQLite connection.
        tender_id: The tender to evaluate.

    Returns:
        List of all evaluation result dicts.
    """
    conn.row_factory = sqlite3.Row

    # Fetch all bidders for this tender (excluding excluded ones)
    bidders = conn.execute(
        "SELECT * FROM bidders WHERE tender_id = ? AND status != 'excluded'",
        (tender_id,),
    ).fetchall()

    # Fetch all approved criteria for this tender
    criteria = conn.execute(
        "SELECT * FROM criteria WHERE tender_id = ? AND status = 'approved'",
        (tender_id,),
    ).fetchall()

    results = []

    for bidder in bidders:
        # Run debarment check before evaluation
        debarment_result = check_debarment(
            conn=conn,
            company_name=bidder["company_name"],
            pan_number=bidder["pan_number"],
        )

        # Log debarment check
        append_audit_event(
            conn=conn,
            tender_id=tender_id,
            event_type="debarment_checked",
            event_data={
                "bidder_id": bidder["id"],
                "company_name": bidder["company_name"],
                "is_debarred": debarment_result["is_debarred"],
                "check_method": debarment_result["check_method"],
            },
            actor="system",
        )

        # Update bidder debarment status
        debarment_status = "flagged" if debarment_result["is_debarred"] else "clear"
        conn.execute(
            "UPDATE bidders SET debarment_status = ?, debarment_check_timestamp = ? WHERE id = ?",
            (
                debarment_status,
                datetime.now(timezone.utc).isoformat(),
                bidder["id"],
            ),
        )

        # If debarred, halt evaluation for this bidder
        if debarment_result["is_debarred"]:
            conn.execute(
                "UPDATE bidders SET status = 'excluded' WHERE id = ?",
                (bidder["id"],),
            )
            results.append({
                "bidder_id": bidder["id"],
                "company_name": bidder["company_name"],
                "status": "debarment_flagged",
                "debarment_result": debarment_result,
                "evaluations": [],
            })
            continue

        # Evaluate each criterion for this bidder
        bidder_evaluations = []
        for criterion in criteria:
            evaluation = evaluate_criterion(
                conn=conn,
                tender_id=tender_id,
                bidder_id=bidder["id"],
                criterion_id=criterion["id"],
            )
            bidder_evaluations.append(evaluation)

        # Update bidder status
        conn.execute(
            "UPDATE bidders SET status = 'evaluated' WHERE id = ?",
            (bidder["id"],),
        )

        results.append({
            "bidder_id": bidder["id"],
            "company_name": bidder["company_name"],
            "status": "evaluated",
            "evaluations": bidder_evaluations,
        })

    return results


# ─── Type-Specific Evaluation Logic ─────────────────────────────────────────


def _evaluate_by_type(
    criterion_type: str,
    criterion: sqlite3.Row,
    evidence: dict,
) -> tuple[str, float, str]:
    """Apply type-specific evaluation logic to evidence.

    Args:
        criterion_type: The type of criterion.
        criterion: The criterion database row.
        evidence: The extracted evidence dict.

    Returns:
        Tuple of (verdict, confidence, evaluation_method).
    """
    if criterion_type == "numeric_threshold":
        return _evaluate_numeric_threshold(criterion, evidence)
    elif criterion_type == "categorical_presence":
        return _evaluate_categorical_presence(criterion, evidence)
    elif criterion_type == "temporal_recency":
        return _evaluate_temporal_recency(criterion, evidence)
    elif criterion_type == "qualitative_assessment":
        return _evaluate_qualitative_assessment(criterion, evidence)
    else:
        # Unknown type - route to review
        return "REVIEW", 0.0, "unknown"


def _evaluate_numeric_threshold(
    criterion: sqlite3.Row,
    evidence: dict,
) -> tuple[str, float, str]:
    """Evaluate numeric threshold criterion by comparing extracted
    value (in rupees) against the threshold (also in rupees).

    Both the extractor and the criterion schema normalise amounts to
    rupees via ``to_rupees``. If either side is stored in raw units
    (older data), we re-normalise here using the ``unit`` field.
    """
    from backend.services.criterion_extractor import to_rupees

    value = evidence.get("value")
    extraction_confidence = evidence.get("extraction_confidence", 0.0)

    if value is None or value.get("amount") is None:
        return "REVIEW", 0.0, "deterministic"

    threshold_data = (
        json.loads(criterion["threshold_value"])
        if criterion["threshold_value"]
        else {}
    )

    # Extract threshold in rupees — prefer pre-computed ``rupees`` field,
    # otherwise convert ``value`` + ``unit``.
    threshold_rupees: int = 0
    if isinstance(threshold_data, dict):
        if "rupees" in threshold_data and threshold_data["rupees"]:
            try:
                threshold_rupees = int(threshold_data["rupees"])
            except (TypeError, ValueError):
                threshold_rupees = 0
        if not threshold_rupees:
            raw_val = threshold_data.get("value", 0)
            unit = str(threshold_data.get("unit") or "").strip()
            try:
                threshold_rupees = to_rupees(float(raw_val), unit)
            except (TypeError, ValueError):
                threshold_rupees = 0

    # Extracted value is always stored in rupees (evidence_extractor
    # normalises via to_rupees). Defensive cast.
    try:
        extracted_amount = int(value.get("amount", 0))
    except (TypeError, ValueError):
        extracted_amount = 0

    if threshold_rupees <= 0:
        # Threshold isn't well-defined — route to review rather than guess.
        return "REVIEW", extraction_confidence, "deterministic"

    verdict = "PASS" if extracted_amount >= threshold_rupees else "FAIL"
    return verdict, extraction_confidence, "deterministic"


def _evaluate_categorical_presence(
    criterion: sqlite3.Row,
    evidence: dict,
) -> tuple[str, float, str]:
    """Evaluate categorical presence criterion.

    Checks if the required document/certificate was found and is valid.
    """
    value = evidence.get("value")
    extraction_confidence = evidence.get("extraction_confidence", 0.0)

    if value is None:
        return "REVIEW", 0.0, "deterministic"

    found = value.get("found", False)
    is_valid = value.get("is_valid")

    if not found:
        # Document not found
        return "FAIL", extraction_confidence, "deterministic"

    if is_valid is True:
        # Document found and valid
        return "PASS", extraction_confidence, "deterministic"

    if is_valid is None:
        # Validity unclear - route to review
        return "REVIEW", extraction_confidence * 0.7, "deterministic"

    # Document found but invalid
    return "FAIL", extraction_confidence, "deterministic"


def _evaluate_temporal_recency(
    criterion: sqlite3.Row,
    evidence: dict,
) -> tuple[str, float, str]:
    """Evaluate temporal recency criterion.

    Checks if sufficient recent projects were found.
    """
    value = evidence.get("value")
    extraction_confidence = evidence.get("extraction_confidence", 0.0)

    if value is None:
        return "REVIEW", 0.0, "deterministic"

    count = value.get("count", 0)
    required_count = value.get("required_count", 2)

    if count >= required_count:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    # Confidence based on extraction confidence
    confidence = extraction_confidence

    return verdict, confidence, "deterministic"


def _evaluate_qualitative_assessment(
    criterion: sqlite3.Row,
    evidence: dict,
) -> tuple[str, float, str]:
    """Evaluate qualitative assessment criterion using LLM Stub verdict.

    Uses the LLM Stub's verdict and confidence directly.
    """
    value = evidence.get("value")
    extraction_confidence = evidence.get("extraction_confidence", 0.0)

    if value is None:
        return "REVIEW", 0.0, "llm_stub"

    llm_verdict = value.get("llm_verdict", "REVIEW")
    llm_confidence = value.get("llm_confidence", 0.5)

    # Use LLM verdict directly
    verdict = llm_verdict

    # Confidence combines extraction confidence and LLM confidence
    confidence = min(extraction_confidence, llm_confidence)

    return verdict, confidence, "llm_stub"


# ─── Helper Functions ────────────────────────────────────────────────────────


def _serialise_with_explanation(
    value: dict | None,
    explanation: dict,
) -> str | None:
    """Serialise an extracted value + officer-grade explanation to JSON.

    Stored in ``evaluations.extracted_value`` so the HITL card and PDF
    report can surface the rich explanation without needing a schema
    migration. Consumers that only care about the raw value can read
    ``parsed["__value__"]``.
    """
    payload = {
        "__value__": value,
        "__explanation__": explanation,
    }
    try:
        return json.dumps(payload, default=str)
    except (TypeError, ValueError):
        # Fall back to string representation — never lose the verdict.
        return json.dumps(
            {"__value__": str(value), "__explanation__": explanation},
            default=str,
        )


def _error_evaluation(
    tender_id: str,
    bidder_id: str,
    criterion_id: str,
    error_reason: str,
) -> dict:
    """Return an error evaluation when processing cannot proceed."""
    evaluation_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    return {
        "id": evaluation_id,
        "tender_id": tender_id,
        "bidder_id": bidder_id,
        "criterion_id": criterion_id,
        "verdict": "REVIEW",
        "confidence": 0.0,
        "evaluation_method": "error",
        "route": "mandatory_review",
        "routing_reason": f"Error: {error_reason}",
        "extracted_value": None,
        "source_document_id": None,
        "source_page_number": None,
        "source_bbox": None,
        "ocr_confidence": 0.0,
        "extraction_confidence": 0.0,
        "entity_match_flag": 0,
        "status": "pending_review",
        "created_at": created_at,
        "routing_decision": {
            "route": "mandatory_review",
            "confidence": 0.0,
            "reasons": [f"Error: {error_reason}"],
            "flags": [],
            "gfr_override_permitted": False,
            "is_mandatory_criterion": False,
        },
    }
