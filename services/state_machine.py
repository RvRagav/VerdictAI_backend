"""State machine for tender evaluation workflow.

Implements the VerdictAI tender lifecycle state transitions with
guard conditions from the design state transition table.

States:
    DOCUMENTS_UPLOADED → PROCESSING_OCR → OCR_COMPLETE → EXTRACTING_CRITERIA →
    SCHEMA_PENDING_REVIEW → SCHEMA_APPROVED → DEBARMENT_CHECK →
    DEBARMENT_FLAGGED → EVALUATING → VERDICTS_COMPUTED → HITL_PENDING →
    EVALUATION_COMPLETE → REPORT_GENERATED → BIDDER_EXCLUDED

Requirements: 4.1, 5.1, 9.1
"""

import sqlite3
from datetime import datetime, timezone


# All valid states in the tender lifecycle
VALID_STATES = [
    "DOCUMENTS_UPLOADED",
    "PROCESSING_OCR",
    "OCR_COMPLETE",
    "EXTRACTING_CRITERIA",
    "SCHEMA_PENDING_REVIEW",
    "SCHEMA_APPROVED",
    "DEBARMENT_CHECK",
    "DEBARMENT_FLAGGED",
    "BIDDER_EXCLUDED",
    "EVALUATING",
    "VERDICTS_COMPUTED",
    "HITL_PENDING",
    "EVALUATION_COMPLETE",
    "REPORT_GENERATED",
]

# Valid state transitions: current_state → list of allowed next states
VALID_TRANSITIONS: dict[str, list[str]] = {
    "DOCUMENTS_UPLOADED": ["PROCESSING_OCR"],
    "PROCESSING_OCR": ["OCR_COMPLETE"],
    "OCR_COMPLETE": ["EXTRACTING_CRITERIA"],
    "EXTRACTING_CRITERIA": ["SCHEMA_PENDING_REVIEW"],
    "SCHEMA_PENDING_REVIEW": ["SCHEMA_APPROVED", "SCHEMA_PENDING_REVIEW"],
    "SCHEMA_APPROVED": ["DEBARMENT_CHECK"],
    "DEBARMENT_CHECK": ["EVALUATING", "DEBARMENT_FLAGGED"],
    "DEBARMENT_FLAGGED": ["EVALUATING", "BIDDER_EXCLUDED"],
    "EVALUATING": ["VERDICTS_COMPUTED"],
    "VERDICTS_COMPUTED": ["HITL_PENDING", "EVALUATION_COMPLETE"],
    "HITL_PENDING": ["EVALUATION_COMPLETE"],
    "EVALUATION_COMPLETE": ["REPORT_GENERATED"],
    "REPORT_GENERATED": [],
    "BIDDER_EXCLUDED": [],
}

# Guard conditions for each transition
# Each guard is a function(conn, tender_id) -> (bool, str)
# Returns (True, "") if guard passes, (False, reason) if it fails


def _guard_has_documents(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str]:
    """Guard: At least one document must be uploaded."""
    count = conn.execute(
        "SELECT COUNT(*) as c FROM documents WHERE tender_id = ?",
        (tender_id,),
    ).fetchone()["c"]
    if count == 0:
        return (False, "At least one document must be uploaded before processing")
    return (True, "")


def _guard_all_pages_processed(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str]:
    """Guard: All document pages must be processed (OCR complete)."""
    # Check if any documents are still in processing state
    pending = conn.execute(
        "SELECT COUNT(*) as c FROM documents WHERE tender_id = ? AND processing_status != 'complete'",
        (tender_id,),
    ).fetchone()["c"]
    if pending > 0:
        return (False, f"{pending} document(s) still processing")
    return (True, "")


def _guard_schema_approved(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str]:
    """Guard: Schema must be explicitly approved by an officer."""
    tender = conn.execute(
        "SELECT status FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    if not tender:
        return (False, "Tender not found")
    # This guard is implicitly satisfied by the state transition itself
    return (True, "")


def _guard_no_debarment_match(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str]:
    """Guard: No debarment match found (for transition to EVALUATING from DEBARMENT_CHECK)."""
    flagged = conn.execute(
        "SELECT COUNT(*) as c FROM bidders WHERE tender_id = ? AND debarment_status = 'flagged'",
        (tender_id,),
    ).fetchone()["c"]
    if flagged > 0:
        return (False, f"{flagged} bidder(s) flagged for debarment — must resolve first")
    return (True, "")


def _guard_all_evaluated(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str]:
    """Guard: All (bidder, criterion) pairs must be evaluated."""
    # Count bidders and criteria
    bidders = conn.execute(
        "SELECT COUNT(*) as c FROM bidders WHERE tender_id = ? AND status != 'excluded'",
        (tender_id,),
    ).fetchone()["c"]
    criteria = conn.execute(
        "SELECT COUNT(*) as c FROM criteria WHERE tender_id = ? AND status = 'approved'",
        (tender_id,),
    ).fetchone()["c"]

    expected = bidders * criteria
    if expected == 0:
        return (True, "")

    actual = conn.execute(
        "SELECT COUNT(*) as c FROM evaluations WHERE tender_id = ?",
        (tender_id,),
    ).fetchone()["c"]

    if actual < expected:
        return (False, f"Only {actual}/{expected} evaluations completed")
    return (True, "")


def _guard_has_hitl_cases(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str]:
    """Guard: At least one case routed to HITL/Mandatory review."""
    pending = conn.execute(
        "SELECT COUNT(*) as c FROM evaluations WHERE tender_id = ? AND status IN ('pending_review', 'pending_second_officer')",
        (tender_id,),
    ).fetchone()["c"]
    if pending == 0:
        return (False, "No cases pending HITL review")
    return (True, "")


def _guard_all_cases_resolved(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str]:
    """Guard: All pending HITL cases must be resolved."""
    pending = conn.execute(
        "SELECT COUNT(*) as c FROM evaluations WHERE tender_id = ? AND status IN ('pending_review', 'pending_second_officer')",
        (tender_id,),
    ).fetchone()["c"]
    if pending > 0:
        return (False, f"{pending} case(s) still pending officer review")
    return (True, "")


def _guard_always_pass(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str]:
    """Guard: Always passes (no condition required)."""
    return (True, "")


# Map of (from_state, to_state) → guard function
TRANSITION_GUARDS: dict[tuple[str, str], callable] = {
    ("DOCUMENTS_UPLOADED", "PROCESSING_OCR"): _guard_has_documents,
    ("PROCESSING_OCR", "OCR_COMPLETE"): _guard_all_pages_processed,
    ("OCR_COMPLETE", "EXTRACTING_CRITERIA"): _guard_always_pass,
    ("EXTRACTING_CRITERIA", "SCHEMA_PENDING_REVIEW"): _guard_always_pass,
    ("SCHEMA_PENDING_REVIEW", "SCHEMA_APPROVED"): _guard_always_pass,
    ("SCHEMA_PENDING_REVIEW", "SCHEMA_PENDING_REVIEW"): _guard_always_pass,
    ("SCHEMA_APPROVED", "DEBARMENT_CHECK"): _guard_always_pass,
    ("DEBARMENT_CHECK", "EVALUATING"): _guard_no_debarment_match,
    ("DEBARMENT_CHECK", "DEBARMENT_FLAGGED"): _guard_always_pass,
    ("DEBARMENT_FLAGGED", "EVALUATING"): _guard_always_pass,
    ("DEBARMENT_FLAGGED", "BIDDER_EXCLUDED"): _guard_always_pass,
    ("EVALUATING", "VERDICTS_COMPUTED"): _guard_all_evaluated,
    ("VERDICTS_COMPUTED", "HITL_PENDING"): _guard_has_hitl_cases,
    ("VERDICTS_COMPUTED", "EVALUATION_COMPLETE"): _guard_always_pass,
    ("HITL_PENDING", "EVALUATION_COMPLETE"): _guard_all_cases_resolved,
    ("EVALUATION_COMPLETE", "REPORT_GENERATED"): _guard_always_pass,
}


def get_allowed_transitions(current_status: str) -> list[str]:
    """Return the list of allowed next states from the current status.

    Args:
        current_status: The current tender state.

    Returns:
        List of valid target states. Empty list if no transitions available.
    """
    return VALID_TRANSITIONS.get(current_status, [])


def transition_tender(
    conn: sqlite3.Connection,
    tender_id: str,
    target_status: str,
) -> dict:
    """Validate and execute a state transition for a tender.

    Checks:
    1. Tender exists
    2. Target state is a valid transition from current state
    3. Guard condition for the transition passes

    Args:
        conn: Active SQLite connection (caller manages transaction).
        tender_id: The tender to transition.
        target_status: The desired next state.

    Returns:
        Dict with keys: tender_id, previous_status, new_status, transitioned_at.

    Raises:
        ValueError: If tender not found, transition invalid, or guard fails.
    """
    conn.row_factory = sqlite3.Row

    # Fetch current state
    tender = conn.execute(
        "SELECT id, status FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()

    if not tender:
        raise ValueError(f"Tender {tender_id} not found")

    current_status = tender["status"]

    # Check if transition is valid
    allowed = get_allowed_transitions(current_status)
    if target_status not in allowed:
        raise ValueError(
            f"Invalid state transition: {current_status} → {target_status}. "
            f"Allowed transitions: {allowed}"
        )

    # Check guard condition
    guard_key = (current_status, target_status)
    guard_fn = TRANSITION_GUARDS.get(guard_key, _guard_always_pass)
    passes, reason = guard_fn(conn, tender_id)

    if not passes:
        raise ValueError(
            f"Guard condition failed for {current_status} → {target_status}: {reason}"
        )

    # Execute transition
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE tenders SET status = ?, updated_at = ? WHERE id = ?",
        (target_status, now, tender_id),
    )

    return {
        "tender_id": tender_id,
        "previous_status": current_status,
        "new_status": target_status,
        "transitioned_at": now,
    }
