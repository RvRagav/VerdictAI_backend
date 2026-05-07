"""Layer 5: Audit Ledger service for VerdictAI.

Provides an append-only, hash-chained audit trail for all system events.
Each entry links to the previous via SHA-256, forming a tamper-evident chain.

Functions:
- append_audit_event: Add a new event to the ledger with hash chain linkage
- verify_hash_chain: Validate integrity of the entire chain for a tender
- get_audit_trail: Query audit events with optional filtering
"""

import json
import sqlite3
from datetime import datetime, timezone

from backend.utils.hash_utils import compute_entry_hash


# Genesis hash used as prev_hash for the first entry in a tender's chain
GENESIS_HASH = "0" * 64


def append_audit_event(
    conn: sqlite3.Connection,
    tender_id: str,
    event_type: str,
    event_data: dict,
    actor: str,
) -> dict:
    """Append an event to the audit ledger with hash chain linkage.

    Retrieves the most recent entry's hash for this tender (or uses the
    genesis hash for the first entry), computes the new entry hash, and
    inserts the record.

    Args:
        conn: Active SQLite connection (caller manages transaction).
        tender_id: The tender this event belongs to.
        event_type: Type of event (e.g. "document_received", "verdict_computed").
        event_data: Arbitrary JSON-serialisable event payload.
        actor: Who triggered the event ("system" or an officer ID).

    Returns:
        Dict representing the created audit event with all fields including
        id, tender_id, event_type, event_data, actor, timestamp, prev_hash,
        and entry_hash.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Get previous entry's hash (or genesis for first entry)
    cursor = conn.execute(
        "SELECT entry_hash FROM audit_events WHERE tender_id = ? ORDER BY id DESC LIMIT 1",
        (tender_id,),
    )
    row = cursor.fetchone()
    prev_hash = row["entry_hash"] if row else GENESIS_HASH

    # Compute this entry's hash
    entry_hash = compute_entry_hash(event_type, event_data, actor, timestamp, prev_hash)

    # Insert into audit_events
    event_data_json = json.dumps(event_data, sort_keys=True, separators=(",", ":"))
    cursor = conn.execute(
        """INSERT INTO audit_events (tender_id, event_type, event_data, actor, timestamp, prev_hash, entry_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tender_id, event_type, event_data_json, actor, timestamp, prev_hash, entry_hash),
    )

    event_id = cursor.lastrowid

    return {
        "id": event_id,
        "tender_id": tender_id,
        "event_type": event_type,
        "event_data": event_data,
        "actor": actor,
        "timestamp": timestamp,
        "prev_hash": prev_hash,
        "entry_hash": entry_hash,
    }


def verify_hash_chain(conn: sqlite3.Connection, tender_id: str) -> tuple[bool, str | None]:
    """Verify the integrity of the audit hash chain for a tender.

    Checks two invariants for every entry:
    1. event[i].prev_hash == event[i-1].entry_hash (chain linkage)
    2. Recomputed hash from stored fields matches event[i].entry_hash

    Args:
        conn: Active SQLite connection.
        tender_id: The tender whose chain to verify.

    Returns:
        Tuple of (is_valid, error_message).
        - (True, None) if the chain is intact.
        - (False, "description") if a break is detected.
    """
    cursor = conn.execute(
        """SELECT id, event_type, event_data, actor, timestamp, prev_hash, entry_hash
           FROM audit_events
           WHERE tender_id = ?
           ORDER BY id ASC""",
        (tender_id,),
    )
    rows = cursor.fetchall()

    if not rows:
        return (True, None)  # Empty chain is trivially valid

    for i, row in enumerate(rows):
        event_id = row["id"]
        event_type = row["event_type"]
        event_data_raw = row["event_data"]
        actor = row["actor"]
        timestamp = row["timestamp"]
        prev_hash = row["prev_hash"]
        entry_hash = row["entry_hash"]

        # Parse event_data from JSON string
        try:
            event_data = json.loads(event_data_raw)
        except (json.JSONDecodeError, TypeError):
            return (False, f"Event {event_id}: event_data is not valid JSON")

        # Check 1: prev_hash linkage
        if i == 0:
            expected_prev = GENESIS_HASH
        else:
            expected_prev = rows[i - 1]["entry_hash"]

        if prev_hash != expected_prev:
            return (
                False,
                f"Event {event_id}: prev_hash mismatch. "
                f"Expected '{expected_prev}', got '{prev_hash}'",
            )

        # Check 2: Recompute entry_hash and verify
        recomputed = compute_entry_hash(event_type, event_data, actor, timestamp, prev_hash)
        if recomputed != entry_hash:
            return (
                False,
                f"Event {event_id}: entry_hash mismatch. "
                f"Recomputed '{recomputed}', stored '{entry_hash}'",
            )

    return (True, None)


def get_audit_trail(
    conn: sqlite3.Connection,
    tender_id: str,
    event_type: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[dict]:
    """Fetch audit events for a tender with optional filtering.

    Args:
        conn: Active SQLite connection.
        tender_id: The tender to query.
        event_type: Optional filter by event type (exact match).
        from_date: Optional ISO 8601 lower bound (inclusive) on timestamp.
        to_date: Optional ISO 8601 upper bound (inclusive) on timestamp.

    Returns:
        List of audit event dicts ordered by id ascending.
    """
    query = "SELECT * FROM audit_events WHERE tender_id = ?"
    params: list = [tender_id]

    if event_type is not None:
        query += " AND event_type = ?"
        params.append(event_type)

    if from_date is not None:
        query += " AND timestamp >= ?"
        params.append(from_date)

    if to_date is not None:
        query += " AND timestamp <= ?"
        params.append(to_date)

    query += " ORDER BY id ASC"

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for row in rows:
        event_data_raw = row["event_data"]
        try:
            event_data = json.loads(event_data_raw)
        except (json.JSONDecodeError, TypeError):
            event_data = event_data_raw

        results.append({
            "id": row["id"],
            "tender_id": row["tender_id"],
            "event_type": row["event_type"],
            "event_data": event_data,
            "actor": row["actor"],
            "timestamp": row["timestamp"],
            "prev_hash": row["prev_hash"],
            "entry_hash": row["entry_hash"],
        })

    return results
