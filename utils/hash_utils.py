"""SHA-256 hash utilities for VerdictAI audit ledger.

Provides deterministic hashing for:
- Audit event entries (hash chain integrity)
- File content verification (document deduplication)
"""

import hashlib
import json


def compute_entry_hash(
    event_type: str,
    event_data: dict,
    actor: str,
    timestamp: str,
    prev_hash: str,
) -> str:
    """Compute SHA-256 hash of an audit event entry for hash chain linkage.

    Uses deterministic JSON serialisation (sorted keys, compact separators)
    to ensure identical inputs always produce the same hash regardless of
    dict insertion order.

    Args:
        event_type: The type of audit event (e.g. "document_received").
        event_data: Arbitrary event payload as a dict.
        actor: The actor who triggered the event ("system" or officer ID).
        timestamp: ISO 8601 timestamp string.
        prev_hash: SHA-256 hex digest of the previous entry in the chain.

    Returns:
        Lowercase 64-character hex digest (SHA-256).
    """
    payload = json.dumps(
        {
            "event_type": event_type,
            "event_data": event_data,
            "actor": actor,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's contents.

    Reads the file in 8KB chunks to handle large files without
    excessive memory usage.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Lowercase 64-character hex digest (SHA-256).

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
