"""Criterion Precedent Memory (CPM) service for VerdictAI.

Provides FTS5-based similarity search over officer interpretation
precedents and storage of new precedent entries. Used by the ETS
Builder (L2) and Evidence Extraction (L3) layers to surface relevant
historical decisions.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.6
"""

import re
import sqlite3
import uuid
from datetime import datetime, timezone

from services import embedding_service


# Stop words to remove from FTS5 queries
STOP_WORDS = {
    "the", "a", "an", "of", "in", "for", "and", "or", "to",
    "is", "shall", "must", "be", "not",
}


def tokenise_for_search(text: str) -> list[str]:
    """Remove stop words and short tokens for FTS5 query.

    Extracts word tokens from text, filters out stop words and
    tokens with 2 or fewer characters.

    Args:
        text: Raw text to tokenise.

    Returns:
        List of filtered tokens suitable for FTS5 query construction.
    """
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def search_cpm_precedents(
    conn: sqlite3.Connection,
    criterion_text: str,
    department: str,
    category: str,
    limit: int = 3,
) -> list[dict]:
    """Search CPM using SQLite FTS5 with BM25 ranking.

    Tokenises the criterion text, builds an OR-joined FTS5 query,
    filters by department and category, and returns up to `limit`
    results ranked by BM25 relevance.

    Args:
        conn: Active SQLite connection.
        criterion_text: The criterion text to search for similar precedents.
        department: Department filter (exact match).
        category: Tender category filter (exact match).
        limit: Maximum number of results (default 3, capped at 3).

    Returns:
        List of CPMEntry-like dicts with all cpm_entries fields.
    """
    # Cap limit to maximum 3
    limit = min(limit, 3)

    # Tokenise and build FTS5 query
    tokens = tokenise_for_search(criterion_text)
    if not tokens:
        return []

    fts_query = " OR ".join(tokens)

    sql = """
        SELECT cpm_entries.*, bm25(cpm_fts) as rank
        FROM cpm_fts
        JOIN cpm_entries ON cpm_entries.rowid = cpm_fts.rowid
        WHERE cpm_fts MATCH ?
          AND cpm_entries.department = ?
          AND cpm_entries.tender_category = ?
        ORDER BY rank
        LIMIT ?
    """

    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql, (fts_query, department, category, limit))
    rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "criterion_text": row["criterion_text"],
            "resolved_interpretation": row["resolved_interpretation"],
            "department": row["department"],
            "tender_category": row["tender_category"],
            "verdict": row["verdict"],
            "officer_action": row["officer_action"],
            "officer_id": row["officer_id"],
            "tender_id": row["tender_id"],
            "criterion_id": row["criterion_id"],
            "created_at": row["created_at"],
        })

    return results


def search_cpm_precedents_semantic(
    conn: sqlite3.Connection,
    criterion_text: str,
    department: str,
    category: str,
    limit: int = 3,
) -> list[dict]:
    """Semantic CPM precedent search using sentence-transformer embeddings.

    For a given ``criterion_text`` (plus department/category scope),
    ranks every CPM entry by cosine similarity and returns the top
    ``limit`` matches. This is the preferred search strategy when the
    scoped CPM corpus is small (< ~1000 entries): scoring all rows is
    fast and avoids the tokenisation / stopword brittleness of FTS5.

    For larger corpora the ``search_cpm_precedents`` FTS5 path should
    be used as a first-stage filter and this function called on the
    returned candidates as a re-rank step.

    Args:
        conn: Active SQLite connection.
        criterion_text: The criterion text to search with.
        department: Department filter (exact match).
        category: Tender category filter (exact match).
        limit: Maximum results (capped at 3 to match FTS5 variant).

    Returns:
        List of CPMEntry-like dicts, each augmented with a
        ``similarity`` float. Sorted by similarity descending.
    """
    limit = min(limit, 3)
    if not criterion_text or not criterion_text.strip():
        return []

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT * FROM cpm_entries
        WHERE department = ? AND tender_category = ?
        """,
        (department, category),
    ).fetchall()

    if not rows:
        return []

    candidate_texts = [row["criterion_text"] for row in rows]
    ranked = embedding_service.rank_by_similarity(
        criterion_text, candidate_texts, top_k=limit
    )

    results: list[dict] = []
    for idx, sim in ranked:
        row = rows[idx]
        results.append(
            {
                "id": row["id"],
                "criterion_text": row["criterion_text"],
                "resolved_interpretation": row["resolved_interpretation"],
                "department": row["department"],
                "tender_category": row["tender_category"],
                "verdict": row["verdict"],
                "officer_action": row["officer_action"],
                "officer_id": row["officer_id"],
                "tender_id": row["tender_id"],
                "criterion_id": row["criterion_id"],
                "created_at": row["created_at"],
                "similarity": sim,
            }
        )
    return results


def store_precedent(
    conn: sqlite3.Connection,
    criterion_text: str,
    resolved_interpretation: str,
    department: str,
    tender_category: str,
    verdict: str,
    officer_action: str,
    officer_id: str,
    tender_id: str,
    criterion_id: str,
) -> dict:
    """Store a new CPM precedent entry.

    Creates a new entry in the cpm_entries table. The FTS5 sync
    trigger automatically updates the cpm_fts virtual table.

    Args:
        conn: Active SQLite connection.
        criterion_text: The criterion text being interpreted.
        resolved_interpretation: The officer's resolved interpretation.
        department: Department this precedent belongs to.
        tender_category: Tender category for this precedent.
        verdict: The verdict (PASS or FAIL).
        officer_action: The officer action (confirmed or overridden).
        officer_id: Anonymised officer identifier.
        tender_id: The tender this precedent originated from.
        criterion_id: The criterion this precedent relates to.

    Returns:
        Dict representing the created CPM entry.
    """
    entry_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO cpm_entries
            (id, criterion_text, resolved_interpretation, department,
             tender_category, verdict, officer_action, officer_id,
             tender_id, criterion_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry_id,
            criterion_text,
            resolved_interpretation,
            department,
            tender_category,
            verdict,
            officer_action,
            officer_id,
            tender_id,
            criterion_id,
            created_at,
        ),
    )
    conn.commit()

    return {
        "id": entry_id,
        "criterion_text": criterion_text,
        "resolved_interpretation": resolved_interpretation,
        "department": department,
        "tender_category": tender_category,
        "verdict": verdict,
        "officer_action": officer_action,
        "officer_id": officer_id,
        "tender_id": tender_id,
        "criterion_id": criterion_id,
        "created_at": created_at,
    }


def get_cpm_stats(conn: sqlite3.Connection) -> dict:
    """Return CPM statistics.

    Provides aggregate statistics about the CPM corpus including
    total entries, breakdown by department and category, and whether
    the corpus has reached calibration threshold (50 entries).

    Args:
        conn: Active SQLite connection.

    Returns:
        Dict with keys: total_entries, by_department, by_category,
        calibration_ready.
    """
    conn.row_factory = sqlite3.Row

    # Total entries
    total = conn.execute("SELECT COUNT(*) as cnt FROM cpm_entries").fetchone()["cnt"]

    # By department
    dept_rows = conn.execute(
        "SELECT department, COUNT(*) as cnt FROM cpm_entries GROUP BY department"
    ).fetchall()
    by_department = {row["department"]: row["cnt"] for row in dept_rows}

    # By category
    cat_rows = conn.execute(
        "SELECT tender_category, COUNT(*) as cnt FROM cpm_entries GROUP BY tender_category"
    ).fetchall()
    by_category = {row["tender_category"]: row["cnt"] for row in cat_rows}

    return {
        "total_entries": total,
        "by_department": by_department,
        "by_category": by_category,
        "calibration_ready": total >= 50,
    }
