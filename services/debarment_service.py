"""Debarment checking service for VerdictAI.

Cross-references bidder company names and PAN numbers against the
debarment_list table to detect excluded entities. Uses the Entity
Matcher for fuzzy name matching when PAN is unavailable.

Requirements: 5.1, 5.2, 5.3
"""

import sqlite3

from services.entity_matcher import normalise_company_name

from difflib import SequenceMatcher


def check_debarment(
    conn: sqlite3.Connection,
    company_name: str,
    pan_number: str | None = None,
) -> dict:
    """Cross-reference bidder against debarment_list table.

    Matching logic:
    1. If PAN provided, exact match on pan_number
    2. Fuzzy match on entity_name using normalise_company_name
    3. Return all matches found

    Args:
        conn: Active SQLite connection.
        company_name: The bidder's company name.
        pan_number: Optional PAN number for exact matching.

    Returns:
        Dict with keys:
        - is_debarred: bool indicating if any match was found
        - matches: list of match dicts with entity_name, pan_number,
          reason, date, source
        - check_method: "pan_match", "name_match", or "clear"
    """
    conn.row_factory = sqlite3.Row
    matches = []

    # Step 1: If PAN provided, try exact PAN match
    if pan_number:
        pan_rows = conn.execute(
            "SELECT * FROM debarment_list WHERE pan_number = ?",
            (pan_number,),
        ).fetchall()

        if pan_rows:
            for row in pan_rows:
                matches.append({
                    "entity_name": row["entity_name"],
                    "pan_number": row["pan_number"],
                    "reason": row["debarment_reason"],
                    "date": row["debarment_date"],
                    "source": row["source"],
                })
            return {
                "is_debarred": True,
                "matches": matches,
                "check_method": "pan_match",
            }

    # Step 2: Fuzzy match on entity_name
    all_entries = conn.execute(
        "SELECT * FROM debarment_list"
    ).fetchall()

    norm_company = normalise_company_name(company_name)

    for row in all_entries:
        norm_entity = normalise_company_name(row["entity_name"])

        # Exact match after normalisation
        if norm_company == norm_entity:
            matches.append({
                "entity_name": row["entity_name"],
                "pan_number": row["pan_number"],
                "reason": row["debarment_reason"],
                "date": row["debarment_date"],
                "source": row["source"],
            })
            continue

        # Fuzzy similarity check
        score = SequenceMatcher(None, norm_company, norm_entity).ratio()
        if score >= 0.85:
            matches.append({
                "entity_name": row["entity_name"],
                "pan_number": row["pan_number"],
                "reason": row["debarment_reason"],
                "date": row["debarment_date"],
                "source": row["source"],
            })

    if matches:
        return {
            "is_debarred": True,
            "matches": matches,
            "check_method": "name_match",
        }

    # Step 3: No matches found
    return {
        "is_debarred": False,
        "matches": [],
        "check_method": "clear",
    }
