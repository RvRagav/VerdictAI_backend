"""CPM (Criterion Precedent Memory) API endpoints for VerdictAI.

Handles CPM search and statistics.

Endpoints:
- GET /cpm/search
- GET /cpm/stats

Requirements: 11.3, 16.1
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.database.connection import get_db
from backend.services.cpm_service import get_cpm_stats, search_cpm_precedents

router = APIRouter(prefix="/cpm", tags=["cpm"])


def _get_conn():
    """Dependency that provides a database connection."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


@router.get("/search")
async def search_precedents(
    query: str = Query(..., description="Criterion text to search for"),
    department: str = Query("", description="Department filter"),
    category: str = Query("", description="Tender category filter"),
    limit: int = Query(3, ge=1, le=3, description="Max results (capped at 3)"),
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Search CPM precedents by criterion text similarity.

    Uses FTS5 with BM25 ranking, filtered by department and category.
    Results are capped at 3.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail={
            "error": {
                "code": "EMPTY_QUERY",
                "message": "Query parameter cannot be empty",
            }
        })

    results = search_cpm_precedents(
        conn=conn,
        criterion_text=query,
        department=department,
        category=category,
        limit=limit,
    )

    return results


@router.get("/stats")
async def get_stats(
    conn: sqlite3.Connection = Depends(_get_conn),
):
    """Get CPM data statistics.

    Returns total entries, breakdown by department and category,
    and whether the corpus has reached calibration threshold.
    """
    stats = get_cpm_stats(conn)
    return stats
