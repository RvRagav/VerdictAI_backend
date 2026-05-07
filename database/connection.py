"""SQLite connection factory with WAL mode and transaction handling.

Provides database connection utilities for VerdictAI:
- get_db(): Returns a configured SQLite connection with WAL mode
- get_db_connection(): Context manager for transaction handling
- init_db(): Initializes the database schema on first run
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import settings


def get_db(db_path: str | None = None) -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode enabled and Row factory.

    Args:
        db_path: Optional path to the database file. Defaults to settings.db_path.

    Returns:
        A configured sqlite3.Connection with WAL journal mode and Row factory.
    """
    path = db_path or settings.db_path
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db_connection(db_path: str | None = None):
    """Context manager for database transactions.

    Auto-commits on successful exit, rolls back on exception.

    Args:
        db_path: Optional path to the database file. Defaults to settings.db_path.

    Yields:
        A configured sqlite3.Connection.

    Example:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO tenders ...")
    """
    conn = get_db(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Initialize the database by creating all tables and seeding demo data.

    Creates the schema on first run and populates demo data.
    Both operations are idempotent (safe to call multiple times).

    Args:
        db_path: Optional path to the database file. Defaults to settings.db_path.
    """
    from database.schema import create_tables
    from database.seed import seed_demo_data

    with get_db_connection(db_path) as conn:
        create_tables(conn)
        seed_demo_data(conn)
