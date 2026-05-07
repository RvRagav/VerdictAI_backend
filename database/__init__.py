"""Database connection, schema, and seeding utilities.

Exports:
    init_db: Initialize the database schema on first run.
    seed_demo_data: Populate demo data (idempotent).
    get_db: Get a configured SQLite connection.
    get_db_connection: Context manager for transactions.
"""

from database.connection import get_db, get_db_connection, init_db
from database.seed import seed_demo_data

__all__ = ["init_db", "seed_demo_data", "get_db", "get_db_connection"]
