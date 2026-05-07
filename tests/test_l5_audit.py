"""Unit tests for layers.l5_audit module."""

import json
import sqlite3

import pytest

from database.connection import get_db
from database.schema import create_tables
from layers.l5_audit import (
    GENESIS_HASH,
    append_audit_event,
    get_audit_trail,
    verify_hash_chain,
)
from utils.hash_utils import compute_entry_hash


@pytest.fixture
def db_conn():
    """Create an in-memory SQLite database with schema for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    # Insert a tender for FK constraint
    conn.execute(
        "INSERT INTO tenders (id, title, department, category, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("tender-1", "Test Tender", "PWD", "construction", "DOCUMENTS_UPLOADED", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    )
    conn.commit()
    yield conn
    conn.close()


class TestAppendAuditEvent:
    """Tests for append_audit_event function."""

    def test_first_event_uses_genesis_hash(self, db_conn):
        event = append_audit_event(
            db_conn, "tender-1", "document_received", {"doc_id": "d1"}, "system"
        )
        db_conn.commit()
        assert event["prev_hash"] == GENESIS_HASH

    def test_returns_complete_event_dict(self, db_conn):
        event = append_audit_event(
            db_conn, "tender-1", "document_received", {"doc_id": "d1"}, "system"
        )
        db_conn.commit()
        assert "id" in event
        assert event["tender_id"] == "tender-1"
        assert event["event_type"] == "document_received"
        assert event["event_data"] == {"doc_id": "d1"}
        assert event["actor"] == "system"
        assert event["timestamp"].endswith("Z")
        assert len(event["prev_hash"]) == 64
        assert len(event["entry_hash"]) == 64

    def test_second_event_links_to_first(self, db_conn):
        event1 = append_audit_event(
            db_conn, "tender-1", "document_received", {"doc_id": "d1"}, "system"
        )
        db_conn.commit()
        event2 = append_audit_event(
            db_conn, "tender-1", "ocr_completed", {"pages": 5}, "system"
        )
        db_conn.commit()
        assert event2["prev_hash"] == event1["entry_hash"]

    def test_entry_hash_is_correctly_computed(self, db_conn):
        event = append_audit_event(
            db_conn, "tender-1", "schema_approved", {"officer": "off1"}, "off1"
        )
        db_conn.commit()
        expected_hash = compute_entry_hash(
            event["event_type"],
            event["event_data"],
            event["actor"],
            event["timestamp"],
            event["prev_hash"],
        )
        assert event["entry_hash"] == expected_hash

    def test_event_persisted_in_database(self, db_conn):
        append_audit_event(
            db_conn, "tender-1", "document_received", {"doc_id": "d1"}, "system"
        )
        db_conn.commit()
        row = db_conn.execute("SELECT COUNT(*) as cnt FROM audit_events WHERE tender_id = ?", ("tender-1",)).fetchone()
        assert row["cnt"] == 1


class TestVerifyHashChain:
    """Tests for verify_hash_chain function."""

    def test_empty_chain_is_valid(self, db_conn):
        is_valid, error = verify_hash_chain(db_conn, "tender-1")
        assert is_valid is True
        assert error is None

    def test_single_event_chain_is_valid(self, db_conn):
        append_audit_event(db_conn, "tender-1", "document_received", {"x": 1}, "system")
        db_conn.commit()
        is_valid, error = verify_hash_chain(db_conn, "tender-1")
        assert is_valid is True
        assert error is None

    def test_multi_event_chain_is_valid(self, db_conn):
        for i in range(5):
            append_audit_event(db_conn, "tender-1", "document_received", {"i": i}, "system")
            db_conn.commit()
        is_valid, error = verify_hash_chain(db_conn, "tender-1")
        assert is_valid is True
        assert error is None

    def test_detects_tampered_entry_hash(self, db_conn):
        append_audit_event(db_conn, "tender-1", "document_received", {"x": 1}, "system")
        db_conn.commit()
        # Tamper with the entry_hash directly (bypass trigger by recreating without triggers)
        # We need to disable the trigger temporarily for this test
        db_conn.execute("DROP TRIGGER IF EXISTS audit_no_update")
        db_conn.execute(
            "UPDATE audit_events SET entry_hash = ? WHERE tender_id = ?",
            ("bad_hash" + "0" * 56, "tender-1"),
        )
        db_conn.commit()
        is_valid, error = verify_hash_chain(db_conn, "tender-1")
        assert is_valid is False
        assert "entry_hash mismatch" in error

    def test_detects_broken_prev_hash_link(self, db_conn):
        append_audit_event(db_conn, "tender-1", "document_received", {"x": 1}, "system")
        db_conn.commit()
        append_audit_event(db_conn, "tender-1", "ocr_completed", {"y": 2}, "system")
        db_conn.commit()
        # Tamper with second event's prev_hash
        db_conn.execute("DROP TRIGGER IF EXISTS audit_no_update")
        db_conn.execute(
            "UPDATE audit_events SET prev_hash = ? WHERE id = 2",
            ("0" * 64,),
        )
        db_conn.commit()
        is_valid, error = verify_hash_chain(db_conn, "tender-1")
        assert is_valid is False
        assert "prev_hash mismatch" in error or "entry_hash mismatch" in error


class TestGetAuditTrail:
    """Tests for get_audit_trail function."""

    def test_returns_empty_list_for_no_events(self, db_conn):
        result = get_audit_trail(db_conn, "tender-1")
        assert result == []

    def test_returns_all_events_for_tender(self, db_conn):
        append_audit_event(db_conn, "tender-1", "document_received", {"a": 1}, "system")
        db_conn.commit()
        append_audit_event(db_conn, "tender-1", "ocr_completed", {"b": 2}, "system")
        db_conn.commit()
        result = get_audit_trail(db_conn, "tender-1")
        assert len(result) == 2
        assert result[0]["event_type"] == "document_received"
        assert result[1]["event_type"] == "ocr_completed"

    def test_filters_by_event_type(self, db_conn):
        append_audit_event(db_conn, "tender-1", "document_received", {"a": 1}, "system")
        db_conn.commit()
        append_audit_event(db_conn, "tender-1", "ocr_completed", {"b": 2}, "system")
        db_conn.commit()
        result = get_audit_trail(db_conn, "tender-1", event_type="ocr_completed")
        assert len(result) == 1
        assert result[0]["event_type"] == "ocr_completed"

    def test_filters_by_date_range(self, db_conn):
        # Insert events and check date filtering works
        append_audit_event(db_conn, "tender-1", "document_received", {"a": 1}, "system")
        db_conn.commit()
        result = get_audit_trail(db_conn, "tender-1", from_date="2020-01-01T00:00:00Z")
        assert len(result) == 1
        # Filter with future from_date should return nothing
        result = get_audit_trail(db_conn, "tender-1", from_date="2099-01-01T00:00:00Z")
        assert len(result) == 0

    def test_event_data_is_parsed_as_dict(self, db_conn):
        append_audit_event(db_conn, "tender-1", "document_received", {"nested": {"key": "val"}}, "system")
        db_conn.commit()
        result = get_audit_trail(db_conn, "tender-1")
        assert isinstance(result[0]["event_data"], dict)
        assert result[0]["event_data"]["nested"]["key"] == "val"
