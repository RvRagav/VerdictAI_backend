"""Unit tests for Layer 3: Evidence Extraction."""

import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from database.connection import get_db
from database.schema import create_tables
from database.seed import seed_demo_data
from layers.l3_evidence import (
    extract_evidence,
    _compute_extraction_confidence,
    _empty_evidence,
)


@pytest.fixture
def db_conn():
    """Create an in-memory SQLite database with schema and seed data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    seed_demo_data(conn)
    yield conn
    conn.close()


@pytest.fixture
def setup_tender_data(db_conn):
    """Set up a tender with bidder and criteria for testing."""
    tender_id = str(uuid.uuid4())
    bidder_id = str(uuid.uuid4())
    criterion_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    # Create tender
    db_conn.execute(
        """INSERT INTO tenders (id, title, department, category, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tender_id, "Test Tender", "Public Works", "Works",
         "EVALUATING", datetime.now(timezone.utc).isoformat(),
         datetime.now(timezone.utc).isoformat()),
    )

    # Create bidder
    db_conn.execute(
        """INSERT INTO bidders (id, tender_id, company_name, pan_number, registration_number, status, debarment_status)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (bidder_id, tender_id, "ABC Infrastructure Pvt Ltd", "ABCDE1234F",
         "REG001", "pending", "clear"),
    )

    # Create document for bidder
    db_conn.execute(
        """INSERT INTO documents (id, tender_id, bidder_id, doc_type, filename, file_path, sha256_hash, page_count, avg_ocr_confidence, upload_timestamp, processing_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, tender_id, bidder_id, "bidder_submission", "submission.pdf",
         "/tmp/submission.pdf", "abc123hash", 10, 0.88,
         datetime.now(timezone.utc).isoformat(), "complete"),
    )

    # Seed realistic OCR text so the real evidence extractor has
    # something to regex over. Needs to name the bidder ("ABC
    # Infrastructure") so the entity matcher returns a clean match and
    # include a turnover figure and a GSTIN.
    page_id = str(uuid.uuid4())
    db_conn.execute(
        """INSERT INTO pages (id, document_id, page_number, raw_text, ocr_confidence)
        VALUES (?, ?, ?, ?, ?)""",
        (
            page_id,
            doc_id,
            1,
            "M/s ABC Infrastructure Pvt Ltd\n"
            "PAN: ABCDE1234F   GST: 07ABCDE1234F1Z5\n"
            "Annual turnover for FY 2023-24: Rs. 15.50 Crore\n"
            "Valid up to 31-12-2026.\n"
            "Completed 3 similar supply orders in the last 5 years.",
            0.88,
        ),
    )

    # Create numeric_threshold criterion
    db_conn.execute(
        """INSERT INTO criteria (id, tender_id, criterion_text, criterion_type, threshold_value, gfr_override_permitted, is_mandatory, source_clause_ref, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (criterion_id, tender_id,
         "Average annual turnover of last 3 financial years",
         "numeric_threshold",
         '{"value": 10, "unit": "Cr", "period": "3 years"}',
         0, 1, "Clause 4.1(a)", "approved"),
    )

    db_conn.commit()

    return {
        "tender_id": tender_id,
        "bidder_id": bidder_id,
        "criterion_id": criterion_id,
        "doc_id": doc_id,
    }


class TestExtractEvidence:
    """Tests for the extract_evidence function."""

    def test_extract_evidence_numeric_threshold(self, db_conn, setup_tender_data):
        """Test evidence extraction for numeric_threshold criterion."""
        data = setup_tender_data
        result = extract_evidence(
            db_conn, data["tender_id"], data["bidder_id"], data["criterion_id"]
        )

        assert result is not None
        assert result["tender_id"] == data["tender_id"]
        assert result["bidder_id"] == data["bidder_id"]
        assert result["criterion_id"] == data["criterion_id"]
        assert result["criterion_type"] == "numeric_threshold"
        assert result["value"] is not None
        assert "amount" in result["value"]
        assert result["source_document_id"] == data["doc_id"]
        assert result["extraction_confidence"] > 0.0
        assert isinstance(result["entity_match_flag"], bool)

    def test_extract_evidence_missing_criterion(self, db_conn, setup_tender_data):
        """Test evidence extraction with non-existent criterion."""
        data = setup_tender_data
        result = extract_evidence(
            db_conn, data["tender_id"], data["bidder_id"], "nonexistent-id"
        )

        assert result["value"] is None
        assert result["extraction_confidence"] == 0.0

    def test_extract_evidence_missing_bidder(self, db_conn, setup_tender_data):
        """Test evidence extraction with non-existent bidder."""
        data = setup_tender_data
        result = extract_evidence(
            db_conn, data["tender_id"], "nonexistent-id", data["criterion_id"]
        )

        assert result["value"] is None
        assert result["extraction_confidence"] == 0.0

    def test_extract_evidence_categorical_presence(self, db_conn, setup_tender_data):
        """Test evidence extraction for categorical_presence criterion."""
        data = setup_tender_data
        cat_criterion_id = str(uuid.uuid4())

        db_conn.execute(
            """INSERT INTO criteria (id, tender_id, criterion_text, criterion_type, threshold_value, gfr_override_permitted, is_mandatory, source_clause_ref, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cat_criterion_id, data["tender_id"],
             "Valid GST registration certificate",
             "categorical_presence",
             '{"required": true}',
             0, 1, "Clause 4.1(b)", "approved"),
        )
        db_conn.commit()

        result = extract_evidence(
            db_conn, data["tender_id"], data["bidder_id"], cat_criterion_id
        )

        assert result is not None
        assert result["criterion_type"] == "categorical_presence"
        assert result["value"] is not None
        assert "found" in result["value"]

    def test_extract_evidence_logs_audit_event(self, db_conn, setup_tender_data):
        """Test that evidence extraction logs an audit event."""
        data = setup_tender_data
        extract_evidence(
            db_conn, data["tender_id"], data["bidder_id"], data["criterion_id"]
        )

        # Check audit event was logged
        events = db_conn.execute(
            "SELECT * FROM audit_events WHERE tender_id = ? AND event_type = 'evidence_extracted'",
            (data["tender_id"],),
        ).fetchall()

        assert len(events) >= 1
        assert events[0]["event_type"] == "evidence_extracted"


class TestComputeExtractionConfidence:
    """Tests for the _compute_extraction_confidence helper."""

    def test_zero_ocr_confidence(self):
        """Zero OCR confidence should return zero extraction confidence."""
        result = _compute_extraction_confidence(0.0, "numeric_threshold", False)
        assert result == 0.0

    def test_entity_mismatch_penalty(self):
        """Entity mismatch should reduce confidence by 50%."""
        without_flag = _compute_extraction_confidence(0.9, "numeric_threshold", False)
        with_flag = _compute_extraction_confidence(0.9, "numeric_threshold", True)
        assert with_flag < without_flag
        assert with_flag == pytest.approx(0.9 * 0.5 * 0.95)

    def test_qualitative_lower_confidence(self):
        """Qualitative assessments should have lower extraction confidence."""
        numeric = _compute_extraction_confidence(0.9, "numeric_threshold", False)
        qualitative = _compute_extraction_confidence(0.9, "qualitative_assessment", False)
        assert qualitative < numeric

    def test_confidence_clamped_to_unit_interval(self):
        """Confidence should always be in [0.0, 1.0]."""
        result = _compute_extraction_confidence(1.0, "numeric_threshold", False)
        assert 0.0 <= result <= 1.0

        result = _compute_extraction_confidence(0.5, "categorical_presence", True)
        assert 0.0 <= result <= 1.0


class TestEmptyEvidence:
    """Tests for the _empty_evidence helper."""

    def test_empty_evidence_structure(self):
        """Empty evidence should have all required fields."""
        result = _empty_evidence("t1", "b1", "c1")
        assert result["value"] is None
        assert result["extraction_confidence"] == 0.0
        assert result["entity_match_flag"] is False
        assert result["tender_id"] == "t1"
        assert result["bidder_id"] == "b1"
        assert result["criterion_id"] == "c1"
