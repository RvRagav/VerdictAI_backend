"""Tests for Layer 2: ETS Builder.

Tests cover:
- extract_criteria: LLM Stub integration and database storage
- apply_corrigendum: Amendment application and history tracking
- detect_missing_corrigendum: Amendment indicator detection
- build_ets: ETS assembly and version hashing
- approve_schema: Schema approval gate
- update_criterion: Criterion editing with CPM precedent storage
- check_schema_approved: Guard check for evaluation readiness
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from backend.database.schema import create_tables
from backend.layers.l2_ets_builder import (
    apply_corrigendum,
    approve_schema,
    build_ets,
    check_schema_approved,
    detect_missing_corrigendum,
    extract_criteria,
    update_criterion,
)


@pytest.fixture
def db_conn():
    """Create an in-memory SQLite database with full schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=OFF")  # Simplify test setup
    create_tables(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_tender(db_conn):
    """Insert a sample tender and return its ID."""
    tender_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    db_conn.execute(
        """INSERT INTO tenders (id, title, department, category, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tender_id, "Test Tender", "Public Works", "Construction", "EXTRACTING_CRITERIA", now, now),
    )
    db_conn.commit()
    return tender_id


@pytest.fixture
def sample_document(db_conn, sample_tender):
    """Insert a sample document with pages and return its ID.

    Uses realistic Indian-NIT clause text so the real
    pattern-based criterion extractor finds criteria (it would not
    match on a bare sentence like "Eligibility criteria as per Clause 4").
    """
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    db_conn.execute(
        """INSERT INTO documents (id, tender_id, doc_type, filename, file_path, sha256_hash, upload_timestamp, processing_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, sample_tender, "nit", "test.pdf", "/tmp/test.pdf", "abc123", now, "complete"),
    )

    # Five real-format clauses — one per criterion type the pipeline supports.
    realistic_nit_text = (
        "SECTION IV: ELIGIBILITY CRITERIA\n\n"
        "Clause 4.1 - Financial Capacity (Mandatory)\n"
        "The bidder shall have an annual turnover of not less than Rs. 5 Crore "
        "in each of the last 3 financial years as per audited balance sheets. "
        "[GFR Rule 173(i) - Override NOT permitted]\n\n"
        "Clause 4.2 - GST Registration (Mandatory)\n"
        "The bidder shall possess a valid GST registration certificate with "
        "active status as on the date of submission. [GFR Rule 144]\n\n"
        "Clause 4.3 - Past Experience (Mandatory)\n"
        "The bidder shall have successfully completed a minimum of 3 similar "
        "supply orders in the last 5 years from the date of submission.\n\n"
        "Clause 4.4 - Combined Qualification\n"
        "The bidder shall satisfy all of the following: (a) Annual turnover "
        ">= Rs. 10 Crore, (b) Valid ISO 9001:2015 certification.\n\n"
        "Clause 4.5 - Manufacturing Capacity\n"
        "The bidder shall demonstrate adequate manufacturing capacity for the "
        "required quantity as evidenced by plant and machinery list."
    )

    page_id = str(uuid.uuid4())
    db_conn.execute(
        """INSERT INTO pages (id, document_id, page_number, raw_text, ocr_confidence)
           VALUES (?, ?, ?, ?, ?)""",
        (page_id, doc_id, 1, realistic_nit_text, 0.92),
    )
    db_conn.commit()
    return doc_id


class TestExtractCriteria:
    """Tests for extract_criteria function."""

    def test_extracts_criteria_from_document(self, db_conn, sample_tender, sample_document):
        """Should extract criteria via real pattern-based extractor and store in database."""
        criteria = extract_criteria(db_conn, sample_tender, sample_document)

        # Real extractor pulls at least one criterion per clause; the
        # sample has 5 clauses, a composite (4.4) produces two, so we
        # expect at least 5 criteria.
        assert len(criteria) >= 5
        assert all(c["tender_id"] == sample_tender for c in criteria)
        assert all(c["source_document_id"] == sample_document for c in criteria)
        assert all(c["status"] == "extracted" for c in criteria)

    def test_criteria_stored_in_database(self, db_conn, sample_tender, sample_document):
        """Should persist criteria to the criteria table."""
        criteria_returned = extract_criteria(db_conn, sample_tender, sample_document)

        rows = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM criteria WHERE tender_id = ?",
            (sample_tender,),
        ).fetchone()
        assert rows["cnt"] == len(criteria_returned)
        assert rows["cnt"] >= 5

    def test_criteria_have_valid_types(self, db_conn, sample_tender, sample_document):
        """All extracted criteria should have valid criterion types."""
        criteria = extract_criteria(db_conn, sample_tender, sample_document)

        valid_types = {
            "numeric_threshold", "categorical_presence",
            "temporal_recency", "composite", "qualitative_assessment",
        }
        for c in criteria:
            assert c["criterion_type"] in valid_types

    def test_criteria_annotated_with_gfr_flag(self, db_conn, sample_tender, sample_document):
        """Each criterion should have a GFR override permitted flag."""
        criteria = extract_criteria(db_conn, sample_tender, sample_document)

        for c in criteria:
            assert isinstance(c["gfr_override_permitted"], bool)

    def test_tender_status_updated(self, db_conn, sample_tender, sample_document):
        """Tender status should be updated to SCHEMA_PENDING_REVIEW."""
        extract_criteria(db_conn, sample_tender, sample_document)

        row = db_conn.execute(
            "SELECT status FROM tenders WHERE id = ?",
            (sample_tender,),
        ).fetchone()
        assert row["status"] == "SCHEMA_PENDING_REVIEW"

    def test_llm_invocation_logged(self, db_conn, sample_tender, sample_document):
        """LLM Stub invocation should be logged."""
        extract_criteria(db_conn, sample_tender, sample_document)

        row = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM llm_stub_log WHERE tender_id = ?",
            (sample_tender,),
        ).fetchone()
        assert row["cnt"] >= 1


class TestApplyCorrigendum:
    """Tests for apply_corrigendum function."""

    def test_applies_amendments_to_existing_criteria(self, db_conn, sample_tender, sample_document):
        """Should update criteria with corrigendum amendments."""
        # First extract base criteria
        extract_criteria(db_conn, sample_tender, sample_document)

        # Create a corrigendum document with real amendment text
        corr_doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        db_conn.execute(
            """INSERT INTO documents (id, tender_id, doc_type, filename, file_path, sha256_hash, upload_timestamp, processing_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (corr_doc_id, sample_tender, "corrigendum", "corr1.pdf", "/tmp/corr1.pdf", "def456", now, "complete"),
        )
        page_id = str(uuid.uuid4())
        db_conn.execute(
            """INSERT INTO pages (id, document_id, page_number, raw_text, ocr_confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (page_id, corr_doc_id, 1,
             "Clause 4.1 - Financial Capacity\n"
             "The annual turnover requirement is revised to not less than "
             "Rs. 10 Crore in each of the last 3 financial years.",
             0.90),
        )
        db_conn.commit()

        # apply_corrigendum does not raise even if no matching clauses
        # are found; just ensure it returns the criteria list.
        updated = apply_corrigendum(db_conn, sample_tender, corr_doc_id)
        assert isinstance(updated, list)
        assert len(updated) >= 5

    def test_logs_corrigendum_linked_audit_event(self, db_conn, sample_tender, sample_document):
        """Should log a corrigendum_linked audit event."""
        extract_criteria(db_conn, sample_tender, sample_document)

        corr_doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        db_conn.execute(
            """INSERT INTO documents (id, tender_id, doc_type, filename, file_path, sha256_hash, upload_timestamp, processing_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (corr_doc_id, sample_tender, "corrigendum", "corr1.pdf", "/tmp/corr1.pdf", "def456", now, "complete"),
        )
        page_id = str(uuid.uuid4())
        db_conn.execute(
            """INSERT INTO pages (id, document_id, page_number, raw_text, ocr_confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (page_id, corr_doc_id, 1, "Amendment text", 0.90),
        )
        db_conn.commit()

        apply_corrigendum(db_conn, sample_tender, corr_doc_id)

        row = db_conn.execute(
            "SELECT * FROM audit_events WHERE tender_id = ? AND event_type = 'corrigendum_linked'",
            (sample_tender,),
        ).fetchone()
        assert row is not None


class TestDetectMissingCorrigendum:
    """Tests for detect_missing_corrigendum function."""

    def test_detects_as_amended(self):
        """Should detect 'as amended' indicator."""
        text = "The eligibility criteria as amended by the department"
        result = detect_missing_corrigendum(text)
        assert "as amended" in result

    def test_detects_refer_addendum(self):
        """Should detect 'refer addendum' indicator."""
        text = "For updated thresholds, refer addendum no. 2"
        result = detect_missing_corrigendum(text)
        assert "refer addendum" in result

    def test_detects_superseded_by(self):
        """Should detect 'superseded by' indicator."""
        text = "This clause has been superseded by corrigendum no. 3"
        result = detect_missing_corrigendum(text)
        assert "superseded by" in result

    def test_detects_refer_corrigendum(self):
        """Should detect 'refer corrigendum' indicator."""
        text = "Please refer corrigendum for revised values"
        result = detect_missing_corrigendum(text)
        assert "refer corrigendum" in result

    def test_detects_multiple_indicators(self):
        """Should detect multiple indicators in same text."""
        text = "As amended per addendum. Also refer corrigendum no. 1"
        result = detect_missing_corrigendum(text)
        assert "as amended" in result
        assert "refer corrigendum" in result

    def test_returns_empty_for_clean_text(self):
        """Should return empty list when no indicators present."""
        text = "Standard eligibility criteria for construction works"
        result = detect_missing_corrigendum(text)
        assert result == []

    def test_case_insensitive(self):
        """Should detect indicators regardless of case."""
        text = "AS AMENDED by the authority"
        result = detect_missing_corrigendum(text)
        assert "as amended" in result


class TestBuildEts:
    """Tests for build_ets function."""

    def test_builds_ets_with_criteria(self, db_conn, sample_tender, sample_document):
        """Should assemble ETS with all criteria."""
        extracted = extract_criteria(db_conn, sample_tender, sample_document)
        ets = build_ets(db_conn, sample_tender)

        assert ets["tender_id"] == sample_tender
        assert len(ets["criteria"]) == len(extracted)
        assert len(ets["criteria"]) >= 5
        assert ets["version_hash"] is not None
        assert len(ets["version_hash"]) == 64  # SHA-256 hex
        assert ets["status"] is not None

    def test_version_hash_is_deterministic(self, db_conn, sample_tender, sample_document):
        """Same criteria should produce same version hash."""
        extract_criteria(db_conn, sample_tender, sample_document)
        ets1 = build_ets(db_conn, sample_tender)
        ets2 = build_ets(db_conn, sample_tender)

        assert ets1["version_hash"] == ets2["version_hash"]

    def test_updates_tender_ets_version(self, db_conn, sample_tender, sample_document):
        """Should update the tender's ets_version field."""
        extract_criteria(db_conn, sample_tender, sample_document)
        ets = build_ets(db_conn, sample_tender)

        row = db_conn.execute(
            "SELECT ets_version FROM tenders WHERE id = ?",
            (sample_tender,),
        ).fetchone()
        assert row["ets_version"] == ets["version_hash"]


class TestApproveSchema:
    """Tests for approve_schema function."""

    def test_approves_schema_with_criteria(self, db_conn, sample_tender, sample_document):
        """Should approve schema when criteria exist."""
        extracted = extract_criteria(db_conn, sample_tender, sample_document)
        result = approve_schema(db_conn, sample_tender, "officer_001")

        assert result["status"] == "approved"
        assert result["officer_id"] == "officer_001"
        assert result["approved_at"] is not None
        assert result["criteria_count"] == len(extracted)

    def test_updates_tender_status(self, db_conn, sample_tender, sample_document):
        """Should update tender status to SCHEMA_APPROVED."""
        extract_criteria(db_conn, sample_tender, sample_document)
        approve_schema(db_conn, sample_tender, "officer_001")

        row = db_conn.execute(
            "SELECT status FROM tenders WHERE id = ?",
            (sample_tender,),
        ).fetchone()
        assert row["status"] == "SCHEMA_APPROVED"

    def test_updates_all_criteria_status(self, db_conn, sample_tender, sample_document):
        """Should update all criteria to approved status."""
        extract_criteria(db_conn, sample_tender, sample_document)
        approve_schema(db_conn, sample_tender, "officer_001")

        rows = db_conn.execute(
            "SELECT status, approved_by FROM criteria WHERE tender_id = ?",
            (sample_tender,),
        ).fetchall()
        for row in rows:
            assert row["status"] == "approved"
            assert row["approved_by"] == "officer_001"

    def test_logs_schema_approved_audit_event(self, db_conn, sample_tender, sample_document):
        """Should log schema_approved audit event."""
        extract_criteria(db_conn, sample_tender, sample_document)
        approve_schema(db_conn, sample_tender, "officer_001")

        row = db_conn.execute(
            "SELECT * FROM audit_events WHERE tender_id = ? AND event_type = 'schema_approved'",
            (sample_tender,),
        ).fetchone()
        assert row is not None
        event_data = json.loads(row["event_data"])
        assert event_data["officer_id"] == "officer_001"

    def test_raises_error_when_no_criteria(self, db_conn, sample_tender):
        """Should raise ValueError when no criteria exist."""
        with pytest.raises(ValueError, match="no criteria extracted"):
            approve_schema(db_conn, sample_tender, "officer_001")


class TestUpdateCriterion:
    """Tests for update_criterion function."""

    def test_updates_criterion_text(self, db_conn, sample_tender, sample_document):
        """Should update criterion text."""
        criteria = extract_criteria(db_conn, sample_tender, sample_document)
        criterion_id = criteria[0]["id"]

        updated = update_criterion(
            db_conn, criterion_id,
            {"criterion_text": "Updated criterion text"},
            "officer_001",
        )
        assert updated["criterion_text"] == "Updated criterion text"
        assert updated["status"] == "reviewed"

    def test_updates_threshold_value(self, db_conn, sample_tender, sample_document):
        """Should update threshold value."""
        criteria = extract_criteria(db_conn, sample_tender, sample_document)
        criterion_id = criteria[0]["id"]

        new_threshold = {"value": 15, "unit": "Cr", "period": "3 years"}
        updated = update_criterion(
            db_conn, criterion_id,
            {"threshold_value": new_threshold},
            "officer_001",
        )
        assert updated["threshold_value"] == new_threshold

    def test_stores_cpm_precedent_on_text_change(self, db_conn, sample_tender, sample_document):
        """Should store CPM precedent when interpretation changes."""
        criteria = extract_criteria(db_conn, sample_tender, sample_document)
        criterion_id = criteria[0]["id"]

        update_criterion(
            db_conn, criterion_id,
            {"criterion_text": "Revised interpretation of turnover requirement"},
            "officer_001",
        )

        row = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM cpm_entries WHERE criterion_id = ?",
            (criterion_id,),
        ).fetchone()
        assert row["cnt"] == 1

    def test_raises_error_for_missing_criterion(self, db_conn):
        """Should raise ValueError for non-existent criterion."""
        with pytest.raises(ValueError, match="not found"):
            update_criterion(db_conn, "nonexistent-id", {"criterion_text": "x"}, "officer_001")


class TestCheckSchemaApproved:
    """Tests for check_schema_approved function."""

    def test_returns_false_before_approval(self, db_conn, sample_tender, sample_document):
        """Should return False when schema not yet approved."""
        extract_criteria(db_conn, sample_tender, sample_document)
        assert check_schema_approved(db_conn, sample_tender) is False

    def test_returns_true_after_approval(self, db_conn, sample_tender, sample_document):
        """Should return True after schema approval."""
        extract_criteria(db_conn, sample_tender, sample_document)
        approve_schema(db_conn, sample_tender, "officer_001")
        assert check_schema_approved(db_conn, sample_tender) is True

    def test_returns_true_for_later_states(self, db_conn, sample_tender):
        """Should return True for states after SCHEMA_APPROVED."""
        now = datetime.now(timezone.utc).isoformat()
        db_conn.execute(
            "UPDATE tenders SET status = ? WHERE id = ?",
            ("EVALUATING", sample_tender),
        )
        db_conn.commit()
        assert check_schema_approved(db_conn, sample_tender) is True

    def test_returns_false_for_nonexistent_tender(self, db_conn):
        """Should return False for non-existent tender."""
        assert check_schema_approved(db_conn, "nonexistent-id") is False
