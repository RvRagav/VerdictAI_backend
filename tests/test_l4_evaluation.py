"""Unit tests for Layer 4: Evaluation Engine."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from database.schema import create_tables
from database.seed import seed_demo_data
from layers.l4_evaluation import (
    compute_route,
    evaluate_criterion,
    evaluate_all_bidders,
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
def setup_evaluation_data(db_conn):
    """Set up a complete tender with bidders and criteria for evaluation testing."""
    tender_id = str(uuid.uuid4())
    bidder_id = str(uuid.uuid4())
    criterion_id_numeric = str(uuid.uuid4())
    criterion_id_categorical = str(uuid.uuid4())
    criterion_id_temporal = str(uuid.uuid4())
    criterion_id_qualitative = str(uuid.uuid4())
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
        (bidder_id, tender_id, "XYZ Construction Pvt Ltd", "XYZAB5678G",
         "REG002", "pending", "clear"),
    )

    # Create document for bidder
    db_conn.execute(
        """INSERT INTO documents (id, tender_id, bidder_id, doc_type, filename, file_path, sha256_hash, page_count, avg_ocr_confidence, upload_timestamp, processing_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, tender_id, bidder_id, "bidder_submission", "xyz_submission.pdf",
         "/tmp/xyz_submission.pdf", "xyz123hash", 15, 0.90,
         datetime.now(timezone.utc).isoformat(), "complete"),
    )

    # Create criteria (all approved)
    db_conn.execute(
        """INSERT INTO criteria (id, tender_id, criterion_text, criterion_type, threshold_value, gfr_override_permitted, is_mandatory, source_clause_ref, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (criterion_id_numeric, tender_id,
         "Average annual turnover of last 3 financial years",
         "numeric_threshold",
         '{"value": 10, "unit": "Cr", "period": "3 years"}',
         0, 1, "Clause 4.1(a)", "approved"),
    )

    db_conn.execute(
        """INSERT INTO criteria (id, tender_id, criterion_text, criterion_type, threshold_value, gfr_override_permitted, is_mandatory, source_clause_ref, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (criterion_id_categorical, tender_id,
         "Valid GST registration certificate",
         "categorical_presence",
         '{"required": true}',
         0, 1, "Clause 4.1(b)", "approved"),
    )

    db_conn.execute(
        """INSERT INTO criteria (id, tender_id, criterion_text, criterion_type, threshold_value, gfr_override_permitted, is_mandatory, source_clause_ref, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (criterion_id_temporal, tender_id,
         "Completion of at least 2 similar works in last 5 years",
         "temporal_recency",
         '{"count": 2, "period": "5 years", "work_type": "similar"}',
         1, 0, "Clause 4.2(a)", "approved"),
    )

    db_conn.execute(
        """INSERT INTO criteria (id, tender_id, criterion_text, criterion_type, threshold_value, gfr_override_permitted, is_mandatory, source_clause_ref, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (criterion_id_qualitative, tender_id,
         "Technical capability assessment for RCC structures",
         "qualitative_assessment",
         '{"assessment_area": "RCC structures"}',
         1, 0, "Clause 4.3", "approved"),
    )

    db_conn.commit()

    return {
        "tender_id": tender_id,
        "bidder_id": bidder_id,
        "criterion_id_numeric": criterion_id_numeric,
        "criterion_id_categorical": criterion_id_categorical,
        "criterion_id_temporal": criterion_id_temporal,
        "criterion_id_qualitative": criterion_id_qualitative,
        "doc_id": doc_id,
    }


class TestComputeRoute:
    """Tests for the compute_route function (Confidence Router)."""

    def test_rule1_mandatory_fail_routes_to_mandatory_review(self):
        """Rule 1: Mandatory FAIL always routes to mandatory_review."""
        result = compute_route(
            verdict="FAIL",
            confidence=0.95,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=True,
            gfr_override_permitted=False,
            cpm_data_count=100,
        )
        assert result["route"] == "mandatory_review"
        assert "Mandatory criterion FAIL" in result["reasons"][0]

    def test_rule1_mandatory_pass_does_not_trigger(self):
        """Rule 1 only applies to FAIL verdicts."""
        result = compute_route(
            verdict="PASS",
            confidence=0.95,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=True,
            gfr_override_permitted=False,
            cpm_data_count=100,
        )
        assert result["route"] == "auto_commit"

    def test_rule2_flags_force_mandatory_review(self):
        """Rule 2: Flags present → mandatory_review."""
        result = compute_route(
            verdict="PASS",
            confidence=0.99,
            criterion_type="numeric_threshold",
            flags=["entity_mismatch"],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert result["route"] == "mandatory_review"
        assert "entity_mismatch" in result["reasons"][0]

    def test_rule3_low_confidence_mandatory_review(self):
        """Rule 3: Low confidence (< floor) → mandatory_review."""
        result = compute_route(
            verdict="PASS",
            confidence=0.30,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert result["route"] == "mandatory_review"
        assert "below mandatory floor" in result["reasons"][0]

    def test_rule4_llm_fail_routes_to_hitl(self):
        """Rule 4: LLM FAIL → hitl_review (never auto-commit)."""
        result = compute_route(
            verdict="FAIL",
            confidence=0.95,
            criterion_type="qualitative_assessment",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert result["route"] == "hitl_review"
        assert "LLM-based FAIL" in result["reasons"][0]

    def test_rule5_medium_confidence_hitl_review(self):
        """Rule 5: Medium confidence → hitl_review."""
        result = compute_route(
            verdict="PASS",
            confidence=0.70,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert result["route"] == "hitl_review"
        assert "below auto-commit threshold" in result["reasons"][0]

    def test_rule6_high_confidence_deterministic_auto_commit(self):
        """Rule 6: High confidence + deterministic + no flags → auto_commit."""
        result = compute_route(
            verdict="PASS",
            confidence=0.92,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert result["route"] == "auto_commit"
        assert "High confidence deterministic" in result["reasons"][0]

    def test_qualitative_high_confidence_still_hitl(self):
        """Qualitative assessment at high confidence PASS still routes to HITL."""
        result = compute_route(
            verdict="PASS",
            confidence=0.95,
            criterion_type="qualitative_assessment",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert result["route"] == "hitl_review"
        assert "Qualitative assessment requires review" in result["reasons"][0]

    def test_conservative_thresholds_low_cpm(self):
        """Conservative thresholds when cpm_data_count < 50."""
        # With standard thresholds (cpm >= 50), 0.87 would auto-commit
        result_standard = compute_route(
            verdict="PASS",
            confidence=0.87,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert result_standard["route"] == "auto_commit"

        # With conservative thresholds (cpm < 50), 0.87 routes to HITL
        result_conservative = compute_route(
            verdict="PASS",
            confidence=0.87,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=30,
        )
        assert result_conservative["route"] == "hitl_review"

    def test_conservative_mandatory_floor(self):
        """Conservative mandatory floor is 0.60 when cpm < 50."""
        # With standard floor (0.50), confidence 0.55 routes to HITL
        result_standard = compute_route(
            verdict="PASS",
            confidence=0.55,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert result_standard["route"] == "hitl_review"

        # With conservative floor (0.60), confidence 0.55 routes to mandatory
        result_conservative = compute_route(
            verdict="PASS",
            confidence=0.55,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=30,
        )
        assert result_conservative["route"] == "mandatory_review"

    def test_routing_decision_structure(self):
        """Routing decision should have all required fields."""
        result = compute_route(
            verdict="PASS",
            confidence=0.92,
            criterion_type="numeric_threshold",
            flags=[],
            is_mandatory=False,
            gfr_override_permitted=True,
            cpm_data_count=100,
        )
        assert "route" in result
        assert "confidence" in result
        assert "reasons" in result
        assert "flags" in result
        assert "gfr_override_permitted" in result
        assert "is_mandatory_criterion" in result

    def test_priority_order_mandatory_fail_over_flags(self):
        """Rule 1 takes priority over Rule 2."""
        result = compute_route(
            verdict="FAIL",
            confidence=0.95,
            criterion_type="numeric_threshold",
            flags=["entity_mismatch"],
            is_mandatory=True,
            gfr_override_permitted=False,
            cpm_data_count=100,
        )
        assert result["route"] == "mandatory_review"
        assert "Mandatory criterion FAIL" in result["reasons"][0]

    def test_all_deterministic_types_can_auto_commit(self):
        """All three deterministic types can reach auto_commit."""
        for ctype in ("numeric_threshold", "categorical_presence", "temporal_recency"):
            result = compute_route(
                verdict="PASS",
                confidence=0.92,
                criterion_type=ctype,
                flags=[],
                is_mandatory=False,
                gfr_override_permitted=True,
                cpm_data_count=100,
            )
            assert result["route"] == "auto_commit", f"Failed for {ctype}"


class TestEvaluateCriterion:
    """Tests for the evaluate_criterion function."""

    def test_evaluate_numeric_criterion(self, db_conn, setup_evaluation_data):
        """Test evaluation of a numeric_threshold criterion."""
        data = setup_evaluation_data
        result = evaluate_criterion(
            db_conn, data["tender_id"], data["bidder_id"], data["criterion_id_numeric"]
        )

        assert result is not None
        assert result["tender_id"] == data["tender_id"]
        assert result["bidder_id"] == data["bidder_id"]
        assert result["criterion_id"] == data["criterion_id_numeric"]
        assert result["verdict"] in ("PASS", "FAIL", "REVIEW")
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["evaluation_method"] == "deterministic"
        assert result["route"] in ("auto_commit", "hitl_review", "mandatory_review")

    def test_evaluate_qualitative_criterion(self, db_conn, setup_evaluation_data):
        """Test evaluation of a qualitative_assessment criterion."""
        data = setup_evaluation_data
        result = evaluate_criterion(
            db_conn, data["tender_id"], data["bidder_id"], data["criterion_id_qualitative"]
        )

        assert result is not None
        assert result["evaluation_method"] == "llm_stub"
        # Qualitative should never auto-commit
        assert result["route"] in ("hitl_review", "mandatory_review")

    def test_evaluate_stores_in_database(self, db_conn, setup_evaluation_data):
        """Test that evaluation is stored in the evaluations table."""
        data = setup_evaluation_data
        result = evaluate_criterion(
            db_conn, data["tender_id"], data["bidder_id"], data["criterion_id_numeric"]
        )

        # Verify stored in database
        stored = db_conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (result["id"],)
        ).fetchone()

        assert stored is not None
        assert stored["verdict"] == result["verdict"]
        assert stored["route"] == result["route"]

    def test_evaluate_logs_audit_events(self, db_conn, setup_evaluation_data):
        """Test that evaluation logs verdict_computed and case_routed events."""
        data = setup_evaluation_data
        evaluate_criterion(
            db_conn, data["tender_id"], data["bidder_id"], data["criterion_id_numeric"]
        )

        # Check for verdict_computed event
        verdict_events = db_conn.execute(
            "SELECT * FROM audit_events WHERE tender_id = ? AND event_type = 'verdict_computed'",
            (data["tender_id"],),
        ).fetchall()
        assert len(verdict_events) >= 1

        # Check for case_routed event
        route_events = db_conn.execute(
            "SELECT * FROM audit_events WHERE tender_id = ? AND event_type = 'case_routed'",
            (data["tender_id"],),
        ).fetchall()
        assert len(route_events) >= 1

    def test_evaluate_nonexistent_criterion(self, db_conn, setup_evaluation_data):
        """Test evaluation with non-existent criterion returns error evaluation."""
        data = setup_evaluation_data
        result = evaluate_criterion(
            db_conn, data["tender_id"], data["bidder_id"], "nonexistent-id"
        )

        assert result["verdict"] == "REVIEW"
        assert result["route"] == "mandatory_review"
        assert "error" in result["routing_reason"].lower()


class TestEvaluateAllBidders:
    """Tests for the evaluate_all_bidders function."""

    def test_evaluate_all_bidders_basic(self, db_conn, setup_evaluation_data):
        """Test evaluating all bidders for a tender."""
        data = setup_evaluation_data
        results = evaluate_all_bidders(db_conn, data["tender_id"])

        assert len(results) >= 1
        bidder_result = results[0]
        assert bidder_result["bidder_id"] == data["bidder_id"]
        assert bidder_result["status"] in ("evaluated", "debarment_flagged")

        if bidder_result["status"] == "evaluated":
            # Should have evaluations for all approved criteria
            assert len(bidder_result["evaluations"]) == 4

    def test_evaluate_all_bidders_debarment_check(self, db_conn, setup_evaluation_data):
        """Test that debarment check is performed before evaluation."""
        data = setup_evaluation_data
        evaluate_all_bidders(db_conn, data["tender_id"])

        # Check debarment_checked audit event exists
        events = db_conn.execute(
            "SELECT * FROM audit_events WHERE tender_id = ? AND event_type = 'debarment_checked'",
            (data["tender_id"],),
        ).fetchall()
        assert len(events) >= 1

    def test_evaluate_all_bidders_empty_tender(self, db_conn):
        """Test evaluating a tender with no bidders."""
        tender_id = str(uuid.uuid4())
        db_conn.execute(
            """INSERT INTO tenders (id, title, department, category, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (tender_id, "Empty Tender", "Finance", "Goods",
             "EVALUATING", datetime.now(timezone.utc).isoformat(),
             datetime.now(timezone.utc).isoformat()),
        )
        db_conn.commit()

        results = evaluate_all_bidders(db_conn, tender_id)
        assert results == []
