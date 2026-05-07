"""Demo data seeding for VerdictAI.

Provides idempotent seed functions to populate:
- CVC debarment list (5+ entries)
- CPM bootstrap corpus (10+ synthetic precedents)
- Sample tender with NIT criteria (5 criteria covering all types)

All seed functions check for existing data before inserting,
making them safe to call multiple times.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta


def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def _past_date(days_ago: int) -> str:
    """Return an ISO date string for N days in the past."""
    return (datetime.utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def seed_debarment_list(conn: sqlite3.Connection) -> None:
    """Seed the CVC debarment list with 5+ demo entries.

    Idempotent: checks if entries already exist by entity_name before inserting.
    """
    entries = [
        {
            "entity_name": "Blacklisted Construction Co",
            "pan_number": "AABCB1234A",
            "debarment_reason": "Fraudulent documentation",
            "debarment_date": "2023-03-15",
            "source": "CVC",
        },
        {
            "entity_name": "Corrupt Infra Pvt Ltd",
            "pan_number": "AACCI5678B",
            "debarment_reason": "Bid rigging",
            "debarment_date": "2023-06-20",
            "source": "CVC",
        },
        {
            "entity_name": "Fake Enterprises",
            "pan_number": "AABCF9012C",
            "debarment_reason": "Forged certificates",
            "debarment_date": "2022-11-10",
            "source": "GeM",
        },
        {
            "entity_name": "Debarred Tech Solutions",
            "pan_number": "AADCD3456D",
            "debarment_reason": "Non-performance",
            "debarment_date": "2024-01-05",
            "source": "CVC",
        },
        {
            "entity_name": "Suspended Works Ltd",
            "pan_number": "AASCS7890E",
            "debarment_reason": "Quality violations",
            "debarment_date": "2023-09-28",
            "source": "GeM",
        },
        {
            "entity_name": "Fraudulent Suppliers Inc",
            "pan_number": "AAFFS2345F",
            "debarment_reason": "Supply of counterfeit goods",
            "debarment_date": "2024-02-14",
            "source": "CVC",
        },
    ]

    cursor = conn.cursor()
    for entry in entries:
        # Check if entry already exists by entity_name
        cursor.execute(
            "SELECT COUNT(*) FROM debarment_list WHERE entity_name = ?",
            (entry["entity_name"],),
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                """INSERT INTO debarment_list (id, entity_name, pan_number, debarment_reason, debarment_date, source)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    _generate_uuid(),
                    entry["entity_name"],
                    entry["pan_number"],
                    entry["debarment_reason"],
                    entry["debarment_date"],
                    entry["source"],
                ),
            )


def seed_cpm_corpus(conn: sqlite3.Connection) -> None:
    """Seed the CPM bootstrap corpus with 10+ synthetic precedents.

    Covers all criterion types across multiple departments and categories.
    Idempotent: checks if entries already exist by criterion_text before inserting.
    """
    # We need a reference tender and criterion for FK constraints.
    # Create a bootstrap tender if it doesn't exist.
    cursor = conn.cursor()

    bootstrap_tender_id = "00000000-0000-0000-0000-000000000001"
    cursor.execute("SELECT COUNT(*) FROM tenders WHERE id = ?", (bootstrap_tender_id,))
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            """INSERT INTO tenders (id, title, department, category, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                bootstrap_tender_id,
                "CPM Bootstrap Reference Tender",
                "PWD",
                "construction",
                "EVALUATION_COMPLETE",
                "2023-01-01T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ),
        )

    bootstrap_criterion_id = "00000000-0000-0000-0000-000000000002"
    cursor.execute("SELECT COUNT(*) FROM criteria WHERE id = ?", (bootstrap_criterion_id,))
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            """INSERT INTO criteria (id, tender_id, criterion_text, criterion_type, status)
            VALUES (?, ?, ?, ?, ?)""",
            (
                bootstrap_criterion_id,
                bootstrap_tender_id,
                "Bootstrap reference criterion",
                "numeric_threshold",
                "approved",
            ),
        )

    precedents = [
        # numeric_threshold precedents
        {
            "criterion_text": "Annual turnover >= Rs. 5 crore in last 3 financial years",
            "resolved_interpretation": "Average annual turnover across 3 most recent audited financial years must meet or exceed Rs. 5 crore. Provisional figures not accepted.",
            "department": "PWD",
            "tender_category": "construction",
            "verdict": "PASS",
            "officer_action": "confirmed",
            "officer_id": "officer_001",
        },
        {
            "criterion_text": "Annual turnover >= Rs. 10 crore in last 3 financial years",
            "resolved_interpretation": "Turnover computed from audited balance sheets. Each individual year must meet threshold, not just average.",
            "department": "CRPF",
            "tender_category": "security_equipment",
            "verdict": "PASS",
            "officer_action": "confirmed",
            "officer_id": "officer_002",
        },
        {
            "criterion_text": "Net worth >= Rs. 2 crore as on last date of preceding financial year",
            "resolved_interpretation": "Net worth calculated as total assets minus total liabilities from audited balance sheet of most recent completed FY.",
            "department": "MoD",
            "tender_category": "equipment",
            "verdict": "FAIL",
            "officer_action": "confirmed",
            "officer_id": "officer_003",
        },
        # categorical_presence precedents
        {
            "criterion_text": "Valid GST registration certificate",
            "resolved_interpretation": "GST registration must be active (not cancelled/suspended) as of tender submission date. GSTIN format validated.",
            "department": "PWD",
            "tender_category": "construction",
            "verdict": "PASS",
            "officer_action": "confirmed",
            "officer_id": "officer_001",
        },
        {
            "criterion_text": "ISO 9001:2015 certification from accredited body",
            "resolved_interpretation": "ISO certificate must be current (not expired), issued by NABCB-accredited body. Scope must cover relevant work category.",
            "department": "MoD",
            "tender_category": "equipment",
            "verdict": "PASS",
            "officer_action": "overridden",
            "officer_id": "officer_004",
        },
        # temporal_recency precedents
        {
            "criterion_text": "3 similar works in last 5 years",
            "resolved_interpretation": "Minimum 3 completed works of similar nature and value (at least 50% of estimated cost) within 5 years preceding tender submission date. Completion certificates required.",
            "department": "PWD",
            "tender_category": "construction",
            "verdict": "PASS",
            "officer_action": "confirmed",
            "officer_id": "officer_001",
        },
        {
            "criterion_text": "Minimum 2 supply orders of similar items in last 3 years",
            "resolved_interpretation": "At least 2 purchase orders for similar category items completed satisfactorily within 3 years. Satisfactory completion certificates from ordering authority required.",
            "department": "CRPF",
            "tender_category": "security_equipment",
            "verdict": "PASS",
            "officer_action": "confirmed",
            "officer_id": "officer_002",
        },
        {
            "criterion_text": "5 years of operational experience in relevant domain",
            "resolved_interpretation": "Company incorporation date must be at least 5 years prior to tender submission. Domain relevance assessed from work order descriptions.",
            "department": "MoD",
            "tender_category": "services",
            "verdict": "FAIL",
            "officer_action": "confirmed",
            "officer_id": "officer_003",
        },
        # qualitative_assessment precedents
        {
            "criterion_text": "Adequate technical capacity for the required quantity",
            "resolved_interpretation": "Technical capacity assessed from plant/machinery list, manpower details, and production capacity certificates. Must demonstrate ability to deliver within specified timeline.",
            "department": "CRPF",
            "tender_category": "security_equipment",
            "verdict": "PASS",
            "officer_action": "confirmed",
            "officer_id": "officer_002",
        },
        {
            "criterion_text": "Satisfactory track record with no adverse performance reports",
            "resolved_interpretation": "No blacklisting, debarment, or termination of contract for default in last 5 years. Self-declaration accepted subject to verification.",
            "department": "MoD",
            "tender_category": "services",
            "verdict": "PASS",
            "officer_action": "overridden",
            "officer_id": "officer_004",
        },
        # composite precedent
        {
            "criterion_text": "Turnover + experience + ISO certification",
            "resolved_interpretation": "All three sub-criteria must be independently satisfied: (1) turnover >= threshold, (2) minimum experience years, (3) valid ISO certification. Partial compliance not accepted.",
            "department": "PWD",
            "tender_category": "construction",
            "verdict": "FAIL",
            "officer_action": "confirmed",
            "officer_id": "officer_001",
        },
        # Additional precedent for diversity
        {
            "criterion_text": "EMD of Rs. 5 lakh in form of bank guarantee",
            "resolved_interpretation": "Earnest Money Deposit must be in the form of bank guarantee from scheduled bank, valid for 90 days beyond bid validity. FDR also acceptable per GFR 2017.",
            "department": "PWD",
            "tender_category": "construction",
            "verdict": "PASS",
            "officer_action": "confirmed",
            "officer_id": "officer_001",
        },
    ]

    for precedent in precedents:
        # Check if precedent already exists by criterion_text
        cursor.execute(
            "SELECT COUNT(*) FROM cpm_entries WHERE criterion_text = ?",
            (precedent["criterion_text"],),
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                """INSERT INTO cpm_entries
                (id, criterion_text, resolved_interpretation, department, tender_category,
                 verdict, officer_action, officer_id, tender_id, criterion_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _generate_uuid(),
                    precedent["criterion_text"],
                    precedent["resolved_interpretation"],
                    precedent["department"],
                    precedent["tender_category"],
                    precedent["verdict"],
                    precedent["officer_action"],
                    precedent["officer_id"],
                    bootstrap_tender_id,
                    bootstrap_criterion_id,
                    _past_date(90),
                ),
            )


def seed_sample_tender(conn: sqlite3.Connection) -> None:
    """Seed a sample tender with NIT criteria covering all 5 criterion types.

    Creates:
    - A tender: "Supply of Security Equipment for CRPF"
    - 5 criteria covering all types

    Idempotent: checks if the sample tender already exists by title.
    """
    cursor = conn.cursor()

    tender_title = "Supply of Security Equipment for CRPF"

    # Check if sample tender already exists
    cursor.execute("SELECT COUNT(*) FROM tenders WHERE title = ?", (tender_title,))
    if cursor.fetchone()[0] > 0:
        return

    tender_id = _generate_uuid()
    now = _now_iso()

    # Insert sample tender
    cursor.execute(
        """INSERT INTO tenders (id, title, department, category, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            tender_id,
            tender_title,
            "CRPF",
            "security_equipment",
            "SCHEMA_PENDING_REVIEW",
            now,
            now,
        ),
    )

    # Define 5 criteria covering all types
    criteria = [
        {
            "criterion_text": "Annual turnover >= Rs. 10 crore in last 3 financial years",
            "criterion_type": "numeric_threshold",
            "threshold_value": json.dumps({
                "value": 100000000,
                "unit": "INR",
                "period": "3 financial years",
                "comparison": ">=",
            }),
            "gfr_override_permitted": 0,
            "gfr_rule_number": "GFR Rule 173(i)",
            "is_mandatory": 1,
            "acceptable_evidence_types": json.dumps(["audited_balance_sheet", "ca_certificate"]),
            "measurement_period": "last 3 financial years",
        },
        {
            "criterion_text": "Valid GST registration certificate",
            "criterion_type": "categorical_presence",
            "threshold_value": json.dumps({
                "document_type": "GST registration",
                "validity": "active",
            }),
            "gfr_override_permitted": 0,
            "gfr_rule_number": "GFR Rule 144",
            "is_mandatory": 1,
            "acceptable_evidence_types": json.dumps(["gst_certificate"]),
            "measurement_period": None,
        },
        {
            "criterion_text": "Minimum 3 similar supply orders completed in last 5 years",
            "criterion_type": "temporal_recency",
            "threshold_value": json.dumps({
                "count": 3,
                "similarity": "similar supply orders",
                "period": "5 years",
            }),
            "gfr_override_permitted": 1,
            "gfr_rule_number": None,
            "is_mandatory": 1,
            "acceptable_evidence_types": json.dumps([
                "completion_certificate",
                "purchase_order",
                "satisfactory_performance_certificate",
            ]),
            "measurement_period": "last 5 years",
        },
        {
            "criterion_text": "Turnover + experience + ISO certification",
            "criterion_type": "composite",
            "threshold_value": json.dumps({
                "sub_criteria": [
                    {"type": "numeric_threshold", "description": "Turnover >= Rs. 10 crore"},
                    {"type": "temporal_recency", "description": "5 years experience"},
                    {"type": "categorical_presence", "description": "ISO 9001:2015 certification"},
                ],
            }),
            "gfr_override_permitted": 1,
            "gfr_rule_number": None,
            "is_mandatory": 0,
            "acceptable_evidence_types": json.dumps([
                "audited_balance_sheet",
                "experience_certificate",
                "iso_certificate",
            ]),
            "measurement_period": "last 5 years",
        },
        {
            "criterion_text": "Adequate manufacturing capacity for the required quantity",
            "criterion_type": "qualitative_assessment",
            "threshold_value": json.dumps({
                "assessment_basis": "manufacturing capacity",
                "evidence_required": "plant and machinery list, production records",
            }),
            "gfr_override_permitted": 1,
            "gfr_rule_number": None,
            "is_mandatory": 0,
            "acceptable_evidence_types": json.dumps([
                "plant_machinery_list",
                "production_capacity_certificate",
                "factory_inspection_report",
            ]),
            "measurement_period": None,
        },
    ]

    for criterion in criteria:
        cursor.execute(
            """INSERT INTO criteria
            (id, tender_id, criterion_text, criterion_type, threshold_value,
             gfr_override_permitted, gfr_rule_number, is_mandatory,
             acceptable_evidence_types, measurement_period, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _generate_uuid(),
                tender_id,
                criterion["criterion_text"],
                criterion["criterion_type"],
                criterion["threshold_value"],
                criterion["gfr_override_permitted"],
                criterion["gfr_rule_number"],
                criterion["is_mandatory"],
                criterion["acceptable_evidence_types"],
                criterion["measurement_period"],
                "extracted",
            ),
        )


def seed_demo_data(conn: sqlite3.Connection) -> None:
    """Populate all demo data. Idempotent — safe to call multiple times.

    Seeds:
    1. CVC debarment list (6 entries)
    2. CPM bootstrap corpus (12 synthetic precedents)
    3. Sample tender with 5 NIT criteria covering all types

    Args:
        conn: An active SQLite connection (caller manages commit/rollback).
    """
    seed_debarment_list(conn)
    seed_cpm_corpus(conn)
    seed_sample_tender(conn)
