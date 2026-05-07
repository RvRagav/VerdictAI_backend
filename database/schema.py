"""Database schema for VerdictAI.

Contains CREATE TABLE statements for all tables, FTS5 virtual table
for CPM text similarity search, and audit immutability triggers.
"""

import sqlite3


def create_tables(conn: sqlite3.Connection) -> None:
    """Execute all CREATE TABLE IF NOT EXISTS statements and triggers.

    Creates the complete VerdictAI schema including:
    - 11 data tables matching the design ERD
    - FTS5 virtual table for CPM similarity search
    - Audit immutability triggers (prevent UPDATE/DELETE on audit_events)

    Args:
        conn: An active SQLite connection.
    """
    cursor = conn.cursor()

    # ─── tenders ───────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tenders (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            department TEXT NOT NULL,
            category TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'DOCUMENTS_UPLOADED',
            ets_version TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # ─── documents ─────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            tender_id TEXT NOT NULL,
            bidder_id TEXT,
            doc_type TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            sha256_hash TEXT NOT NULL,
            page_count INTEGER,
            avg_ocr_confidence REAL,
            upload_timestamp TEXT NOT NULL,
            processing_status TEXT NOT NULL DEFAULT 'pending',
            FOREIGN KEY (tender_id) REFERENCES tenders(id),
            FOREIGN KEY (bidder_id) REFERENCES bidders(id)
        )
    """)

    # ─── pages ─────────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            image_path TEXT,
            ocr_confidence REAL,
            raw_text TEXT,
            dpi INTEGER,
            processing_notes TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    # ─── word_objects ──────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS word_objects (
            id TEXT PRIMARY KEY,
            page_id TEXT NOT NULL,
            text_content TEXT NOT NULL,
            x_min REAL NOT NULL,
            y_min REAL NOT NULL,
            x_max REAL NOT NULL,
            y_max REAL NOT NULL,
            confidence REAL NOT NULL,
            source_engine TEXT NOT NULL DEFAULT 'tesseract',
            FOREIGN KEY (page_id) REFERENCES pages(id)
        )
    """)

    # ─── criteria ──────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS criteria (
            id TEXT PRIMARY KEY,
            tender_id TEXT NOT NULL,
            criterion_text TEXT NOT NULL,
            criterion_type TEXT NOT NULL,
            threshold_value TEXT,
            gfr_override_permitted INTEGER NOT NULL DEFAULT 1,
            gfr_rule_number TEXT,
            source_document_id TEXT,
            source_clause_ref TEXT,
            amendment_history TEXT,
            is_mandatory INTEGER NOT NULL DEFAULT 0,
            acceptable_evidence_types TEXT,
            measurement_period TEXT,
            status TEXT NOT NULL DEFAULT 'extracted',
            approved_by TEXT,
            approved_at TEXT,
            FOREIGN KEY (tender_id) REFERENCES tenders(id),
            FOREIGN KEY (source_document_id) REFERENCES documents(id)
        )
    """)

    # ─── bidders ───────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bidders (
            id TEXT PRIMARY KEY,
            tender_id TEXT NOT NULL,
            company_name TEXT NOT NULL,
            pan_number TEXT,
            registration_number TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            debarment_status TEXT NOT NULL DEFAULT 'clear',
            debarment_check_timestamp TEXT,
            FOREIGN KEY (tender_id) REFERENCES tenders(id)
        )
    """)

    # ─── evaluations ──────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            tender_id TEXT NOT NULL,
            bidder_id TEXT NOT NULL,
            criterion_id TEXT NOT NULL,
            verdict TEXT,
            confidence REAL,
            evaluation_method TEXT,
            route TEXT,
            routing_reason TEXT,
            extracted_value TEXT,
            source_document_id TEXT,
            source_page_number INTEGER,
            source_bbox TEXT,
            ocr_confidence REAL,
            extraction_confidence REAL,
            entity_match_flag INTEGER DEFAULT 0,
            officer_decision TEXT,
            officer_id TEXT,
            officer_reason TEXT,
            officer_decision_timestamp TEXT,
            second_officer_id TEXT,
            second_officer_timestamp TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            FOREIGN KEY (tender_id) REFERENCES tenders(id),
            FOREIGN KEY (bidder_id) REFERENCES bidders(id),
            FOREIGN KEY (criterion_id) REFERENCES criteria(id),
            FOREIGN KEY (source_document_id) REFERENCES documents(id)
        )
    """)

    # ─── audit_events ─────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tender_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            actor TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            entry_hash TEXT NOT NULL,
            FOREIGN KEY (tender_id) REFERENCES tenders(id)
        )
    """)

    # ─── cpm_entries ──────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cpm_entries (
            id TEXT PRIMARY KEY,
            criterion_text TEXT NOT NULL,
            resolved_interpretation TEXT NOT NULL,
            department TEXT NOT NULL,
            tender_category TEXT NOT NULL,
            verdict TEXT NOT NULL,
            officer_action TEXT NOT NULL,
            officer_id TEXT NOT NULL,
            tender_id TEXT NOT NULL,
            criterion_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (tender_id) REFERENCES tenders(id),
            FOREIGN KEY (criterion_id) REFERENCES criteria(id)
        )
    """)

    # ─── debarment_list ───────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS debarment_list (
            id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            pan_number TEXT,
            debarment_reason TEXT NOT NULL,
            debarment_date TEXT NOT NULL,
            source TEXT NOT NULL
        )
    """)

    # ─── llm_stub_log ─────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_stub_log (
            id TEXT PRIMARY KEY,
            tender_id TEXT NOT NULL,
            prompt_type TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            prompt_content TEXT NOT NULL,
            response_content TEXT NOT NULL,
            model_version TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (tender_id) REFERENCES tenders(id)
        )
    """)

    # ─── FTS5 virtual table for CPM similarity search ─────────────────────
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS cpm_fts USING fts5(
            criterion_text,
            resolved_interpretation,
            department,
            tender_category,
            content='cpm_entries',
            content_rowid='rowid'
        )
    """)

    # ─── FTS5 sync trigger: INSERT on cpm_entries ─────────────────────────
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS cpm_fts_insert
        AFTER INSERT ON cpm_entries
        BEGIN
            INSERT INTO cpm_fts(rowid, criterion_text, resolved_interpretation, department, tender_category)
            VALUES (new.rowid, new.criterion_text, new.resolved_interpretation, new.department, new.tender_category);
        END
    """)

    # ─── Audit immutability triggers ──────────────────────────────────────
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS audit_no_update
        BEFORE UPDATE ON audit_events
        BEGIN
            SELECT RAISE(ABORT, 'UPDATE not permitted on audit_events: append-only ledger');
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS audit_no_delete
        BEFORE DELETE ON audit_events
        BEGIN
            SELECT RAISE(ABORT, 'DELETE not permitted on audit_events: append-only ledger');
        END
    """)

    conn.commit()
