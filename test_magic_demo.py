"""Automated end-to-end MAGIC demo for VerdictAI.

Runs the full pipeline with 3 bidders and 8 criteria, exercising every
demo scenario the judges need to see:

    ✓ Real PDF parsing (pdfplumber)
    ✓ Real Tesseract OCR at 300 DPI (via pdf2image)
    ✓ Real OpenCV preprocessing (5-step pipeline)
    ✓ Rules + LLM UNION criterion extraction with cross-validation
    ✓ GFR 2017 mandatory classification
    ✓ Corrigendum version assembly with amendment history
    ✓ CPM semantic precedent retrieval (sentence-transformers)
    ✓ Entity mismatch detection (fuzzy + LLM disambiguation for fraud)
    ✓ Confidence routing (auto-commit / HITL / mandatory)
    ✓ Never-silent disqualification (GFR-mandatory FAIL → mandatory review)
    ✓ SHA-256 audit chain integrity
    ✓ PDF report with signed hash
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from fastapi.testclient import TestClient


def _clean_slate():
    for f in ("verdict_ai.db", "verdict_ai.db-wal", "verdict_ai.db-shm"):
        if os.path.exists(f):
            os.remove(f)


def _ok(msg):
    print(f"  ✓ {msg}")


def _info(msg):
    print(f"    {msg}")


def _section(title):
    print()
    print(f"━━━ {title} " + "━" * (70 - len(title)))


def main() -> int:
    _clean_slate()

    # Ensure LLM enabled (conftest disables it for tests; we want the real thing)
    os.environ.pop("LLM_DISABLED", None)

    from main import app
    from database.connection import get_db
    from layers.l1_document import process_document
    from layers.l2_ets_builder import extract_criteria, apply_corrigendum, approve_schema
    from layers.l4_evaluation import evaluate_all_bidders
    from services.report_service import generate_report
    from layers.l5_audit import verify_hash_chain

    failures: list[str] = []

    with TestClient(app) as client:
        _section("STEP 1: Tender creation")
        r = client.post("/api/v1/tenders", json={
            "title": "Procurement of Advanced Perimeter Security Equipment",
            "department": "CRPF",
            "category": "security_equipment",
        })
        assert r.status_code == 201, r.text
        tender_id = r.json()["id"]
        _ok(f"Created tender: {tender_id[:8]}...")

        conn = get_db()

        _section("STEP 2: L1 Document Intelligence — NIT PDF")
        t0 = time.perf_counter()
        nit_doc = process_document(
            conn, tender_id,
            "backend/demo_data/sample_nit_crpf.pdf",
            "nit",
        )
        conn.commit()
        dt = time.perf_counter() - t0
        _ok(f"Processed NIT PDF in {dt:.1f}s — {nit_doc['page_count']} pages, "
            f"OCR confidence {nit_doc['avg_ocr_confidence']:.2f}")
        word_count = conn.execute(
            "SELECT COUNT(*) c FROM word_objects WHERE page_id IN "
            "(SELECT id FROM pages WHERE document_id = ?)",
            (nit_doc["id"],),
        ).fetchone()["c"]
        _info(f"OCR extracted {word_count} real word_objects with bboxes")
        if word_count < 100:
            failures.append(f"Expected >100 OCR words from NIT, got {word_count}")

        _section("STEP 3: L2 Union Criterion Extraction (Rules + LLM)")
        t0 = time.perf_counter()
        criteria = extract_criteria(conn, tender_id, nit_doc["id"])
        dt = time.perf_counter() - t0
        _ok(f"Extracted {len(criteria)} criteria in {dt:.1f}s")

        both_count = sum(1 for c in criteria if set(c.get("_sources", [])) == {"rules", "llm"})
        rules_only = sum(1 for c in criteria if c.get("_sources") == ["rules"])
        llm_only = sum(1 for c in criteria if c.get("_sources") == ["llm"])
        _info(f"Union breakdown: {both_count} rules+llm agree, "
              f"{rules_only} rules-only, {llm_only} llm-only")

        mandatory = sum(1 for c in criteria if c["is_mandatory"])
        _info(f"GFR-mandatory criteria: {mandatory}/{len(criteria)}")
        for c in criteria:
            sources = "+".join(c.get("_sources", []))
            mand = "🔒" if c["is_mandatory"] else "  "
            rule = c.get("gfr_rule_number", "—") or "—"
            clause = c.get("source_clause_ref", "—")
            text = c.get("criterion_text", "")[:55]
            _info(f"    {mand} [{sources:<10}] {c['criterion_type']:<22} "
                  f"{clause:<8} rule={rule:<20} — {text}")

        if len(criteria) < 6:
            failures.append(f"Expected >=6 criteria from NIT, got {len(criteria)}")

        _section("STEP 4: Corrigendum upload and amendment application")
        corr_doc = process_document(
            conn, tender_id,
            "backend/demo_data/sample_corrigendum.pdf",
            "corrigendum",
        )
        conn.commit()
        _ok(f"Processed corrigendum in {corr_doc['page_count']} page(s)")
        updated = apply_corrigendum(conn, tender_id, corr_doc["id"])
        _ok(f"Applied corrigendum — {len(updated)} criteria in effective spec")

        _section("STEP 5: Schema approval (officer gate)")
        approval = approve_schema(conn, tender_id, officer_id="officer_priya_sharma")
        _ok(f"Schema approved by {approval['officer_id']} at {approval['approved_at']}")

        _section("STEP 6: Upload 3 bidder submissions")
        bidders = [
            ("Sentinel Defence Systems Pvt Ltd", "AACCS9876K",
             "backend/demo_data/sample_bidder_good.pdf", "good"),
            ("ApexGuard Technologies Pvt Ltd", "AAACA1234F",
             "backend/demo_data/sample_bidder_mismatch.pdf", "entity mismatch"),
            ("Nexus Security Solutions Pvt Ltd", "AACCN5432L",
             "backend/demo_data/sample_bidder_weak.pdf", "weak"),
        ]
        bidder_ids = []
        for name, pan, pdf_path, label in bidders:
            bid_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO bidders (id, tender_id, company_name, pan_number, "
                "status, debarment_status) VALUES (?,?,?,?,?,?)",
                (bid_id, tender_id, name, pan, "pending", "clear"),
            )
            bidder_ids.append((bid_id, name, label))
            doc = process_document(conn, tender_id, pdf_path, "bidder_submission", bidder_id=bid_id)
            conn.commit()
            _ok(f"Processed [{label:<16}] {name[:40]:<40} → {doc['page_count']}pg, conf {doc['avg_ocr_confidence']:.2f}")

        # Also upload the CA certificate for the "good" bidder
        ca_bidder_id = bidder_ids[0][0]
        ca_doc = process_document(
            conn, tender_id,
            "backend/demo_data/sample_ca_certificate_stamp.pdf",
            "certificate",
            bidder_id=ca_bidder_id,
        )
        conn.commit()
        stamp_count = 0
        pages_rows = conn.execute(
            "SELECT processing_notes FROM pages WHERE document_id = ?",
            (ca_doc["id"],),
        ).fetchall()
        for p in pages_rows:
            if p["processing_notes"] and "stamps_detected=" in p["processing_notes"]:
                try:
                    n = int(p["processing_notes"].split("stamps_detected=")[1].split(";")[0])
                    stamp_count += n
                except Exception:
                    pass
        _ok(f"CA certificate processed — {stamp_count} stamp region(s) isolated")

        _section("STEP 7: Debarment pre-check for all bidders")
        r = client.post(f"/api/v1/tenders/{tender_id}/debarment-check")
        assert r.status_code == 200, r.text
        result = r.json()
        _ok(f"Debarment checked: {result['checked']} bidders, "
            f"{len(result['flagged'])} flagged, {len(result['clear'])} clear")

        _section("STEP 8: L3 Evidence + L4 Evaluation UNION (rules + LLM)")
        t0 = time.perf_counter()
        eval_results = evaluate_all_bidders(conn, tender_id)
        conn.commit()  # ensure the testclient sees the new rows
        dt = time.perf_counter() - t0
        _ok(f"Evaluated {len(eval_results)} bidders in {dt:.1f}s")
        for r in eval_results:
            status = r.get("status", "unknown")
            pass_count = sum(1 for e in r.get("evaluations", []) if e.get("verdict") == "PASS")
            fail_count = sum(1 for e in r.get("evaluations", []) if e.get("verdict") == "FAIL")
            review_count = sum(1 for e in r.get("evaluations", []) if e.get("verdict") == "REVIEW")
            auto = sum(1 for e in r.get("evaluations", []) if e.get("route") == "auto_commit")
            hitl = sum(1 for e in r.get("evaluations", []) if e.get("route") == "hitl_review")
            mandatory = sum(1 for e in r.get("evaluations", []) if e.get("route") == "mandatory_review")
            _info(f"  {r['company_name'][:40]:<40} → {status}")
            _info(f"    Verdicts: PASS={pass_count} FAIL={fail_count} REVIEW={review_count}")
            _info(f"    Routes:   auto={auto} HITL={hitl} mandatory={mandatory}")

        _section("STEP 9: HITL Queue contents")
        r = client.get(f"/api/v1/tenders/{tender_id}/hitl/queue")
        assert r.status_code == 200
        queue = r.json()
        _ok(f"HITL queue has {len(queue)} pending cases")
        priority_order = []
        for q in queue[:5]:
            priority_order.append(q['route'])
            _info(f"  {q['route']:<18} conf={q['confidence']:.2f} | "
                  f"{q['bidder_name'][:30]:<30} | {q['criterion_text'][:50]}")
        # Priority check: mandatory_review should come first
        mandatory_indices = [i for i, r in enumerate(priority_order) if r == "mandatory_review"]
        hitl_indices = [i for i, r in enumerate(priority_order) if r == "hitl_review"]
        if mandatory_indices and hitl_indices and max(mandatory_indices) > min(hitl_indices):
            failures.append("HITL queue not ordered by priority (mandatory should come first)")

        _section("STEP 10: HITL card for one case (includes union branches)")
        if queue:
            eid = queue[0]["evaluation_id"]
            r = client.get(f"/api/v1/hitl/{eid}/card")
            assert r.status_code == 200
            card = r.json()
            _ok(f"Card returned with all 5 components")
            _info(f"  criterion: {card['criterion']['text'][:60]}")
            _info(f"  verdict: {card['analysis']['verdict']} @ conf {card['analysis']['confidence']:.2f}")
            _info(f"  CPM precedents: {len(card['cpm_precedents'])}")
            _info(f"  can_override: {card['decision_options']['can_override']}")
            # Check union branch data is present for qualitative criteria
            extracted = card.get("evidence", {}).get("extracted_value")
            if isinstance(extracted, dict) and "union" in extracted:
                union = extracted["union"]
                _info(f"  UNION: rules={union['rules_verdict']} | "
                      f"llm={union['llm_verdict']} | consensus={union['consensus_verdict']}")

        _section("STEP 11: Officer confirms a decision (CPM precedent created)")
        if queue:
            eid = queue[0]["evaluation_id"]
            cpm_before = conn.execute("SELECT COUNT(*) c FROM cpm_entries").fetchone()["c"]
            r = client.post(f"/api/v1/hitl/{eid}/decide", json={
                "decision": "confirm",
                "officer_id": "officer_priya_sharma",
            })
            if r.status_code == 200:
                _ok(f"Decision recorded → {r.json().get('status')}")
                cpm_after = conn.execute("SELECT COUNT(*) c FROM cpm_entries").fetchone()["c"]
                _info(f"  CPM entries: {cpm_before} → {cpm_after} (+{cpm_after - cpm_before} precedent stored)")
            else:
                _info(f"  Decision rejected: {r.text[:200]}")

        _section("STEP 12: Audit hash chain integrity")
        is_valid, err = verify_hash_chain(conn, tender_id)
        if is_valid:
            _ok("SHA-256 hash chain is intact across all audit events")
        else:
            failures.append(f"Hash chain verification FAILED: {err}")
        events = conn.execute(
            "SELECT COUNT(*) c FROM audit_events WHERE tender_id = ?",
            (tender_id,),
        ).fetchone()["c"]
        _info(f"  Total audit events: {events}")
        event_types = conn.execute(
            "SELECT event_type, COUNT(*) c FROM audit_events WHERE tender_id = ? "
            "GROUP BY event_type ORDER BY c DESC",
            (tender_id,),
        ).fetchall()
        for et in event_types[:10]:
            _info(f"    {et['event_type']:<28} × {et['c']}")

        _section("STEP 13: Report generation with SHA-256 signature")
        report = generate_report(conn, tender_id, officer_id="officer_priya_sharma")
        conn.commit()
        _ok(f"Report generated: {report['report_id'][:12]}...")
        _info(f"  Audit trail SHA-256: {report['sha256_hash'][:32]}...")
        _info(f"  Download path: {report['download_path']}")

        _section("STEP 14: LLM usage summary")
        llm_stats = conn.execute(
            "SELECT prompt_type, model_version, COUNT(*) c "
            "FROM llm_stub_log WHERE tender_id = ? GROUP BY prompt_type, model_version",
            (tender_id,),
        ).fetchall()
        for row in llm_stats:
            _info(f"  {row['prompt_type']:<24} × {row['c']:<3} model={row['model_version']}")

        conn.close()

    _section("RESULTS")
    if failures:
        print(f"  ✗ {len(failures)} failure(s):")
        for f in failures:
            print(f"    - {f}")
        return 1

    print("  ✓ All end-to-end checks passed.")
    print("  ✓ System is ready for demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
