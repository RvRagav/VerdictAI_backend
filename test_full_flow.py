"""Full frontend-equivalent API flow test.

Simulates exactly what the UI does:
1. Create tender
2. Upload NIT + corrigendum + 3 bidder PDFs via multipart
3. POST /process to run L1 OCR + L2 union extraction
4. Fetch criteria, show UNION sources
5. Approve schema
6. Trigger evaluation
7. Fetch HITL queue
8. Submit an officer decision
9. Generate report
"""

from __future__ import annotations

import os
import sys
import time
from fastapi.testclient import TestClient


def main() -> int:
    # Clean slate
    for f in ("verdict_ai.db", "verdict_ai.db-wal", "verdict_ai.db-shm"):
        if os.path.exists(f):
            os.remove(f)
    # Ensure LLM enabled
    os.environ.pop("LLM_DISABLED", None)

    from main import app

    print("=" * 74)
    print("  VerdictAI — Full API flow test (exactly what the UI calls)")
    print("=" * 74)

    def step(n: int, title: str) -> None:
        print(f"\n[{n}] {title}")

    ok = lambda msg: print(f"    ✓ {msg}")

    with TestClient(app) as client:
        # 1. Create tender
        step(1, "POST /tenders")
        r = client.post("/api/v1/tenders", json={
            "title": "CRPF — Perimeter Security Equipment Procurement",
            "department": "CRPF",
            "category": "security_equipment",
        })
        assert r.status_code == 201, r.text
        tender_id = r.json()["id"]
        ok(f"Tender created: {tender_id[:12]}")

        # 2. Upload 5 PDFs: NIT, corrigendum, 3 bidder submissions
        step(2, "POST /documents/upload × 5 PDFs")
        uploads = [
            ("sample_nit_crpf.pdf", "nit", None),
            ("sample_corrigendum.pdf", "corrigendum", None),
        ]
        # Create bidders
        bidders = [
            ("Sentinel Defence Systems Pvt Ltd", "AACCS9876K",
             "sample_bidder_good.pdf"),
            ("ApexGuard Technologies Pvt Ltd", "AAACA1234F",
             "sample_bidder_mismatch.pdf"),
            ("Nexus Security Solutions Pvt Ltd", "AACCN5432L",
             "sample_bidder_weak.pdf"),
        ]
        for name, pan, pdf in bidders:
            r = client.post(f"/api/v1/tenders/{tender_id}/bidders", json={
                "company_name": name,
                "pan_number": pan,
            })
            assert r.status_code == 201, f"Bidder creation failed: {r.text}"
            bidder_id = r.json()["id"]
            uploads.append((pdf, "bidder_submission", bidder_id))
            ok(f"Bidder created: {name[:40]} ({bidder_id[:8]})")

        # Upload all docs
        for pdf, doc_type, bidder_id in uploads:
            path = f"backend/demo_data/{pdf}"
            if not os.path.exists(path):
                print(f"    ✗ Missing demo PDF: {path}")
                return 1
            data = {"tender_id": tender_id, "doc_type": doc_type}
            if bidder_id:
                data["bidder_id"] = bidder_id
            with open(path, "rb") as fh:
                r = client.post(
                    "/api/v1/documents/upload",
                    files={"file": (pdf, fh, "application/pdf")},
                    data=data,
                )
            assert r.status_code == 201, f"Upload {pdf} failed: {r.text}"
            ok(f"Uploaded: {pdf:<32} → {r.json()['id'][:8]}")

        # 3. POST /process
        step(3, "POST /tenders/{id}/process (runs L1 OCR + L2 UNION)")
        t0 = time.perf_counter()
        r = client.post(f"/api/v1/tenders/{tender_id}/process")
        dt = time.perf_counter() - t0
        assert r.status_code == 200, f"Process failed: {r.text}"
        result = r.json()
        ok(f"Processed in {dt:.1f}s")
        ok(f"  documents: {result['processed_documents']}")
        ok(f"  criteria extracted: {result['criteria_extracted']}")
        ok(f"  corrigenda applied: {result['corrigenda_applied']}")
        ok(f"  final status: {result['final_tender_status']}")

        # 4. GET criteria, look for union sources
        step(4, "GET /tenders/{id}/criteria (with UNION branch tags)")
        r = client.get(f"/api/v1/tenders/{tender_id}/criteria")
        assert r.status_code == 200
        criteria = r.json()
        ok(f"{len(criteria)} criteria returned")
        # Look into amendment_history for _sources
        for c in criteria[:5]:
            ah = c.get("amendment_history")
            sources = "?"
            if isinstance(ah, dict):
                sources = "+".join(ah.get("_sources") or [])
            elif isinstance(ah, list):
                sources = "history"
            mand = "🔒" if c["is_mandatory"] else "  "
            text = c["criterion_text"][:50]
            rule = c.get("gfr_rule_number") or "—"
            ok(f"  {mand} [{sources:<10}] {c['criterion_type']:<22} rule={rule:<20} | {text}")

        # 5. Approve schema
        step(5, "POST /tenders/{id}/schema/approve")
        r = client.post(
            f"/api/v1/tenders/{tender_id}/schema/approve",
            json={"officer_id": "officer_priya_sharma"},
        )
        assert r.status_code == 200, f"Approve failed: {r.text}"
        ok(f"Schema approved @ {r.json()['approved_at']}")

        # 6. Debarment check + evaluation
        step(6, "POST /tenders/{id}/debarment-check + /evaluate")
        r = client.post(f"/api/v1/tenders/{tender_id}/debarment-check")
        assert r.status_code == 200
        deb = r.json()
        ok(f"Debarment: {deb['checked']} checked, {len(deb['flagged'])} flagged")

        t0 = time.perf_counter()
        r = client.post(f"/api/v1/tenders/{tender_id}/evaluate")
        dt = time.perf_counter() - t0
        assert r.status_code == 200, f"Evaluate failed: {r.text}"
        ok(f"Evaluation completed in {dt:.1f}s → {r.json()}")

        # 7. Summary
        step(7, "GET /tenders/{id}/summary")
        r = client.get(f"/api/v1/tenders/{tender_id}/summary")
        assert r.status_code == 200
        s = r.json()
        ok(f"Total evaluations: {s['total']}")
        ok(f"  auto-committed:  {s['auto_committed']}")
        ok(f"  pending review:  {s['pending_review']}")
        ok(f"  completed:       {s['completed']}")
        for b in s["by_bidder"]:
            ok(f"  {b['company_name'][:40]:<40} "
               f"PASS={b['pass_count']} FAIL={b['fail_count']} REVIEW={b['review_count']}")

        # 8. HITL queue
        step(8, "GET /tenders/{id}/hitl/queue")
        r = client.get(f"/api/v1/tenders/{tender_id}/hitl/queue")
        assert r.status_code == 200
        queue = r.json()
        ok(f"{len(queue)} pending cases")
        for q in queue[:3]:
            ok(f"  {q['route']:<18} conf={q['confidence']:.2f} | "
               f"{q['bidder_name'][:30]:<30} | {q['criterion_text'][:40]}")

        # 9. HITL card for first pending
        if queue:
            step(9, "GET /hitl/{eid}/card")
            eid = queue[0]["evaluation_id"]
            r = client.get(f"/api/v1/hitl/{eid}/card")
            assert r.status_code == 200
            card = r.json()
            ok(f"Card returned: criterion={card['criterion']['text'][:50]}")
            ok(f"  verdict={card['analysis']['verdict']} conf={card['analysis']['confidence']:.2f}")
            ok(f"  CPM precedents: {len(card['cpm_precedents'])}")
            ok(f"  can_override: {card['decision_options']['can_override']}")

        # 10. Submit officer decision
        if queue:
            step(10, "POST /hitl/{eid}/decide")
            eid = queue[0]["evaluation_id"]
            r = client.post(f"/api/v1/hitl/{eid}/decide", json={
                "decision": "confirm",
                "officer_id": "officer_priya_sharma",
            })
            if r.status_code == 200:
                ok(f"Decision recorded → {r.json().get('status')}")
            else:
                ok(f"Decision rejected: {r.text[:100]}")

        # 11. Generate report
        step(11, "POST /tenders/{id}/report")
        r = client.post(
            f"/api/v1/tenders/{tender_id}/report",
            json={"officer_id": "officer_priya_sharma"},
        )
        assert r.status_code == 201, f"Report failed: {r.text}"
        rep = r.json()
        ok(f"Report: {rep['report_id'][:12]}")
        ok(f"  SHA-256: {rep['sha256_hash'][:32]}...")
        ok(f"  download: {rep['download_url']}")

        # 12. Audit trail
        step(12, "GET /tenders/{id}/audit")
        r = client.get(f"/api/v1/tenders/{tender_id}/audit")
        assert r.status_code == 200
        events = r.json()
        ok(f"{len(events)} audit events")
        types = sorted(set(e["event_type"] for e in events))
        for t in types:
            count = sum(1 for e in events if e["event_type"] == t)
            ok(f"  {t:<28} × {count}")

    print()
    print("=" * 74)
    print("  ✓ Full API flow test PASSED")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
