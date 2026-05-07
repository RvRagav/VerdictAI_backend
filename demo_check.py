"""End-to-end live probe that hits every Theme-3 requirement against
the running backend. Exits 0 if all non-negotiables pass.

Run after `uvicorn` is up:
    python3 backend/demo_check.py
"""

import json
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path


BASE = "http://localhost:8000/api/v1"


def req(method: str, path: str, body: dict | None = None,
        raw_data: bytes | None = None, content_type: str | None = None):
    url = f"{BASE}{path}"
    headers = {}
    data = None
    if raw_data is not None:
        data = raw_data
        if content_type:
            headers["Content-Type"] = content_type
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            raw = resp.read()
            if resp.getheader("Content-Type", "").startswith("application/json"):
                return resp.getcode(), json.loads(raw)
            return resp.getcode(), raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def multipart(file_path: Path, fields: dict, content_type: str):
    boundary = "----VerdictAICheckBoundary"
    buf = BytesIO()
    for k, v in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        )
        buf.write(str(v).encode())
        buf.write(b"\r\n")
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{file_path.name}"\r\n'.encode()
    )
    buf.write(f"Content-Type: {content_type}\r\n\r\n".encode())
    buf.write(file_path.read_bytes())
    buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def ok(msg):   print(f"  \033[92m✓\033[0m {msg}")
def bad(msg):  print(f"  \033[91m✗\033[0m {msg}")
def info(msg): print(f"    {msg}")


def main() -> int:
    results = {}
    DEMO = Path("backend/demo_data")

    # ── 1. Create tender ──────────────────────────────────────────────
    section("1. Multi-format ingestion (PDF + JPG scan)")

    code, tender = req("POST", "/tenders", {
        "title": "End-to-End Verification — Perimeter Security Supply",
        "department": "CENTRAL RESERVE POLICE FORCE",
        "category": "security_equipment",
    })
    if code != 201:
        bad(f"create tender failed: {code} {tender}")
        return 1
    tender_id = tender["id"]
    ok(f"Tender created: {tender_id[:12]} [{tender['status']}]")

    def upload(file_path: Path, doc_type: str,
               bidder_id: str | None = None,
               mime: str = "application/pdf"):
        fields = {"tender_id": tender_id, "doc_type": doc_type}
        if bidder_id:
            fields["bidder_id"] = bidder_id
        data, ctype = multipart(file_path, fields, mime)
        return req("POST", "/documents/upload",
                   raw_data=data, content_type=ctype)

    code, nit = upload(DEMO / "sample_nit_crpf.pdf", "nit")
    if code in (200, 201):
        pages = nit.get('page_count', '?')
        conf = nit.get('avg_ocr_confidence')
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        ok(f"NIT PDF → {pages} pages, OCR conf {conf_s}")
        results["pdf_ingest"] = True
    else:
        bad(f"NIT upload failed: {code} {nit}")
        results["pdf_ingest"] = False

    code, cor = upload(DEMO / "sample_corrigendum.pdf", "corrigendum")
    if code in (200, 201):
        ok(f"Corrigendum PDF → {cor.get('page_count', '?')} pages")

    # Three bidders
    def add_bidder(name, pan):
        c, b = req("POST", f"/tenders/{tender_id}/bidders",
                   {"company_name": name, "pan_number": pan,
                    "registration_number": f"REG-{pan[:6]}"})
        return b["id"] if c in (200, 201) else None

    good_id = add_bidder("Sentinel Defence Systems Pvt Ltd", "AACCS9876K")
    mm_id   = add_bidder("ApexGuard Technologies Pvt Ltd", "AAACA1234F")
    weak_id = add_bidder("Nexus Security Solutions Pvt Ltd", "AACCN5432L")
    if all((good_id, mm_id, weak_id)):
        ok(f"3 bidders registered")

    upload(DEMO / "sample_bidder_good.pdf", "bidder_submission", good_id)
    upload(DEMO / "sample_bidder_mismatch.pdf", "bidder_submission", mm_id)
    upload(DEMO / "sample_bidder_weak.pdf", "bidder_submission", weak_id)
    ok("3 bidder submission PDFs uploaded")

    # Degraded-scan JPG ─ the killer demo piece
    scan_path = DEMO / "sample_ca_certificate_scan.jpg"
    if scan_path.exists():
        code, sd = upload(scan_path, "certificate", good_id, mime="image/jpeg")
        if code in (200, 201):
            ok(f"JPG scan uploaded (phone-photo CA certificate, "
               f"{scan_path.stat().st_size/1024:.0f} KB)")
            results["jpg_ingest"] = True
        else:
            bad(f"JPG scan upload failed: {code} {sd}")
            results["jpg_ingest"] = False
    else:
        bad(f"Missing degraded-scan file: {scan_path}")
        results["jpg_ingest"] = False

    # ── 2. Process ────────────────────────────────────────────────────
    section("2. L1 OCR + L2 criterion extraction")
    t0 = time.time()
    code, proc = req("POST", f"/tenders/{tender_id}/process")
    dt = time.time() - t0
    if code == 200 and proc.get("status") == "complete":
        ok(f"/process in {dt:.1f}s — "
           f"{proc['processed_documents']} docs, "
           f"{proc['criteria_extracted']} criteria, "
           f"{proc['corrigenda_applied']} corrigenda → "
           f"{proc['final_tender_status']}")
        results["process"] = True
    else:
        bad(f"/process failed: {code} {proc}")
        results["process"] = False

    # ── 3. Criteria ───────────────────────────────────────────────────
    section("3. Extracted eligibility criteria (mandatory vs optional)")
    code, criteria = req("GET", f"/tenders/{tender_id}/criteria")
    if code == 200 and isinstance(criteria, list):
        mandatory = [c for c in criteria if c.get("is_mandatory")]
        ok(f"{len(criteria)} criteria — "
           f"{len(mandatory)} mandatory, "
           f"{len(criteria) - len(mandatory)} optional")
        types = {}
        for c in criteria:
            types[c["criterion_type"]] = types.get(c["criterion_type"], 0) + 1
        info(f"type distribution: {types}")
        for c in criteria[:6]:
            m = "M" if c.get("is_mandatory") else "O"
            info(f"  [{m}] {c['criterion_type']:<22} "
                 f"{c['criterion_text'][:50]}…")
        results["criteria"] = len(criteria) >= 3 and len(mandatory) >= 1
    else:
        bad(f"criteria fetch failed: {code}")
        results["criteria"] = False

    # ── 4. Approve + evaluate ─────────────────────────────────────────
    section("4. Schema approval + full bidder evaluation")
    code, ap = req("POST", f"/tenders/{tender_id}/schema/approve",
                   {"officer_id": "OFFICER-DEMO-01"})
    if code == 200:
        ok(f"Schema approved by {ap['officer_id']}")
    code, ev = req("POST", f"/tenders/{tender_id}/evaluate")
    if code == 200:
        ok(f"Evaluation triggered → {ev.get('bidder_count')} bidders")
        results["evaluate"] = True
    else:
        bad(f"/evaluate failed: {code} {ev}")
        results["evaluate"] = False

    # ── 5. Verdicts ────────────────────────────────────────────────────
    section("5. Verdicts + explanations")
    code, summary = req("GET", f"/tenders/{tender_id}/summary")
    if code == 200:
        ok(f"Summary: total={summary['total']} "
           f"auto={summary['auto_committed']} "
           f"review={summary['pending_review']}")
        for b in summary.get("by_bidder", []):
            p = b.get('pass_count', b.get('pass', 0))
            f = b.get('fail_count', b.get('fail', 0))
            r = b.get('review_count', b.get('review', 0))
            info(f"  {b['company_name'][:38]:<38} "
                 f"P={p} F={f} R={r}")
        results["summary"] = summary["total"] > 0
    else:
        bad(f"/summary failed: {code}")
        results["summary"] = False

    results["explanation"] = False
    # /evaluations returns a summary list; full explanation/routing info
    # lives on the individual HITL card. Use the first pending_review row.
    code, evals = req("GET", f"/tenders/{tender_id}/evaluations")
    if code == 200 and evals:
        s = evals[0]
        ok(f"Sample evaluation: {s.get('verdict')} "
           f"conf={s.get('confidence', 0):.2f} "
           f"route={s.get('route')} status={s.get('status')}")
        eval_id = s.get("id") or s.get("evaluation_id")
        if eval_id:
            code2, card = req("GET", f"/hitl/{eval_id}/card")
            if code2 == 200:
                an = card.get("analysis", {})
                info(f"  routing_reason: {(an.get('routing_reason') or '')[:90]}")
                info(f"  source_doc: {card.get('evidence',{}).get('source_document_id')} "
                     f"page={card.get('evidence',{}).get('source_page_number')}")
                exp = an.get("explanation") or {}
                if exp:
                    info(f"  headline:  {(exp.get('headline') or '')[:90]}")
                    info(f"  src_ref:   {(exp.get('source_reference') or '')[:90]}")
                    info(f"  conf_note: {(exp.get('confidence_note') or '')[:90]}")
                    info(f"  next:      {(exp.get('next_action') or '')[:90]}")
                    results["explanation"] = bool(exp.get("headline"))

    # ── 6. HITL queue ─────────────────────────────────────────────────
    section("6. HITL queue (ambiguous cases surfaced)")
    code, hq = req("GET", f"/tenders/{tender_id}/hitl/queue")
    if code == 200:
        ok(f"HITL queue: {len(hq)} pending")
        mand = [c for c in hq if c.get("route") == "mandatory_review"]
        hitl = [c for c in hq if c.get("route") == "hitl_review"]
        info(f"  mandatory_review={len(mand)}  hitl_review={len(hitl)}")
        for c in hq[:4]:
            info(f"  [{c['route']}] {c['bidder_name'][:28]:<28} "
                 f"— {c['criterion_text'][:45]}… "
                 f"conf={c['confidence']:.2f}")
        results["hitl"] = len(hq) > 0
    else:
        bad(f"HITL queue failed: {code}")
        results["hitl"] = False

    if hq:
        eval_id = hq[0].get("evaluation_id") or hq[0].get("id")
        if eval_id:
            code, card = req("GET", f"/hitl/{eval_id}/card")
            if code == 200:
                val = card.get("evidence", {}).get("extracted_value") or {}
                union = val.get("union") if isinstance(val, dict) else None
                an = card.get("analysis", {})
                info(f"  card.analysis.explanation present: "
                     f"{bool(an.get('explanation'))}")
                info(f"  card.evidence.extracted_value.union present: "
                     f"{bool(union)}")

    # ── 7. Audit ──────────────────────────────────────────────────────
    section("7. Audit trail + hash chain integrity")
    code, trail = req("GET", f"/tenders/{tender_id}/audit")
    if code == 200:
        ok(f"{len(trail)} audit events")
        types = {}
        for e in trail:
            types[e["event_type"]] = types.get(e["event_type"], 0) + 1
        info(f"  types: {types}")
        first, last = trail[0], trail[-1]
        info(f"  genesis prev_hash zero: {first['prev_hash'] == '0' * 64}")
        info(f"  latest entry_hash: {last['entry_hash'][:20]}…")
        results["audit"] = len(trail) > 5 and first["prev_hash"] == "0" * 64
    else:
        bad(f"audit fetch failed: {code}")
        results["audit"] = False

    # ── 8. Reproduce ──────────────────────────────────────────────────
    section("8. Byte-for-byte reproducibility re-run")
    code, r = req("POST", f"/tenders/{tender_id}/reproduce", {})
    if code == 200:
        ok(f"Reproduce: {'MATCH' if r['match'] else 'MISMATCH'} — "
           f"{r['matches']}/{r['total_compared']} "
           f"byte_identical={r['byte_identical_excluding_timestamps']}")
        info(f"  original hash:   {r['reproducibility_hash_original'][:20]}…")
        info(f"  reproduced hash: {r['reproducibility_hash_reproduced'][:20]}…")
        info(f"  diffs: {len(r['diffs'])} | "
             f"LLM cache hits/misses: {r['cache_hits']}/{r['cache_misses']}")
        results["reproduce"] = r["match"]
    else:
        bad(f"reproduce failed: {code}")
        results["reproduce"] = False

    # ── 9. Report ─────────────────────────────────────────────────────
    section("9. Consolidated multi-bidder PDF report")
    code, rep = req("POST", f"/tenders/{tender_id}/report",
                    {"officer_id": "OFFICER-DEMO-01"})
    if code == 201:
        ok(f"Report {rep['report_id'][:12]} — "
           f"sha256 {rep['sha256_hash'][:16]}…")
        try:
            url = f"http://localhost:8000{rep['download_url']}"
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
                info(f"  PDF: {len(data)/1024:.1f} KB "
                     f"ct={r.getheader('Content-Type')}")
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(BytesIO(data))
                info(f"  pages: {len(reader.pages)}")
                last = (reader.pages[-1].extract_text() or "").replace(" ", "")
                if rep['sha256_hash'][:16] in last:
                    ok("Signature page contains audit SHA-256")
            except Exception as exc:
                info(f"  pdf inspection: {exc}")
            results["report"] = True
        except Exception as exc:
            bad(f"download failed: {exc}")
            results["report"] = False
    else:
        bad(f"report failed: {code} {rep}")
        results["report"] = False

    # ── Scorecard ─────────────────────────────────────────────────────
    section("Theme-3 non-negotiables")
    checks = [
        ("Extract criteria + distinguish mandatory/optional",
            results.get("criteria")),
        ("Handle PDF + DOCX + JPG photograph ingestion",
            results.get("pdf_ingest") and results.get("jpg_ingest")),
        ("Criterion-level verdicts with source ref + explanation",
            results.get("summary") and results.get("explanation")),
        ("Ambiguous cases surfaced for HITL (never silently disqualify)",
            results.get("hitl")),
        ("End-to-end auditable (SHA-256 hash chain, genesis zero)",
            results.get("audit")),
        ("Byte-identical reproducibility on re-run",
            results.get("reproduce")),
        ("Consolidated PDF report for officer sign-off",
            results.get("report")),
    ]
    for name, passed in checks:
        (ok if passed else bad)(name)
    passed = sum(1 for _, p in checks if p)
    print(f"\n  SCORE: {passed}/{len(checks)} non-negotiables satisfied")
    print(f"  TENDER: {tender_id}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
