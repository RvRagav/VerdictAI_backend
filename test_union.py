"""Test the Union Architecture: Rules + LLM + cross-validation."""

import os
from fastapi.testclient import TestClient

for f in ("verdict_ai.db", "verdict_ai.db-wal", "verdict_ai.db-shm"):
    if os.path.exists(f):
        os.remove(f)

from backend.main import app
from backend.database.connection import get_db
from backend.layers.l1_document import process_document
from backend.layers.l2_ets_builder import extract_criteria


def main():
    print("=" * 78)
    print("  UNION ARCHITECTURE TEST: Rules + LLM Cross-Validated Extraction")
    print("=" * 78)

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/tenders",
            json={
                "title": "Union Test",
                "department": "CRPF",
                "category": "security_equipment",
            },
        )
        tender_id = r.json()["id"]
        print(f"\n  → Created tender: {tender_id[:8]}\n")

        print("  → Running L1: PDF → Tesseract OCR...")
        conn = get_db()
        result = process_document(
            conn, tender_id, "backend/demo_data/sample_nit.pdf", "nit"
        )
        conn.commit()
        print(
            f"    Pages: {result['page_count']}  "
            f"OCR confidence: {result['avg_ocr_confidence']:.2f}\n"
        )

        print("  → Running L2: UNION criterion extraction (rules + LLM)...")
        print("  " + "─" * 70)
        criteria = extract_criteria(conn, tender_id, result["id"])
        print("  " + "─" * 70)

        print(f"\n  Final merged criteria: {len(criteria)}\n")

        branches = {"rules": 0, "llm": 0, "both": 0}
        for c in criteria:
            sources = c.get("_sources", ["rules"])
            if "rules" in sources and "llm" in sources:
                branches["both"] += 1
                badge = "🤝 RULES+LLM agree"
            elif "llm" in sources:
                branches["llm"] += 1
                badge = "🤖 LLM only  "
            else:
                branches["rules"] += 1
                badge = "📐 RULES only"

            clause = c["source_clause_ref"] or "—"
            mandatory = "🔒" if c["is_mandatory"] else "  "
            ctype = c["criterion_type"]
            text = c["criterion_text"][:60]
            print(f"  {badge}  {mandatory} [{ctype:<22}] clause {clause:<8} {text}...")

        print()
        print(f"  BRANCH AGREEMENT SUMMARY:")
        print(f"    Both branches found:  {branches['both']}  (high-confidence)")
        print(f"    Rules found only:     {branches['rules']}  (regex pattern matched)")
        print(f"    LLM found only:       {branches['llm']}  (semantic understanding)")

        # Show LLM invocation log
        print(f"\n  LLM invocations logged:")
        rows = conn.execute(
            "SELECT prompt_type, model_version, "
            "LENGTH(prompt_content) as plen, LENGTH(response_content) as rlen "
            "FROM llm_stub_log WHERE tender_id = ?",
            (tender_id,),
        ).fetchall()
        for row in rows:
            print(
                f"    prompt_type={row['prompt_type']:<22}  "
                f"model={row['model_version']:<50}  "
                f"prompt={row['plen']}B  response={row['rlen']}B"
            )

        conn.close()

    print("\n" + "=" * 78)
    print("  UNION TEST COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
