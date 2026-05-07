"""Verification script for the real-implementation services.

Run: python backend/verify_real_impl.py

Runs the four checks the task spec calls out:
  1. extract_criteria_from_text on the real NIT PDF → ~5 real criteria
  2. extract_numeric_value("Annual turnover of Rs. 12.45 Crore")
     → amount=124500000, unit="crore"
  3. embedding_service.similarity_score("construction of bridge",
     "building an overpass") → >0.5
  4. LLMStub.invoke returns the expected response shape with real data
     (is_simulated=False, real model_version).
"""

from __future__ import annotations

import sys


def check_1_nit_extraction() -> bool:
    from backend.utils.pdf_utils import parse_pdf
    from backend.services.criterion_extractor import extract_criteria_from_text

    parsed = parse_pdf("backend/demo_data/sample_nit.pdf")
    text = "\n".join(p["text"] for p in parsed["pages"])
    criteria = extract_criteria_from_text(text, source_document_id="doc-nit")
    print(f"1) Criteria extracted from real NIT PDF: {len(criteria)}")
    for c in criteria:
        print(f"   [{c['source_clause_ref'] or '—':<6}] {c['criterion_type']}"
              f"  mandatory={c['is_mandatory']}  gfr={c['gfr_rule_number']}")
    assert len(criteria) >= 4, (
        f"Expected at least 4 criteria from real NIT, got {len(criteria)}"
    )
    return True


def check_2_numeric_extraction() -> bool:
    from backend.services.evidence_extractor import extract_numeric_value

    result = extract_numeric_value("Annual turnover of Rs. 12.45 Crore")
    print(f"2) extract_numeric_value result: {result}")
    assert result is not None, "extract_numeric_value returned None"
    assert result["amount"] == 124_500_000, (
        f"Expected amount=124500000, got {result['amount']}"
    )
    assert result["unit"] == "crore", (
        f"Expected unit='crore', got {result['unit']!r}"
    )
    return True


def check_3_embedding_similarity() -> bool:
    from backend.services import embedding_service

    score = embedding_service.similarity_score(
        "construction of bridge", "building an overpass"
    )
    print(f"3) similarity('construction of bridge', 'building an overpass') = {score:.4f}")
    assert score > 0.5, f"Expected > 0.5, got {score:.4f}"
    return True


def check_4_llm_stub_shape() -> bool:
    from backend.services.llm_stub import LLMStub

    llm = LLMStub()

    request = {
        "prompt_type": "similarity_assessment",
        "context": {
            "text_a": "annual turnover of last 3 financial years",
            "text_b": "average turnover over past three years",
        },
        "tender_id": "t-verify",
    }
    resp = llm.invoke(request)
    print(f"4) LLMStub.invoke response keys: {sorted(resp.keys())}")
    print(f"   is_simulated:  {resp['is_simulated']}")
    print(f"   model_version: {resp['model_version']}")
    print(f"   confidence:    {resp['confidence']:.4f}")
    print(f"   result:        {resp['result']}")
    print(f"   prompt_hash:   {resp['prompt_hash'][:32]}…")

    for key in ("result", "confidence", "reasoning",
                "is_simulated", "model_version", "prompt_hash"):
        assert key in resp, f"LLMStub response missing key {key!r}"
    assert resp["is_simulated"] is False, "is_simulated must be False"
    assert "semantic" in resp["model_version"], (
        f"model_version should reference real model, got {resp['model_version']!r}"
    )
    assert len(resp["prompt_hash"]) == 64, "prompt_hash should be sha-256 hex"
    assert resp["confidence"] > 0.5, (
        f"Similar strings should score > 0.5, got {resp['confidence']:.4f}"
    )
    return True


def main() -> int:
    checks = [
        check_1_nit_extraction,
        check_2_numeric_extraction,
        check_3_embedding_similarity,
        check_4_llm_stub_shape,
    ]
    failed = 0
    for fn in checks:
        try:
            print()
            fn()
            print(f"   ✓ {fn.__name__} passed")
        except Exception as exc:
            failed += 1
            print(f"   ✗ {fn.__name__} failed: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"FAILED: {failed}/{len(checks)} verification checks")
        return 1
    print(f"OK: {len(checks)}/{len(checks)} verification checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
