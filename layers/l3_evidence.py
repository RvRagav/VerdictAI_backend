"""Layer 3: Evidence Extraction for VerdictAI.

Provides type-specific evidence extraction from bidder documents for
each criterion type. Every extraction is driven by real OCR text and
PDF tables — no hash-seeded or random values. Delegates to:

  * :mod:`services.evidence_extractor` for numeric / categorical /
    temporal extraction,
  * :mod:`services.semantic_service.LLMStub` (re-exported via
    :mod:`services.llm_stub`) for qualitative assessment,
  * :mod:`services.entity_matcher` for company-name mismatch,
  * :mod:`services.cpm_service` for precedent lookup.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import json
import sqlite3
from typing import Optional

from services.entity_matcher import match_entity
from services.cpm_service import (
    search_cpm_precedents,
    search_cpm_precedents_semantic,
)
from services.llm_stub import LLMStub
from services import evidence_extractor
from services.union_service import (
    disambiguate_entity_union,
    evaluate_qualitative_union,
)
from layers.l5_audit import append_audit_event


# Number of CPM entries below which we prefer semantic re-ranking over FTS5.
_CPM_SEMANTIC_PREFERRED_BELOW = 1000


def extract_evidence(
    conn: sqlite3.Connection,
    tender_id: str,
    bidder_id: str,
    criterion_id: str,
) -> dict:
    """Extract evidence for a single (bidder, criterion) pair.

    Loads criterion + bidder + bidder documents, runs type-specific
    real extraction, pairs it with Entity Matcher output, computes the
    extraction_confidence, stores an audit event, and returns the
    evidence dict consumed by L4.

    Args:
        conn: Active SQLite connection.
        tender_id: The tender being evaluated.
        bidder_id: The bidder being evaluated.
        criterion_id: The criterion to extract evidence for.

    Returns:
        Dict with keys: value, source_document_id, page, bbox,
        ocr_confidence, extraction_confidence, entity_match_flag,
        entity_match_result, extracted_company_name, criterion_type,
        evaluation_method, tender_id, bidder_id, criterion_id.
    """
    conn.row_factory = sqlite3.Row

    criterion = conn.execute(
        "SELECT * FROM criteria WHERE id = ?", (criterion_id,)
    ).fetchone()
    if not criterion:
        return _empty_evidence(tender_id, bidder_id, criterion_id)

    criterion_type = criterion["criterion_type"]

    bidder = conn.execute(
        "SELECT * FROM bidders WHERE id = ?", (bidder_id,)
    ).fetchone()
    if not bidder:
        return _empty_evidence(tender_id, bidder_id, criterion_id)

    bidder_docs = conn.execute(
        "SELECT * FROM documents WHERE tender_id = ? AND bidder_id = ?",
        (tender_id, bidder_id),
    ).fetchall()

    # Concatenate every page's OCR'd text for this bidder; the
    # per-type extractors slice into this with regex.
    bidder_text_parts: list[str] = []
    page_confidences: list[float] = []
    for doc in bidder_docs:
        page_rows = conn.execute(
            "SELECT raw_text, ocr_confidence FROM pages "
            "WHERE document_id = ? ORDER BY page_number ASC",
            (doc["id"],),
        ).fetchall()
        for p in page_rows:
            if p["raw_text"]:
                bidder_text_parts.append(p["raw_text"])
            if p["ocr_confidence"] is not None:
                page_confidences.append(p["ocr_confidence"])

    full_text = "\n".join(bidder_text_parts)
    mean_page_confidence = (
        sum(page_confidences) / len(page_confidences)
        if page_confidences
        else 0.0
    )

    # Dispatch to type-specific extraction
    if criterion_type == "numeric_threshold":
        evidence = _extract_numeric_threshold(
            criterion, bidder_docs, bidder, full_text, mean_page_confidence
        )
    elif criterion_type == "categorical_presence":
        evidence = _extract_categorical_presence(
            criterion, bidder_docs, bidder, full_text, mean_page_confidence
        )
    elif criterion_type == "temporal_recency":
        evidence = _extract_temporal_recency(
            criterion, bidder_docs, bidder, full_text, mean_page_confidence
        )
    elif criterion_type == "qualitative_assessment":
        evidence = _extract_qualitative_assessment(
            conn, criterion, bidder_docs, bidder, full_text,
            mean_page_confidence, tender_id,
        )
    else:
        evidence = _empty_evidence(tender_id, bidder_id, criterion_id)

    # Entity match between extracted company name (from documents) and
    # registered bidder name. Falls back to the registered name if the
    # extractor couldn't find one — that preserves the default behaviour
    # of "no mismatch" rather than triggering a false alarm.
    extracted_company_name = (
        evidence.get("extracted_company_name")
        or evidence_extractor.extract_company_name(full_text)
        or bidder["company_name"]
    )

    # UNION entity matching: fuzzy match first, LLM only if the names
    # differ enough that parent/subsidiary detection matters. This keeps
    # LLM latency/cost off the critical path for obvious exact matches.
    fuzzy_result = match_entity(
        registered_name=bidder["company_name"],
        extracted_name=extracted_company_name,
    )
    # Trigger the LLM only when there's meaningful ambiguity
    if (
        fuzzy_result["similarity_score"] < 1.0
        and extracted_company_name
        and extracted_company_name.lower() != bidder["company_name"].lower()
    ):
        union_entity = disambiguate_entity_union(
            registered_name=bidder["company_name"],
            extracted_name=extracted_company_name,
            conn=conn,
            tender_id=tender_id,
        )
        entity_result = union_entity.value
        # Attach branch details for the HITL card to render
        entity_result["_rules_branch"] = union_entity.rules_value
        entity_result["_llm_branch"] = union_entity.llm_value
        entity_result["_agreement"] = union_entity.agreement
    else:
        entity_result = fuzzy_result

    evidence["extracted_company_name"] = extracted_company_name
    evidence["entity_match_flag"] = entity_result.get(
        "requires_review", False
    ) or entity_result.get("fraud_risk", False)
    evidence["entity_match_result"] = entity_result

    # Extraction confidence combines raw extractor output + OCR quality
    # + entity-match penalty.
    evidence["extraction_confidence"] = _compute_extraction_confidence(
        base_confidence=evidence.get("ocr_confidence", 0.0),
        criterion_type=criterion_type,
        entity_match_flag=evidence["entity_match_flag"],
    )

    evidence["tender_id"] = tender_id
    evidence["bidder_id"] = bidder_id
    evidence["criterion_id"] = criterion_id
    evidence["criterion_type"] = criterion_type

    append_audit_event(
        conn=conn,
        tender_id=tender_id,
        event_type="evidence_extracted",
        event_data={
            "bidder_id": bidder_id,
            "criterion_id": criterion_id,
            "criterion_type": criterion_type,
            "source_document_id": evidence.get("source_document_id"),
            "extraction_confidence": evidence.get("extraction_confidence"),
            "entity_match_flag": evidence["entity_match_flag"],
        },
        actor="system",
    )

    return evidence


# ─── Type-specific extraction ────────────────────────────────────────────────


def _extract_numeric_threshold(
    criterion: sqlite3.Row,
    bidder_docs: list,
    bidder: sqlite3.Row,
    full_text: str,
    mean_page_confidence: float,
) -> dict:
    """Extract a numeric Rs/Cr/Lakh figure from bidder documents.

    Strategy:
      1. Scan PDF tables for rows mentioning "turnover" / "net worth"
         etc. and pull a numeric value from the row.
      2. Fall back to regex scanning of the concatenated page text.
    """
    threshold_data = _parse_json(criterion["threshold_value"])
    # Search terms for table-aware lookup
    label_hint = (
        (threshold_data.get("label") if threshold_data else None) or ""
    ).lower()
    search_terms = _numeric_search_terms(label_hint, criterion["criterion_text"])

    source_doc = _find_document_by_type(bidder_docs, "bidder_submission") \
        or _find_document_by_type(bidder_docs, "certificate") \
        or (bidder_docs[0] if bidder_docs else None)

    numeric: Optional[dict] = None
    source_page: Optional[int] = None

    # 1) Table extraction over every uploaded PDF for this bidder
    for doc in bidder_docs:
        rows = evidence_extractor.extract_from_tables(
            doc["file_path"], search_terms
        )
        # Prefer rows whose extracted numeric has the highest confidence
        best = None
        for row in rows:
            if row.get("numeric_value") and (
                best is None
                or row["numeric_value"]["confidence"]
                > best["numeric_value"]["confidence"]
            ):
                best = row
        if best and best.get("numeric_value"):
            numeric = best["numeric_value"]
            source_page = best["page"]
            source_doc = doc
            break

    # 2) Prose fallback against the concatenated OCR text, scoped to
    # lines/sentences that mention the search terms for this criterion.
    # Without scoping, the first numeric in the document wins — which is
    # usually wrong when the same bidder submits turnover AND net worth
    # AND EMD figures.
    if numeric is None and full_text:
        numeric = _extract_scoped_numeric(full_text, search_terms)
        # Last resort: any numeric in the document
        if numeric is None:
            numeric = evidence_extractor.extract_numeric_value(full_text)

    if numeric is None:
        return {
            "value": None,
            "source_document_id": source_doc["id"] if source_doc else None,
            "page": source_page,
            "bbox": None,
            "ocr_confidence": mean_page_confidence,
            "extraction_confidence": 0.0,
            "extracted_company_name": None,
            "evaluation_method": "deterministic",
        }

    threshold_value = threshold_data or {}
    value = {
        "amount": numeric["amount"],
        "raw_value": numeric["raw_value"],
        "unit": numeric["unit"],
        "fiscal_year": numeric.get("fiscal_year"),
        "period": threshold_value.get("period")
        or threshold_value.get("period_unit")
        or threshold_value.get("measurement_period"),
    }

    # ocr_confidence here blends the extractor's own confidence with the
    # underlying page OCR confidence (floor on the lower of the two).
    ocr_conf = min(numeric["confidence"], max(mean_page_confidence, 0.5))

    return {
        "value": value,
        "source_document_id": source_doc["id"] if source_doc else None,
        "page": source_page,
        "bbox": None,
        "ocr_confidence": ocr_conf,
        "extraction_confidence": 0.0,
        "extracted_company_name": None,
        "evaluation_method": "deterministic",
    }


def _extract_categorical_presence(
    criterion: sqlite3.Row,
    bidder_docs: list,
    bidder: sqlite3.Row,
    full_text: str,
    mean_page_confidence: float,
) -> dict:
    """Presence + validity for GST/PAN/ISO/... certificates."""
    threshold_data = _parse_json(criterion["threshold_value"]) or {}
    doc_name = (threshold_data.get("document") or "").lower()
    cert_type = _infer_cert_type(doc_name, criterion["criterion_text"])

    registration_number = (
        evidence_extractor.extract_registration_number(full_text, cert_type)
        if cert_type
        else None
    )
    validity = evidence_extractor.extract_validity_date(full_text)

    found = bool(registration_number)

    source_doc = _find_document_by_type(bidder_docs, "certificate") \
        or _find_document_by_type(bidder_docs, "bidder_submission") \
        or (bidder_docs[0] if bidder_docs else None)

    if not found:
        value = {
            "found": False,
            "registration_number": None,
            "validity_date": None,
            "is_valid": False,
            "certificate_type": cert_type,
        }
        return {
            "value": value,
            "source_document_id": source_doc["id"] if source_doc else None,
            "page": None,
            "bbox": None,
            "ocr_confidence": mean_page_confidence,
            "extraction_confidence": 0.0,
            "extracted_company_name": None,
            "evaluation_method": "deterministic",
        }

    # We found a number. Validity is optional — unknown validity should
    # surface as REVIEW downstream, not FAIL (per design §2.4).
    is_valid = validity.get("is_valid") if validity else None
    validity_date = validity.get("date") if validity else None

    base_conf = 0.9 if validity else 0.6
    ocr_conf = min(base_conf, max(mean_page_confidence, 0.5))

    return {
        "value": {
            "found": True,
            "registration_number": registration_number,
            "validity_date": validity_date,
            "is_valid": is_valid,
            "certificate_type": cert_type,
        },
        "source_document_id": source_doc["id"] if source_doc else None,
        "page": None,
        "bbox": None,
        "ocr_confidence": ocr_conf,
        "extraction_confidence": 0.0,
        "extracted_company_name": None,
        "evaluation_method": "deterministic",
    }


def _extract_temporal_recency(
    criterion: sqlite3.Row,
    bidder_docs: list,
    bidder: sqlite3.Row,
    full_text: str,
    mean_page_confidence: float,
) -> dict:
    """Count past projects from tables/prose for temporal recency."""
    threshold_data = _parse_json(criterion["threshold_value"]) or {}
    required_count = threshold_data.get("count", 2)

    source_doc = _find_document_by_type(bidder_docs, "bidder_submission") \
        or (bidder_docs[0] if bidder_docs else None)

    projects: list[dict] = []
    for doc in bidder_docs:
        projects.extend(
            evidence_extractor.extract_project_list(doc["file_path"], "")
        )
    # Prose fallback if tables yielded nothing.
    if not projects and full_text:
        projects = evidence_extractor.extract_project_list("", full_text)

    # Confidence: high if at least required_count projects came back
    # with a date+value, lower otherwise.
    rich = [
        p for p in projects
        if p.get("value") and p.get("completion_date")
    ]
    if projects:
        ocr_conf = min(
            0.9 if len(rich) >= required_count else 0.7,
            max(mean_page_confidence, 0.5),
        )
    else:
        ocr_conf = mean_page_confidence

    return {
        "value": {
            "projects": projects,
            "count": len(projects),
            "required_count": required_count,
            "period": threshold_data.get("period")
            or threshold_data.get("measurement_period"),
        },
        "source_document_id": source_doc["id"] if source_doc else None,
        "page": None,
        "bbox": None,
        "ocr_confidence": ocr_conf,
        "extraction_confidence": 0.0,
        "extracted_company_name": None,
        "evaluation_method": "deterministic",
    }


def _extract_qualitative_assessment(
    conn: sqlite3.Connection,
    criterion: sqlite3.Row,
    bidder_docs: list,
    bidder: sqlite3.Row,
    full_text: str,
    mean_page_confidence: float,
    tender_id: str,
) -> dict:
    """Qualitative assessment via UNION: semantic similarity + real LLM reasoning.

    Runs both branches and cross-validates. When both agree on PASS,
    confidence is boosted; disagreements route to HITL with both
    interpretations visible side-by-side.
    """
    source_doc = _find_document_by_type(bidder_docs, "bidder_submission") \
        or (bidder_docs[0] if bidder_docs else None)

    tender = conn.execute(
        "SELECT * FROM tenders WHERE id = ?", (tender_id,)
    ).fetchone()
    department = tender["department"] if tender else "General"
    category = tender["category"] if tender else "Works"

    # CPM precedent retrieval — semantic for small corpora, FTS5 for large
    cpm_count_row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM cpm_entries "
        "WHERE department = ? AND tender_category = ?",
        (department, category),
    ).fetchone()
    total_cpm = cpm_count_row["cnt"] if cpm_count_row else 0
    if total_cpm < _CPM_SEMANTIC_PREFERRED_BELOW:
        cpm_precedents = search_cpm_precedents_semantic(
            conn=conn,
            criterion_text=criterion["criterion_text"],
            department=department,
            category=category,
            limit=3,
        )
    else:
        cpm_precedents = search_cpm_precedents(
            conn=conn,
            criterion_text=criterion["criterion_text"],
            department=department,
            category=category,
            limit=3,
        )

    # ── UNION EVALUATION: semantic + LLM cross-validated ──
    union = evaluate_qualitative_union(
        criterion_text=criterion["criterion_text"],
        bidder_name=bidder["company_name"],
        bidder_evidence=full_text,
        cpm_precedents=cpm_precedents,
    )

    # Log the LLM invocation for audit + reproducibility.
    #
    # IMPORTANT: ``result`` must be the *raw LLM branch output*
    # (what the model returned), NOT the merged consensus. The
    # CachedLLMClient looks this up by prompt_hash on re-run and
    # treats it as the LLM response. Storing the consensus instead
    # would cause the re-run to re-merge the consensus with the
    # rules branch a second time and diverge.
    llm = LLMStub()
    log_request = {
        "prompt_type": "qualitative_evaluation",
        "context": {
            "criterion_text": criterion["criterion_text"],
            "bidder_name": bidder["company_name"],
            "cpm_precedent_count": len(cpm_precedents),
        },
        "tender_id": tender_id,
    }
    log_response = {
        "result": union.llm_value or {},
        "confidence": union.llm_confidence,
        "reasoning": union.reasoning,
        "is_simulated": False,
        "model_version": union.llm_model_version or "semantic-only",
        "prompt_hash": union.llm_prompt_hash or "",
        "union": {
            "rules_confidence": union.rules_confidence,
            "llm_confidence": union.llm_confidence,
            "agreement": union.agreement,
            "agreement_score": union.agreement_score,
            "tokens": union.llm_tokens,
        },
    }
    llm.log_invocation(conn, tender_id, log_request, log_response)

    # Combine OCR floor with union confidence
    ocr_conf = min(
        max(union.confidence, 0.3),
        max(mean_page_confidence, 0.3),
    )

    return {
        "value": {
            "llm_verdict": union.value.get("verdict", "REVIEW"),
            "llm_confidence": union.llm_confidence,
            "llm_reasoning": union.value.get("reasoning", ""),
            "relevant_passages": union.value.get("factors", []),
            "similarity_to_criterion": union.value.get(
                "rules_branch", {}
            ).get("similarity_to_criterion"),
            "best_precedent_similarity": union.value.get(
                "rules_branch", {}
            ).get("best_precedent_similarity"),
            "cpm_precedents": cpm_precedents,
            "union": {
                "rules_verdict": union.value.get(
                    "rules_branch", {}
                ).get("verdict"),
                "llm_verdict": union.value.get(
                    "llm_branch", {}
                ).get("verdict"),
                "consensus_verdict": union.value.get("verdict"),
                "agreement": union.agreement,
                "agreement_score": union.agreement_score,
                "key_quote": union.value.get("key_quote", ""),
                "precedent_alignment": union.value.get("precedent_alignment"),
            },
        },
        "source_document_id": source_doc["id"] if source_doc else None,
        "page": None,
        "bbox": None,
        "ocr_confidence": ocr_conf,
        "extraction_confidence": 0.0,
        "extracted_company_name": None,
        "evaluation_method": "union_rules+llm",
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _find_document_by_type(bidder_docs: list, doc_type: str):
    for doc in bidder_docs:
        if doc["doc_type"] == doc_type:
            return doc
    return bidder_docs[0] if bidder_docs else None


def _compute_extraction_confidence(
    base_confidence: float,
    criterion_type: str,
    entity_match_flag: bool,
) -> float:
    """Combine OCR confidence with type- and flag-based penalties."""
    if base_confidence <= 0.0:
        return 0.0

    confidence = base_confidence
    if entity_match_flag:
        confidence *= 0.5

    # Type-specific dampening — qualitative assessments are inherently
    # less reliable than direct numeric extraction.
    adjustments = {
        "qualitative_assessment": 0.85,
        "numeric_threshold": 0.95,
        "categorical_presence": 0.92,
        "temporal_recency": 0.90,
    }
    confidence *= adjustments.get(criterion_type, 1.0)
    return max(0.0, min(1.0, confidence))


def _empty_evidence(tender_id: str, bidder_id: str, criterion_id: str) -> dict:
    return {
        "value": None,
        "source_document_id": None,
        "page": None,
        "bbox": None,
        "ocr_confidence": 0.0,
        "extraction_confidence": 0.0,
        "entity_match_flag": False,
        "entity_match_result": None,
        "extracted_company_name": "",
        "evaluation_method": "none",
        "tender_id": tender_id,
        "bidder_id": bidder_id,
        "criterion_id": criterion_id,
        "criterion_type": None,
    }


def _parse_json(raw) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _numeric_search_terms(label_hint: str, criterion_text: str) -> list[str]:
    """Derive row-search terms for the table extractor from the criterion."""
    ct = (criterion_text or "").lower()
    terms = set()
    if "turnover" in label_hint or "turnover" in ct:
        terms.update({"turnover", "total income", "revenue"})
    if "net worth" in label_hint or "net worth" in ct:
        terms.update({"net worth"})
    if "emd" in label_hint or "bid security" in label_hint or "earnest" in ct:
        terms.update({"emd", "bid security", "earnest money"})
    if "work" in label_hint or "contract" in label_hint or "project" in label_hint:
        terms.update({"order value", "contract value", "amount"})
    # Fallback — at least try the raw label
    if not terms and label_hint:
        terms.add(label_hint)
    if not terms:
        terms.add("turnover")
    return list(terms)


def _infer_cert_type(doc_name: str, criterion_text: str) -> Optional[str]:
    """Map a free-form document name to a registration-number type key."""
    s = f"{doc_name} {criterion_text}".lower()
    if "gst" in s:
        return "gst"
    if "pan" in s:
        return "pan"
    if "cin" in s:
        return "cin"
    if "tan" in s:
        return "tan"
    if "udyam" in s or "msme" in s:
        return "udyam"
    if "iso" in s:
        return "iso"
    return None


def _extract_scoped_numeric(text: str, search_terms: list[str]) -> Optional[dict]:
    """Find a numeric value in prose scoped to lines mentioning search terms.

    When the bidder's document has multiple figures (turnover, net worth,
    EMD, project values), picking the first match is almost always wrong.
    This helper splits the text into lines/sentences, keeps only those
    that mention any search term, and runs the numeric extractor on
    each. The highest-confidence numeric from a matching scope wins.
    """
    import re as _re

    if not text or not search_terms:
        return None
    terms = [t.lower() for t in search_terms if t]

    # Split on both line breaks and sentence terminators — either works
    # as a scope boundary.
    chunks = _re.split(r"(?<=[.!?])\s+|\n+", text)
    best: Optional[dict] = None
    for chunk in chunks:
        cl = chunk.lower()
        if not any(t in cl for t in terms):
            continue
        numeric = evidence_extractor.extract_numeric_value(chunk)
        if numeric is None:
            continue
        if best is None or numeric.get("confidence", 0) > best.get("confidence", 0):
            best = numeric
    return best
