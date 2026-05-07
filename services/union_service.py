"""Union Architecture: Rules + LLM with cross-validation.

The winning insight behind VerdictAI's evaluation engine:

    Rules-only       → brittle, misses ambiguous language
    LLM-only         → hallucinations, non-deterministic, unauditable
    Rules ⊕ LLM     → deterministic floor + semantic ceiling, with
                       DISAGREEMENT SURFACED TO THE OFFICER

When rules and LLM agree, we auto-commit at higher confidence than
either alone could justify. When they disagree, both interpretations
show up in the HITL card side-by-side — the officer picks which is
right, and that decision becomes a CPM precedent that trains both.

This module is the unification layer. Downstream callers don't see
"rules" or "LLM" — they see a single result with both interpretations,
their agreement status, and a combined confidence that reflects the
real epistemic state of the extraction.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from services import criterion_extractor, embedding_service
from services.llm_client import LLMClient, LLMResponse, get_default_client
from services.prompts import (
    CRITERION_EXTRACTION_SCHEMA,
    CRITERION_EXTRACTION_SYSTEM,
    CORRIGENDUM_MAPPING_SCHEMA,
    CORRIGENDUM_MAPPING_SYSTEM,
    ENTITY_DISAMBIGUATION_SCHEMA,
    ENTITY_DISAMBIGUATION_SYSTEM,
    PROMPT_VERSION,
    QUALITATIVE_EVAL_SCHEMA,
    QUALITATIVE_EVAL_SYSTEM,
    build_corrigendum_mapping_user_prompt,
    build_criterion_extraction_user_prompt,
    build_entity_disambiguation_user_prompt,
    build_qualitative_eval_user_prompt,
)


logger = logging.getLogger(__name__)


# ─── Result dataclasses ──────────────────────────────────────────────────


@dataclass
class UnionResult:
    """Combined result from rules + LLM with agreement metadata."""

    # Consensus output (what downstream should use)
    value: Any
    confidence: float

    # Individual branch outputs (for audit + HITL display)
    rules_value: Any
    rules_confidence: float
    llm_value: Any
    llm_confidence: float

    # Agreement metrics
    agreement: str  # "agree", "disagree", "rules_only", "llm_only", "neither"
    agreement_score: float  # 0.0-1.0 semantic similarity of outputs

    # Provenance
    method: str  # "rules+llm" | "rules_only" | "llm_only"
    llm_model_version: str = ""
    llm_prompt_hash: str = ""
    llm_tokens: dict = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict:
        """Serialise for storage in evaluation records / audit events."""
        return asdict(self)


# ─── Criterion extraction (union) ────────────────────────────────────────


def extract_criteria_union(
    document_text: str,
    source_document_id: str = "",
    llm_client: Optional[LLMClient] = None,
) -> UnionResult:
    """Extract criteria using rules AND LLM, then cross-validate.

    Runs both branches in sequence (rules is cheap, LLM is the slow
    path). Merges the results by matching criteria between branches
    using semantic similarity on criterion_text. Produces a single
    consolidated list where every criterion carries provenance
    indicating which branch(es) found it.

    Args:
        document_text: Full OCR'd / extracted NIT text.
        source_document_id: For provenance on extracted criteria.
        llm_client: Optional injected client; defaults to singleton.

    Returns:
        :class:`UnionResult` whose ``value`` is the merged list of
        criterion dicts (each with ``_sources`` field indicating
        which branch(es) found it).
    """
    client = llm_client or get_default_client()

    # ── Branch 1: Rules (always runs) ──
    rules_criteria = criterion_extractor.extract_criteria_from_text(
        text=document_text,
        source_document_id=source_document_id,
    )
    rules_conf = _compute_rules_confidence(rules_criteria, document_text)

    # ── Branch 2: LLM (skipped if no API key) ──
    llm_criteria: list[dict] = []
    llm_conf = 0.0
    llm_response: Optional[LLMResponse] = None

    if client.is_configured and document_text.strip():
        user_prompt = build_criterion_extraction_user_prompt(document_text)
        llm_response = client.structured_extraction(
            system=CRITERION_EXTRACTION_SYSTEM,
            user=user_prompt,
            schema_hint=CRITERION_EXTRACTION_SCHEMA,
        )
        if llm_response.data and isinstance(llm_response.data, dict):
            raw = llm_response.data.get("criteria") or []
            llm_criteria = [
                _normalise_llm_criterion(c, source_document_id) for c in raw
                if isinstance(c, dict) and c.get("criterion_text")
            ]
            llm_conf = _compute_llm_confidence(llm_criteria, llm_response)

    # ── Merge with cross-validation ──
    merged, agreement, agreement_score = _merge_criteria(
        rules_criteria, llm_criteria
    )

    # Consensus confidence rewards agreement, penalises disagreement
    consensus_conf = _compute_consensus_confidence(
        rules_conf, llm_conf, agreement
    )

    reasoning_parts = [
        f"Rules extracted {len(rules_criteria)} criteria (conf {rules_conf:.2f})",
    ]
    if client.is_configured:
        if llm_response and llm_response.error:
            reasoning_parts.append(
                f"LLM unavailable: {llm_response.error}"
            )
        else:
            reasoning_parts.append(
                f"LLM extracted {len(llm_criteria)} criteria (conf {llm_conf:.2f})"
            )
            reasoning_parts.append(
                f"Cross-validation: {agreement} (score {agreement_score:.2f})"
            )
    else:
        reasoning_parts.append("LLM disabled (no API key) — rules only")

    return UnionResult(
        value=merged,
        confidence=consensus_conf,
        rules_value=rules_criteria,
        rules_confidence=rules_conf,
        llm_value=llm_criteria,
        llm_confidence=llm_conf,
        agreement=agreement,
        agreement_score=agreement_score,
        method="rules+llm" if llm_criteria else ("rules_only" if rules_criteria else "neither"),
        llm_model_version=llm_response.model if llm_response else "",
        llm_prompt_hash=llm_response.prompt_hash if llm_response else "",
        llm_tokens={
            "in": llm_response.tokens_in if llm_response else 0,
            "out": llm_response.tokens_out if llm_response else 0,
            "latency_ms": llm_response.latency_ms if llm_response else 0,
        },
        reasoning=" | ".join(reasoning_parts),
    )


# ─── Qualitative assessment (union) ──────────────────────────────────────


def evaluate_qualitative_union(
    criterion_text: str,
    bidder_name: str,
    bidder_evidence: str,
    cpm_precedents: list[dict],
    llm_client: Optional[LLMClient] = None,
) -> UnionResult:
    """Evaluate a qualitative criterion using semantic similarity + LLM.

    Both branches produce a verdict + confidence. The LLM carries
    chain-of-thought reasoning. Both are stored; the consensus is
    what drives routing.

    Args:
        criterion_text: The criterion being evaluated.
        bidder_name: Name of the bidding entity.
        bidder_evidence: Concatenated OCR text of bidder documents.
        cpm_precedents: List of CPM entry dicts for this (dept, category).
        llm_client: Optional injected client.

    Returns:
        :class:`UnionResult` with value={verdict, reasoning, factors, ...}.
    """
    client = llm_client or get_default_client()

    # ── Branch 1: Semantic similarity (rules-ish) ──
    rules_value, rules_conf = _semantic_qualitative(
        criterion_text, bidder_evidence, cpm_precedents
    )

    # ── Branch 2: LLM reasoning ──
    llm_value: dict = {}
    llm_conf = 0.0
    llm_response: Optional[LLMResponse] = None

    if client.is_configured and bidder_evidence.strip():
        user_prompt = build_qualitative_eval_user_prompt(
            criterion_text=criterion_text,
            bidder_name=bidder_name,
            bidder_evidence=bidder_evidence,
            cpm_precedents=cpm_precedents,
        )
        llm_response = client.structured_extraction(
            system=QUALITATIVE_EVAL_SYSTEM,
            user=user_prompt,
            schema_hint=QUALITATIVE_EVAL_SCHEMA,
        )
        if llm_response.data and isinstance(llm_response.data, dict):
            llm_value = llm_response.data
            llm_conf = float(llm_value.get("confidence", 0.0))

    # ── Cross-validate verdicts ──
    rules_verdict = rules_value.get("verdict", "REVIEW")
    llm_verdict = llm_value.get("verdict", "REVIEW")

    agreement, agreement_score = _verdict_agreement(
        rules_verdict, llm_verdict, rules_value, llm_value
    )

    # Consensus verdict logic:
    # - Both PASS at high conf → PASS
    # - Any FAIL → REVIEW (spec §2.4: never auto-commit FAIL from LLM)
    # - Disagreement → REVIEW
    # - Otherwise → majority / fallback to REVIEW
    if llm_verdict == "FAIL" or rules_verdict == "FAIL":
        consensus_verdict = "REVIEW"
    elif llm_verdict == "PASS" and rules_verdict == "PASS":
        consensus_verdict = "PASS"
    elif llm_verdict == rules_verdict:
        consensus_verdict = llm_verdict
    else:
        consensus_verdict = "REVIEW"

    consensus_conf = _compute_consensus_confidence(
        rules_conf, llm_conf, agreement
    )

    # Merge factors + reasoning from both branches
    consensus_value = {
        "verdict": consensus_verdict,
        "confidence": consensus_conf,
        "reasoning": llm_value.get("reasoning", "")
        or _default_reasoning(rules_value, llm_value, agreement),
        "factors": llm_value.get("factors", []) + rules_value.get("factors", []),
        "key_quote": llm_value.get("key_quote", ""),
        "precedent_alignment": llm_value.get("precedent_alignment", "unknown"),
        "rules_branch": rules_value,
        "llm_branch": llm_value,
        "agreement": agreement,
        "agreement_score": agreement_score,
    }

    reasoning = (
        f"Rules: {rules_verdict} ({rules_conf:.2f}) | "
        f"LLM: {llm_verdict} ({llm_conf:.2f}) | "
        f"Consensus: {consensus_verdict} ({consensus_conf:.2f}) | "
        f"{agreement}"
    )

    return UnionResult(
        value=consensus_value,
        confidence=consensus_conf,
        rules_value=rules_value,
        rules_confidence=rules_conf,
        llm_value=llm_value,
        llm_confidence=llm_conf,
        agreement=agreement,
        agreement_score=agreement_score,
        method="rules+llm" if llm_value else "rules_only",
        llm_model_version=llm_response.model if llm_response else "",
        llm_prompt_hash=llm_response.prompt_hash if llm_response else "",
        llm_tokens={
            "in": llm_response.tokens_in if llm_response else 0,
            "out": llm_response.tokens_out if llm_response else 0,
            "latency_ms": llm_response.latency_ms if llm_response else 0,
        },
        reasoning=reasoning,
    )


# ─── Entity disambiguation (union) ───────────────────────────────────────


def disambiguate_entity_union(
    registered_name: str,
    extracted_name: str,
    llm_client: Optional[LLMClient] = None,
    conn: Optional["sqlite3.Connection"] = None,
    tender_id: Optional[str] = None,
) -> UnionResult:
    """Cross-validate fuzzy entity match with LLM semantic judgment.

    Fuzzy matching (rapidfuzz-based) handles surface-form variations
    well but struggles with parent/subsidiary distinctions. The LLM
    reasons about the context — "ABC Infrastructure Pvt Ltd" vs
    "ABC Group Holdings Ltd" is a fraud-risk parent-subsidiary
    substitution that fuzzy match might score at 0.8 but the LLM
    correctly flags.

    When ``conn`` and ``tender_id`` are supplied, the LLM call is
    logged to ``llm_stub_log`` so the reproducibility re-runner
    (CachedLLMClient) can replay it by ``prompt_hash`` and produce
    byte-identical results. Without logging, the rerun would issue a
    fresh LLM call that might diverge.
    """
    from services.entity_matcher import match_entity
    from services.llm_stub import LLMStub

    client = llm_client or get_default_client()

    # ── Branch 1: Fuzzy matching ──
    rules_match = match_entity(registered_name, extracted_name)
    rules_conf = rules_match.get("similarity_score", 0.0)

    # ── Branch 2: LLM disambiguation ──
    llm_match: dict = {}
    llm_conf = 0.0
    llm_response: Optional[LLMResponse] = None

    if client.is_configured and registered_name and extracted_name:
        user_prompt = build_entity_disambiguation_user_prompt(
            registered_name, extracted_name
        )
        llm_response = client.structured_extraction(
            system=ENTITY_DISAMBIGUATION_SYSTEM,
            user=user_prompt,
            schema_hint=ENTITY_DISAMBIGUATION_SCHEMA,
        )
        if llm_response.data and isinstance(llm_response.data, dict):
            llm_match = llm_response.data
            llm_conf = float(llm_match.get("confidence", 0.0))

    # Consensus: LLM's parent_company detection trumps fuzzy high score
    rules_requires_review = rules_match.get("requires_review", False)
    llm_fraud_risk = llm_match.get("fraud_risk", False)
    llm_classification = llm_match.get("classification", "unknown")

    requires_review = (
        rules_requires_review
        or llm_fraud_risk
        or llm_classification in ("parent_company", "subsidiary", "unrelated")
    )

    agreement = "agree"
    if rules_match.get("is_match", False) and llm_classification in (
        "parent_company", "subsidiary", "unrelated"
    ):
        # Fuzzy said "match" but LLM says parent/subsidiary/unrelated — LLM wins
        agreement = "disagree"

    consensus_value = {
        "registered_name": registered_name,
        "extracted_name": extracted_name,
        "similarity_score": rules_conf,
        "is_match": rules_match.get("is_match", False) and not llm_fraud_risk,
        "mismatch_type": llm_classification
        if llm_classification not in ("same_entity", "abbreviation")
        else rules_match.get("mismatch_type"),
        "requires_review": requires_review,
        "fraud_risk": llm_fraud_risk,
        "llm_classification": llm_classification,
        "llm_reasoning": llm_match.get("reasoning", ""),
    }

    agreement_score = 1.0 if agreement == "agree" else 0.0
    consensus_conf = _compute_consensus_confidence(
        rules_conf, llm_conf, agreement
    )

    # Log the LLM call so reproducibility replay can find it by hash.
    if llm_response and conn is not None and tender_id:
        _log_entity_llm_call(
            conn, tender_id, llm_response,
            registered_name, extracted_name, llm_match,
        )

    return UnionResult(
        value=consensus_value,
        confidence=consensus_conf,
        rules_value=rules_match,
        rules_confidence=rules_conf,
        llm_value=llm_match,
        llm_confidence=llm_conf,
        agreement=agreement,
        agreement_score=agreement_score,
        method="rules+llm" if llm_match else "rules_only",
        llm_model_version=llm_response.model if llm_response else "",
        llm_prompt_hash=llm_response.prompt_hash if llm_response else "",
        llm_tokens={
            "in": llm_response.tokens_in if llm_response else 0,
            "out": llm_response.tokens_out if llm_response else 0,
            "latency_ms": llm_response.latency_ms if llm_response else 0,
        },
        reasoning=(
            f"Fuzzy similarity: {rules_conf:.2f} | "
            f"LLM classification: {llm_classification} | "
            f"Fraud risk: {llm_fraud_risk}"
        ),
    )


def _log_entity_llm_call(
    conn,
    tender_id: str,
    llm_response: LLMResponse,
    registered_name: str,
    extracted_name: str,
    llm_match: dict,
) -> None:
    """Log an entity-disambiguation LLM call to ``llm_stub_log``.

    Isolated in a helper so reproducibility replay can find the
    response by ``prompt_hash``. Swallowing exceptions here is fine —
    logging failure should never break evaluation.
    """
    try:
        from services.llm_stub import LLMStub
        stub = LLMStub()
        request = {
            "prompt_type": "entity_disambiguation",
            "context": {
                "registered_name": registered_name,
                "extracted_name": extracted_name,
            },
            "tender_id": tender_id,
        }
        response = {
            "result": llm_match,
            "confidence": float(llm_match.get("confidence", 0.0)),
            "reasoning": llm_match.get("reasoning", ""),
            "is_simulated": False,
            "model_version": llm_response.model or "",
            "prompt_hash": llm_response.prompt_hash or "",
        }
        stub.log_invocation(conn, tender_id, request, response)
    except Exception as exc:
        logger.warning("entity LLM log failed: %s", exc)


# ─── Corrigendum mapping (LLM-first with rules fallback) ─────────────────


def map_corrigendum_union(
    corrigendum_text: str,
    nit_criteria_summary: str,
    llm_client: Optional[LLMClient] = None,
) -> UnionResult:
    """Map amendments in a corrigendum to their target NIT clauses.

    This is an LLM-primary task — reference resolution is hard to do
    with rules alone. Rules act as a sanity check ("did the LLM hallucinate
    a clause reference that doesn't exist in the NIT?").
    """
    client = llm_client or get_default_client()

    llm_amendments: list[dict] = []
    llm_conf = 0.0
    llm_response: Optional[LLMResponse] = None

    if client.is_configured and corrigendum_text.strip():
        user_prompt = build_corrigendum_mapping_user_prompt(
            corrigendum_text=corrigendum_text,
            nit_criteria_summary=nit_criteria_summary,
        )
        llm_response = client.structured_extraction(
            system=CORRIGENDUM_MAPPING_SYSTEM,
            user=user_prompt,
            schema_hint=CORRIGENDUM_MAPPING_SCHEMA,
        )
        if llm_response.data and isinstance(llm_response.data, dict):
            llm_amendments = llm_response.data.get("amendments") or []
            if llm_amendments:
                llm_conf = sum(
                    float(a.get("confidence", 0.5)) for a in llm_amendments
                ) / len(llm_amendments)

    # Rules-based sanity check: extract referenced clauses from NIT text
    import re
    nit_clause_refs = set(re.findall(r"\b\d+(?:\.\d+)+(?:\([a-z0-9]+\))?\b",
                                     nit_criteria_summary))

    # Validate every LLM-proposed target_clause_ref exists in the NIT
    validated: list[dict] = []
    for amendment in llm_amendments:
        ref = amendment.get("target_clause_ref", "")
        amendment["_clause_exists_in_nit"] = ref in nit_clause_refs
        validated.append(amendment)

    hallucinated = sum(1 for a in validated if not a["_clause_exists_in_nit"])
    if validated:
        consensus_conf = (llm_conf * (1 - hallucinated / len(validated))) if validated else 0.0
    else:
        consensus_conf = 0.0

    return UnionResult(
        value=validated,
        confidence=consensus_conf,
        rules_value={"nit_clause_refs_found": list(nit_clause_refs)},
        rules_confidence=1.0 if nit_clause_refs else 0.0,
        llm_value={"amendments": llm_amendments},
        llm_confidence=llm_conf,
        agreement="agree" if hallucinated == 0 else "disagree",
        agreement_score=1.0 - (hallucinated / len(validated) if validated else 0),
        method="rules+llm" if llm_amendments else "rules_only",
        llm_model_version=llm_response.model if llm_response else "",
        llm_prompt_hash=llm_response.prompt_hash if llm_response else "",
        llm_tokens={
            "in": llm_response.tokens_in if llm_response else 0,
            "out": llm_response.tokens_out if llm_response else 0,
            "latency_ms": llm_response.latency_ms if llm_response else 0,
        },
        reasoning=(
            f"LLM found {len(llm_amendments)} amendments, "
            f"{hallucinated} had invalid clause refs (hallucinated)"
        ),
    )


# ─── Internal helpers ────────────────────────────────────────────────────


def _compute_rules_confidence(
    criteria: list[dict],
    document_text: str,
) -> float:
    """Rules confidence scales with number of criteria extracted + coverage."""
    if not criteria:
        return 0.3
    # Each criterion bumps confidence, diminishing returns
    base = 0.5 + min(0.08 * len(criteria), 0.4)
    # Bump if we detected clause references (structured NIT)
    if any(c.get("source_clause_ref") for c in criteria):
        base = min(1.0, base + 0.05)
    return round(base, 3)


def _compute_llm_confidence(
    criteria: list[dict],
    response: LLMResponse,
) -> float:
    """LLM confidence: high when criteria extracted, medium when empty."""
    if response.error:
        return 0.0
    if not criteria:
        return 0.3
    base = 0.6 + min(0.05 * len(criteria), 0.3)
    return round(base, 3)


def _compute_consensus_confidence(
    rules_conf: float,
    llm_conf: float,
    agreement: str,
) -> float:
    """Combine two branch confidences into a single consensus score.

    Agreement is a multiplier — both branches saying the same thing
    justifies higher confidence than either alone. Disagreement caps
    consensus at the lower of the two (we shouldn't be MORE confident
    because we disagreed).
    """
    if llm_conf == 0.0:
        return rules_conf
    if rules_conf == 0.0:
        return llm_conf * 0.8  # mild penalty for no rules backup
    if agreement == "agree":
        # Geometric mean + agreement bonus
        import math
        mean = math.sqrt(rules_conf * llm_conf)
        return round(min(1.0, mean + 0.1), 3)
    if agreement == "disagree":
        return round(min(rules_conf, llm_conf) * 0.7, 3)
    return round((rules_conf + llm_conf) / 2, 3)


def _merge_criteria(
    rules_list: list[dict],
    llm_list: list[dict],
) -> tuple[list[dict], str, float]:
    """Merge two criterion lists using semantic similarity.

    Returns:
        (merged_criteria, agreement_label, agreement_score)

    The agreement_label is one of:
        "agree"      — both branches found the same criteria
        "disagree"   — branches found different criteria
        "rules_only" — LLM found nothing
        "llm_only"   — rules found nothing
        "neither"    — neither branch found anything
    """
    if not rules_list and not llm_list:
        return [], "neither", 0.0
    if not llm_list:
        return _tag_sources(rules_list, ["rules"]), "rules_only", 0.5
    if not rules_list:
        return _tag_sources(llm_list, ["llm"]), "llm_only", 0.5

    # Encode criterion texts and match across branches
    rule_texts = [c.get("criterion_text", "") for c in rules_list]
    llm_texts = [c.get("criterion_text", "") for c in llm_list]

    try:
        rule_embs = embedding_service.encode(rule_texts)
        llm_embs = embedding_service.encode(llm_texts)
    except Exception as exc:
        logger.warning(
            "Embedding failed during merge (%s); falling back to union",
            exc,
        )
        return (
            _tag_sources(rules_list, ["rules"]) + _tag_sources(llm_list, ["llm"]),
            "disagree",
            0.0,
        )

    # Similarity matrix: rules × llm
    sim_matrix = rule_embs @ llm_embs.T  # already normalised

    SIMILARITY_THRESHOLD = 0.70

    merged: list[dict] = []
    matched_llm: set[int] = set()
    agreements = 0

    for ri, rule_crit in enumerate(rules_list):
        best_j = -1
        best_sim = 0.0
        for lj in range(len(llm_list)):
            if lj in matched_llm:
                continue
            sim = float(sim_matrix[ri][lj])
            if sim > best_sim:
                best_sim = sim
                best_j = lj

        if best_j >= 0 and best_sim >= SIMILARITY_THRESHOLD:
            # Branches agree on this criterion — merge with LLM preferred
            # for criterion_text (more natural) but rules preferred for
            # threshold_value (more precise)
            llm_crit = llm_list[best_j]
            merged_crit = dict(llm_crit)  # LLM version as base
            # Keep rules' threshold_value if it has more structured data
            rules_tv = rule_crit.get("threshold_value")
            llm_tv = llm_crit.get("threshold_value")
            if rules_tv and isinstance(rules_tv, dict) and "rupees" in rules_tv:
                merged_crit["threshold_value"] = rules_tv
            elif llm_tv:
                merged_crit["threshold_value"] = llm_tv
            else:
                merged_crit["threshold_value"] = rules_tv
            # Use rules' source_clause_ref if LLM didn't find one
            if not merged_crit.get("source_clause_ref"):
                merged_crit["source_clause_ref"] = rule_crit.get(
                    "source_clause_ref", ""
                )
            merged_crit["_sources"] = ["rules", "llm"]
            merged_crit["_agreement_similarity"] = best_sim
            # Regenerate a fresh uuid so we don't collide with either branch
            import uuid
            merged_crit["id"] = str(uuid.uuid4())
            merged.append(merged_crit)
            matched_llm.add(best_j)
            agreements += 1
        else:
            # Rules found a criterion LLM didn't
            merged_crit = dict(rule_crit)
            merged_crit["_sources"] = ["rules"]
            merged.append(merged_crit)

    # Add LLM criteria that weren't matched by rules
    for lj, llm_crit in enumerate(llm_list):
        if lj in matched_llm:
            continue
        merged_crit = dict(llm_crit)
        merged_crit["_sources"] = ["llm"]
        merged.append(merged_crit)

    total = len(rules_list) + len(llm_list) - agreements
    agreement_score = agreements / total if total > 0 else 0.0
    agreement_label = (
        "agree" if agreement_score >= 0.6
        else "disagree" if agreement_score < 0.3
        else "partial"
    )

    return merged, agreement_label, agreement_score


def _tag_sources(criteria: list[dict], sources: list[str]) -> list[dict]:
    """Annotate every criterion with its source branch(es)."""
    out = []
    for c in criteria:
        cc = dict(c)
        cc["_sources"] = list(sources)
        out.append(cc)
    return out


def _normalise_llm_criterion(raw: dict, source_document_id: str) -> dict:
    """Map an LLM-output criterion to the rules-format schema."""
    import uuid

    # Flatten threshold_value if LLM nested it under sub_criteria
    tv = raw.get("threshold_value") or {}

    return {
        "id": str(uuid.uuid4()),
        "criterion_text": raw.get("criterion_text", "").strip(),
        "criterion_type": raw.get("criterion_type", "qualitative_assessment"),
        "threshold_value": tv if isinstance(tv, dict) else {},
        "gfr_override_permitted": bool(raw.get("gfr_override_permitted", True)),
        "gfr_rule_number": raw.get("gfr_rule_number"),
        "is_mandatory": bool(raw.get("is_mandatory", False)),
        "source_document_id": source_document_id,
        "source_clause_ref": raw.get("source_clause_ref", ""),
        "amendment_history": [],
        "acceptable_evidence_types": raw.get("acceptable_evidence_types") or [],
        "measurement_period": raw.get("measurement_period"),
        "needs_review": bool(raw.get("needs_review", False)),
        "review_reason": raw.get("review_reason"),
    }


def _semantic_qualitative(
    criterion_text: str,
    bidder_evidence: str,
    cpm_precedents: list[dict],
) -> tuple[dict, float]:
    """Rules-branch qualitative eval via cosine similarity."""
    if not criterion_text:
        return ({"verdict": "REVIEW", "factors": []}, 0.0)

    if bidder_evidence:
        crit_doc_sim = embedding_service.similarity_score(
            criterion_text, bidder_evidence
        )
    else:
        crit_doc_sim = 0.0

    best_precedent_sim = 0.0
    best_precedent: Optional[str] = None
    if cpm_precedents:
        cand_texts = [
            p.get("resolved_interpretation") or p.get("criterion_text") or ""
            for p in cpm_precedents
        ]
        ranked = embedding_service.rank_by_similarity(
            criterion_text, cand_texts, top_k=1
        )
        if ranked:
            idx, sim = ranked[0]
            best_precedent = cand_texts[idx]
            best_precedent_sim = sim

    if crit_doc_sim >= 0.55 and best_precedent_sim >= 0.55:
        verdict = "PASS"
    else:
        verdict = "REVIEW"

    confidence = min(1.0, 0.5 * crit_doc_sim + 0.5 * best_precedent_sim)

    factors = [
        f"criterion↔document similarity: {crit_doc_sim:.2f}",
        f"best precedent similarity: {best_precedent_sim:.2f}",
    ]
    if best_precedent:
        factors.append(f"top precedent: {best_precedent[:120]}")

    return (
        {
            "verdict": verdict,
            "factors": factors,
            "similarity_to_criterion": crit_doc_sim,
            "best_precedent_similarity": best_precedent_sim,
        },
        confidence,
    )


def _verdict_agreement(
    rules_verdict: str,
    llm_verdict: str,
    rules_value: dict,
    llm_value: dict,
) -> tuple[str, float]:
    """Determine whether the two branches agree on verdict."""
    if rules_verdict == llm_verdict:
        return ("agree", 1.0)

    # "PASS vs REVIEW" is softer disagreement than "PASS vs FAIL"
    soft_pairs = {("PASS", "REVIEW"), ("REVIEW", "PASS"),
                  ("FAIL", "REVIEW"), ("REVIEW", "FAIL")}
    if (rules_verdict, llm_verdict) in soft_pairs:
        return ("partial", 0.5)

    return ("disagree", 0.0)


def _default_reasoning(
    rules_value: dict,
    llm_value: dict,
    agreement: str,
) -> str:
    """Build a default reasoning string when the LLM didn't provide one."""
    factors = rules_value.get("factors", [])
    top = factors[0] if factors else "no specific factors identified"
    return (
        f"Semantic analysis: {top}. "
        f"Rules-LLM {agreement}. "
        f"Routing to officer review."
    )
