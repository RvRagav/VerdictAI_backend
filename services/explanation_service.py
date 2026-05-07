"""Officer-grade verdict explanations for VerdictAI.

Every verdict produced by the evaluation engine gets a rich,
human-readable explanation through this module. The goal is simple:
a procurement officer should never see a string like
"Semantic score criterion↔document=0.23" — they should see a full
account of what the system found, where, with what confidence, and
what it wants them to do next.

The shape is fixed so the frontend can rely on it:

    {
      "headline":         str,     # one sentence, verdict-forward
      "detail":           str,     # 2-3 sentence full narrative
      "facts":            [str],   # bullet-list of specific observations
      "source_reference": str,     # "Page 2 of <file>, clause 4.1"
      "confidence_note":  str,     # describes HOW the value was captured
      "next_action":      str,     # "Auto-commit" | "Officer confirmation..."
    }

The function is deliberately defensive about missing evidence fields.
Evidence shape is stable across criterion types but individual fields
(value, source_document_id, entity_match_result, etc.) may be None,
so every access goes through a safe getter.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


# Rupee-unit formatting — maps the canonical integer rupee amount used
# internally by the evidence extractor back to the natural phrasing
# ("Rs. 18.45 Cr", "Rs. 52 Lakh") that an officer would recognise.
_RUPEE_CRORE = 10_000_000  # 1 Cr
_RUPEE_LAKH = 100_000      # 1 L


def _format_rupees(amount: Optional[int]) -> str:
    """Pretty-print a rupee amount in crore/lakh/rupees as appropriate."""
    if amount is None:
        return "unknown amount"
    try:
        amount_int = int(amount)
    except (TypeError, ValueError):
        return str(amount)

    if amount_int >= _RUPEE_CRORE:
        value = amount_int / _RUPEE_CRORE
        return f"Rs. {value:.2f} Cr"
    if amount_int >= _RUPEE_LAKH:
        value = amount_int / _RUPEE_LAKH
        return f"Rs. {value:.2f} Lakh"
    return f"Rs. {amount_int:,}"


def _format_confidence(conf: Any) -> str:
    """Format a confidence value as ``0.91`` (two decimal places)."""
    try:
        return f"{float(conf):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _get(d: Optional[Mapping], *keys, default=None):
    """Safe nested getter: ``_get(evidence, 'value', 'amount')``."""
    cur: Any = d
    for key in keys:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _next_action_for_route(route: str) -> str:
    """Map a routing decision to a human-readable next step."""
    return {
        "auto_commit":
            "Auto-committed — no officer action required.",
        "hitl_review":
            "Officer confirmation required.",
        "mandatory_review":
            "Mandatory officer review required. Auto-commit blocked by rule.",
    }.get(route, "Routing decision pending.")


def _source_reference(
    evidence: Optional[Mapping],
    criterion: Optional[Mapping],
) -> str:
    """Build a human source citation: 'Page N of <file>, clause X.Y'."""
    page = _get(evidence, "source_page_number") or _get(evidence, "page")
    filename = (
        _get(evidence, "source_filename")
        or _get(evidence, "filename")
        or "bidder submission"
    )
    clause = _get(criterion, "source_clause_ref")

    parts = []
    if page is not None:
        parts.append(f"Page {page} of {filename}")
    else:
        parts.append(f"{filename}")

    if clause:
        parts.append(f"clause {clause}")

    return ", ".join(parts)


def _confidence_note(
    evidence: Optional[Mapping],
    criterion: Optional[Mapping],
) -> str:
    """Describe HOW the value was captured (OCR conf + method)."""
    extraction_conf = _get(evidence, "extraction_confidence")
    ocr_conf = _get(evidence, "ocr_confidence")
    method = _get(evidence, "evaluation_method") or "deterministic"
    ctype = _get(criterion, "criterion_type") or ""

    method_phrase = {
        "deterministic": "via regex / table extraction",
        "union_rules+llm": "via rules+LLM cross-validation",
        "llm_stub":        "via LLM qualitative reasoning",
        "none":            "with no evidence captured",
        "error":           "after an extraction error",
    }.get(method, f"via {method}")

    ec = _format_confidence(extraction_conf) if extraction_conf is not None else None
    oc = _format_confidence(ocr_conf) if ocr_conf is not None else None

    parts = [f"Extracted {method_phrase}"]
    if ec is not None:
        parts.append(f"at {ec} extraction confidence")
    if oc is not None and oc != "unknown":
        parts.append(f"(page OCR confidence {oc})")
    if ctype:
        parts.append(f"for {ctype.replace('_', ' ')} criterion")
    return ", ".join(parts) + "."


# ─── Per-type explanation builders ───────────────────────────────────────


def _build_numeric_facts(
    verdict: str,
    criterion: Mapping,
    evidence: Mapping,
) -> tuple[str, list[str]]:
    """Build (headline, facts) for numeric_threshold criteria."""
    value = evidence.get("value") or {}
    amount = value.get("amount")
    fy = value.get("fiscal_year")
    raw = value.get("raw_value")
    unit = value.get("unit")

    threshold_data: Mapping = {}
    tv = criterion.get("threshold_value")
    if isinstance(tv, Mapping):
        threshold_data = tv
    elif isinstance(tv, str) and tv:
        try:
            import json as _json
            parsed = _json.loads(tv)
            if isinstance(parsed, dict):
                threshold_data = parsed
        except (ValueError, TypeError):
            threshold_data = {}

    threshold_rupees = threshold_data.get("rupees")
    threshold_text = _format_rupees(threshold_rupees) if threshold_rupees \
        else (
            f"{threshold_data.get('value', 'unknown')} "
            f"{threshold_data.get('unit', '')}".strip()
            or "threshold"
        )

    extracted_text = _format_rupees(amount)

    fy_phrase = f" (FY {fy})" if fy else ""
    clause = criterion.get("source_clause_ref") or "the relevant clause"

    if verdict == "PASS":
        headline = (
            f"{extracted_text}{fy_phrase} meets the "
            f"{threshold_text} threshold set by {clause}."
        )
    elif verdict == "FAIL":
        headline = (
            f"{extracted_text}{fy_phrase} falls short of the "
            f"{threshold_text} threshold set by {clause}."
        )
    else:
        headline = (
            f"Numeric value for {clause} could not be determined with "
            f"sufficient confidence."
        )

    facts = [f"Extracted figure: {extracted_text}"]
    if raw is not None and unit:
        facts.append(f"Raw extraction: {raw} {unit}")
    if fy:
        facts.append(f"Fiscal year on record: {fy}")
    if threshold_rupees:
        facts.append(f"Required threshold: {threshold_text}")
    elif threshold_data:
        facts.append(f"Threshold data: {threshold_data}")
    return headline, facts


def _build_categorical_facts(
    verdict: str,
    criterion: Mapping,
    evidence: Mapping,
) -> tuple[str, list[str]]:
    """Build (headline, facts) for categorical_presence criteria."""
    value = evidence.get("value") or {}
    found = value.get("found", False)
    reg_num = value.get("registration_number")
    cert_type = (value.get("certificate_type") or "").upper() or "certificate"
    is_valid = value.get("is_valid")
    validity = value.get("validity_date")
    clause = criterion.get("source_clause_ref") or "the relevant clause"

    if verdict == "PASS":
        headline = (
            f"{cert_type} certificate found{' and valid' if is_valid else ''}; "
            f"satisfies {clause}."
        )
    elif verdict == "FAIL":
        if not found:
            headline = (
                f"Required {cert_type} certificate not found in bidder "
                f"submission; {clause} not satisfied."
            )
        else:
            headline = (
                f"{cert_type} certificate found but does not satisfy "
                f"validity requirements of {clause}."
            )
    else:
        headline = (
            f"{cert_type} certificate status could not be determined "
            f"with confidence; needs officer review."
        )

    facts = [f"Certificate type: {cert_type}"]
    if reg_num:
        facts.append(f"Registration number: {reg_num}")
    else:
        facts.append("Registration number: not located")
    if validity:
        facts.append(f"Validity date on document: {validity}")
    if is_valid is True:
        facts.append("Currently valid (expiry after today)")
    elif is_valid is False:
        facts.append("Expired as of today")
    return headline, facts


def _build_temporal_facts(
    verdict: str,
    criterion: Mapping,
    evidence: Mapping,
) -> tuple[str, list[str]]:
    """Build (headline, facts) for temporal_recency criteria."""
    value = evidence.get("value") or {}
    count = value.get("count", 0)
    required = value.get("required_count", 2)
    period = value.get("period") or "the required window"
    clause = criterion.get("source_clause_ref") or "the relevant clause"

    if verdict == "PASS":
        headline = (
            f"Bidder documented {count} similar projects within {period} — "
            f"meets the minimum of {required} required by {clause}."
        )
    elif verdict == "FAIL":
        headline = (
            f"Only {count} similar projects found within {period}; "
            f"{clause} requires at least {required}."
        )
    else:
        headline = (
            f"Could not reliably count similar projects within {period}; "
            f"officer review required."
        )

    facts = [
        f"Projects located: {count}",
        f"Required minimum: {required}",
    ]
    projects = value.get("projects") or []
    for p in projects[:3]:
        desc = (p.get("description") or "")[:80]
        date = p.get("completion_date") or "date unknown"
        facts.append(f"• {desc} — completed {date}")
    if len(projects) > 3:
        facts.append(f"… and {len(projects) - 3} more")
    return headline, facts


def _build_qualitative_facts(
    verdict: str,
    criterion: Mapping,
    evidence: Mapping,
    union_agreement: Optional[str],
) -> tuple[str, list[str]]:
    """Build (headline, facts) for qualitative_assessment criteria."""
    value = evidence.get("value") or {}
    reasoning = value.get("llm_reasoning") or ""
    union = value.get("union") or {}
    consensus = union.get("consensus_verdict") or verdict
    rules_verdict = union.get("rules_verdict")
    llm_verdict = union.get("llm_verdict")
    agreement = union_agreement or union.get("agreement")

    clause = criterion.get("source_clause_ref") or "the clause"

    if verdict == "PASS":
        headline = (
            f"Bidder's submission against {clause} satisfies the "
            f"qualitative requirements; consensus verdict PASS."
        )
    elif verdict == "FAIL":
        headline = (
            f"Bidder's submission against {clause} does not satisfy the "
            f"qualitative requirements; consensus verdict FAIL."
        )
    else:
        headline = (
            f"Qualitative assessment for {clause} is inconclusive; "
            f"officer judgment required."
        )

    facts: list[str] = []
    if rules_verdict:
        facts.append(f"Rules branch (semantic): {rules_verdict}")
    if llm_verdict:
        facts.append(f"LLM branch: {llm_verdict}")
    if agreement:
        facts.append(f"Branch agreement: {agreement}")
    if consensus and consensus != verdict:
        facts.append(f"Consensus verdict recorded: {consensus}")
    if reasoning:
        facts.append(f"LLM reasoning: {reasoning[:220]}")

    key_quote = union.get("key_quote") or value.get("key_quote")
    if key_quote:
        facts.append(f"Key passage: \"{key_quote[:220]}\"")

    return headline, facts


def _flag_facts(evidence: Optional[Mapping]) -> list[str]:
    """Append flag-specific facts (entity mismatch, stamps, etc.)."""
    if not evidence:
        return []
    facts: list[str] = []

    entity = evidence.get("entity_match_result") or {}
    if evidence.get("entity_match_flag"):
        reg = entity.get("registered_name") or "registered bidder name"
        ext = entity.get("extracted_name") or "name on documents"
        mismatch_type = entity.get("mismatch_type") or entity.get(
            "llm_classification"
        ) or "mismatch"
        facts.append(
            f"⚠ Entity mismatch: registered '{reg}' vs. extracted "
            f"'{ext}' (type: {mismatch_type})."
        )
        if entity.get("llm_reasoning"):
            facts.append(f"LLM reasoning: {entity['llm_reasoning'][:200]}")

    stamp_regions = evidence.get("stamp_regions") or []
    if stamp_regions:
        facts.append(
            f"⚠ {len(stamp_regions)} stamp region(s) detected on the "
            f"source page; value may be partially obscured."
        )

    return facts


# ─── Public API ──────────────────────────────────────────────────────────


def build_explanation(
    verdict: str,
    criterion: dict,
    evidence: dict,
    route: str,
    union_agreement: Optional[str] = None,
) -> dict:
    """Build a rich, officer-ready explanation dict.

    Args:
        verdict: "PASS", "FAIL", or "REVIEW".
        criterion: Criterion row/dict. Keys used: ``criterion_type``,
                   ``criterion_text``, ``source_clause_ref``,
                   ``threshold_value``, ``is_mandatory``,
                   ``gfr_override_permitted``, ``gfr_rule_number``.
        evidence: Evidence dict from L3. Keys used: ``value``,
                  ``extraction_confidence``, ``ocr_confidence``,
                  ``evaluation_method``, ``source_page_number``,
                  ``source_filename``, ``entity_match_flag``,
                  ``entity_match_result``, ``stamp_regions``.
        route: "auto_commit" | "hitl_review" | "mandatory_review".
        union_agreement: Optional — "agree" | "disagree" | "partial" —
                         for qualitative cases where both branches ran.

    Returns:
        Dict with keys ``headline``, ``detail``, ``facts``,
        ``source_reference``, ``confidence_note``, ``next_action``.
    """
    criterion = criterion or {}
    evidence = evidence or {}

    ctype = criterion.get("criterion_type") or ""

    # Build the per-type headline + facts.
    try:
        if ctype == "numeric_threshold":
            headline, facts = _build_numeric_facts(verdict, criterion, evidence)
        elif ctype == "categorical_presence":
            headline, facts = _build_categorical_facts(verdict, criterion, evidence)
        elif ctype == "temporal_recency":
            headline, facts = _build_temporal_facts(verdict, criterion, evidence)
        elif ctype == "qualitative_assessment":
            headline, facts = _build_qualitative_facts(
                verdict, criterion, evidence, union_agreement
            )
        else:
            headline = (
                f"Verdict {verdict} recorded for "
                f"{criterion.get('criterion_text') or 'criterion'}."
            )
            facts = []
    except Exception as exc:
        logger.warning("build_explanation per-type build failed: %s", exc)
        headline = f"Verdict {verdict} recorded."
        facts = []

    facts.extend(_flag_facts(evidence))

    source_ref = _source_reference(evidence, criterion)
    conf_note = _confidence_note(evidence, criterion)
    next_action = _next_action_for_route(route)

    # Compose the 2-3 sentence narrative. Keeps the tone procurement-
    # officer-ready: headline first, then how we got there, then what
    # to do next.
    detail_parts: list[str] = [headline, conf_note]
    if route == "auto_commit":
        detail_parts.append("Auto-committed — no officer action required.")
    elif route == "mandatory_review":
        reason = ""
        if criterion.get("is_mandatory") and verdict == "FAIL":
            reason = (
                " This is a mandatory criterion, so the FAIL verdict cannot "
                "be auto-committed and must be reviewed by the officer."
            )
        elif evidence.get("entity_match_flag"):
            reason = (
                " A potential entity-match issue was flagged, which "
                "triggers mandatory officer review regardless of verdict."
            )
        detail_parts.append("Officer confirmation required." + reason)
    else:  # hitl_review
        detail_parts.append(
            "Routed for officer review before this verdict is finalised."
        )

    # GFR caveat when relevant.
    if verdict == "FAIL" and criterion.get("gfr_override_permitted") is False:
        rule = criterion.get("gfr_rule_number") or "the applicable GFR rule"
        detail_parts.append(
            f"Override is not permitted under {rule}."
        )

    detail = " ".join(str(p).strip() for p in detail_parts if p)

    return {
        "headline": headline,
        "detail": detail,
        "facts": facts,
        "source_reference": source_ref,
        "confidence_note": conf_note,
        "next_action": next_action,
    }
