"""Rule-based criterion extractor for VerdictAI Layer 2.

Replaces the pre-configured LLM stub with real pattern-matching over
Indian government tender language. The patterns here were derived from
GFR 2017 style NITs, CRPF/CPWD/PWD clauses, and corrigenda. Covers the
~90% of criterion clauses that follow predictable formulaic structure.

Pipeline:
1. Split raw tender text into candidate clauses.
2. For each candidate, try each pattern family:
     - numeric_threshold       (turnover, net worth, EMD, work value)
     - categorical_presence    (GST, PAN, ISO, MSME, CIN, etc.)
     - temporal_recency        (N similar works in last M years)
     - qualitative_assessment  (adequate / satisfactory / suitable / ...)
3. Classify GFR-mandatory status via keyword lookup.
4. Emit criterion dicts compatible with the L2 ETS Builder schema.

No hard-coded scenarios. No randomness. Every output is derivable from
the input text.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional


# ─── Pattern families ─────────────────────────────────────────────────────────
#
# Every pattern uses named groups (?P<name>...) so `parse_threshold` can pull
# values out deterministically. Patterns are intentionally greedy on labels and
# conservative on numeric capture to avoid swallowing adjacent clauses.

NUMERIC_THRESHOLD_PATTERNS: list[str] = [
    # Turnover: "annual turnover of not less than Rs. 5 crore".
    # Also tolerates Indian legal drafting "Rs. 10 (Ten) Crore" where
    # the digit is followed by its word form in brackets.
    r"(?P<label>(?:annual\s+)?(?:average\s+)?turnover)\s+"
    r"(?:of\s+(?:not\s+less\s+than\s+)?|shall\s+be\s+(?:not\s+less\s+than\s+)?|"
    r">=?\s*|≥\s*|minimum\s+(?:of\s+)?)?\s*"
    r"(?:rs\.?|inr|₹)?\s*"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?:\(\s*[A-Za-z][A-Za-z\s\-]{0,40}\)\s*)?"
    r"(?P<unit>crore|lakh|lac|million|billion|cr\.?|l\.?|mn|bn)",
    # Net worth: "positive net worth of Rs X crore / Rs X lakh"
    r"(?P<label>net\s+worth)\s+(?:of\s+)?"
    r"(?:not\s+less\s+than\s+)?"
    r"(?:rs\.?|inr|₹)?\s*"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?:\(\s*[A-Za-z][A-Za-z\s\-]{0,40}\)\s*)?"
    r"(?P<unit>crore|lakh|lac|cr\.?|l\.?)",
    # EMD / Earnest Money Deposit / Bid Security
    r"(?P<label>emd|earnest\s+money\s+deposit|bid\s+security)\s+"
    r"(?:of\s+)?(?:rs\.?|inr|₹)?\s*"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?:\(\s*[A-Za-z][A-Za-z\s\-]{0,40}\)\s*)?"
    r"(?P<unit>crore|lakh|lac|cr\.?|l\.?|thousand)",
    # Work / contract / project value
    r"(?P<label>(?:single\s+)?work(?:\s+order)?|contract|project)\s+"
    r"(?:of\s+)?(?:value\s+)?(?:rs\.?|inr|₹)?\s*"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?:\(\s*[A-Za-z][A-Za-z\s\-]{0,40}\)\s*)?"
    r"(?P<unit>crore|lakh|lac|cr\.?|l\.?)",
]


CATEGORICAL_PRESENCE_PATTERNS: list[str] = [
    # GST registration
    r"\b(?P<doc>(?:valid\s+)?gst(?:\s+registration)?(?:\s+certificate)?|"
    r"goods\s+and\s+services\s+tax)\b",
    # PAN card / Permanent Account Number
    r"\b(?P<doc>pan(?:\s+card)?|permanent\s+account\s+number)\b",
    # ISO certifications like "ISO 9001:2015"
    r"\b(?P<doc>iso\s*\d{4,}(?:[:\-]\d{2,4})?)(?:\s+certification)?\b",
    # MSME / Udyam / SSI / Udyog Aadhaar
    r"\b(?P<doc>msme|udyam|ssi|udyog\s+aadhaar)\s*"
    r"(?:registration|certificate)\b",
    # CIN / Corporate Identification Number
    r"\b(?P<doc>cin|corporate\s+identification\s+number)\b",
    # TAN
    r"\b(?P<doc>tan|tax\s+deduction\s+(?:and\s+)?collection\s+account\s+number)\b",
    # EPF / ESI / ESIC
    r"\b(?P<doc>epf|employee\s+provident\s+fund|esic?)\b",
    # Shop and Establishment
    r"\b(?P<doc>shop\s+(?:and\s+)?establishment(?:\s+(?:act|certificate))?)\b",
    # FSSAI / food safety
    r"\b(?P<doc>fssai|food\s+safety)\b",
    # Drug licence
    r"\b(?P<doc>drug\s+licen[cs]e|form\s+20b)\b",
    # Solvency certificate / banker's certificate
    r"\b(?P<doc>(?:banker'?s?|solvency)\s+certificate)\b",
    # Pollution / PCC
    r"\b(?P<doc>pollution\s+(?:control|under\s+control)\s+certificate|pcc)\b",
]


TEMPORAL_RECENCY_PATTERNS: list[str] = [
    # "N similar works in last M years"
    # Allows an optional parenthetical spell-out after each digit
    # ("3 (three) similar supply orders in the last 5 (five) years"),
    # and accepts a qualifier noun between "similar" and the head noun
    # ("similar supply orders" — supply qualifying orders).
    r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:\s*\([^)]*\))?\s+"
    r"(?P<what>similar\s+(?:\w+\s+)?(?:work|contract|project|supply|order)s?|"
    r"completed\s+(?:\w+\s+)?(?:work|contract|project|supply|order)s?|"
    r"work(?:s)?\s+of\s+similar\s+nature)\s+"
    r"(?:completed\s+)?(?:in\s+)?(?:the\s+)?"
    r"(?:last|preceding|previous)\s+"
    r"(?P<period>\d+)"
    r"(?:\s*\([^)]*\))?\s+"
    r"(?P<period_unit>years?|months?|financial\s+years?|fy)",
    # "X years of operational/working experience"
    r"(?P<what>(?:minimum\s+)?(?:of\s+)?experience|"
    r"operational\s+experience|"
    r"(?:working|business)\s+experience)\s+"
    r"(?:of\s+)?(?P<period>\d+)\s+"
    r"(?P<period_unit>years?|months?)",
]


QUALITATIVE_PATTERNS: list[str] = [
    r"adequate\s+(?:technical\s+)?(?:capacity|capability|experience|"
    r"manpower|infrastructure|facilit(?:y|ies))",
    r"satisfactory\s+(?:track\s+record|performance|past\s+performance)",
    r"sufficient\s+(?:resources?|capacity|experience)",
    r"suitable\s+(?:qualifications?|experience|track\s+record)",
    r"relevant\s+(?:experience|expertise|qualifications?)",
    r"technical\s+(?:competence|capability|expertise)",
    r"good\s+(?:standing|reputation|performance)",
    # Domain demonstrations: "demonstrate adequate manufacturing capacity"
    r"demonstrate\s+(?:adequate|sufficient|relevant)\s+\w+\s+(?:capacity|capability|experience)",
]


# ─── GFR-mandatory keywords ───────────────────────────────────────────────────
#
# If any of these keywords appear in a criterion text for a type where
# mandatory status is gated by content (see is_gfr_mandatory), we treat it
# as GFR-mandatory and record the applicable rule.

GFR_MANDATORY_KEYWORDS = {
    "gst": "GFR Rule 144",
    "pan": "GFR Rule 144",
    "turnover": "GFR Rule 173(i)",
    "net worth": "GFR Rule 173(i)",
    "financial capacity": "GFR Rule 173(i)",
    "bid security": "GFR Rule 170",
    "emd": "GFR Rule 170",
    "earnest money": "GFR Rule 170",
    "solvency": "GFR Rule 173(i)",
    "debarment": "GFR Rule 151",
    "blacklist": "GFR Rule 151",
    "statutory": "GFR Rule 144",
}


# ─── Clause splitting ─────────────────────────────────────────────────────────
#
# Government tender language is organised as numbered clauses ("4.1", "4.1(a)").
# We split on clause-number boundaries first; if no clause markers are found,
# we fall back to sentence boundaries.

_CLAUSE_MARKER_RE = re.compile(
    r"(?im)^\s*(?:clause\s+)?(?P<ref>\d+(?:\.\d+)+(?:\s*\([a-z0-9]+\))?)\s*[-–—:.]?\s+"
)
# Sentence splitter: a period/question/exclam followed by whitespace then
# a capital letter. Refuses to split after common abbreviations (Rs., No.)
# where a digit follows the period, since those are part of a single
# sentence like "Rs. 5 Crore".
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)
_AMENDMENT_INDICATORS = [
    "as amended",
    "refer addendum",
    "superseded by",
    "refer corrigendum",
    "amendment no",
    "revised vide",
    "modified by",
]


# ─── Public API ───────────────────────────────────────────────────────────────


def extract_criteria_from_text(
    text: str,
    source_document_id: str = "",
) -> list[dict]:
    """Extract structured criteria from raw tender text.

    Walks the text clause-by-clause, attempts each pattern family, and
    emits one criterion dict per matched clause (a clause may yield
    multiple criteria if it contains multiple independent requirements,
    e.g. turnover AND experience in a single paragraph).

    The returned dicts are shaped to match the L2 ETS Builder's
    ``raw_criteria`` consumer in ``extract_criteria`` so they can flow
    straight through to the ``criteria`` table.

    Args:
        text: Concatenated page text from a tender NIT or corrigendum.
        source_document_id: ID of the source document, threaded into
            the returned dicts for provenance.

    Returns:
        List of criterion dicts with keys:
            id, criterion_text, criterion_type, threshold_value,
            gfr_override_permitted, gfr_rule_number, is_mandatory,
            source_clause_ref, amendment_history,
            acceptable_evidence_types, measurement_period.
    """
    if not text or not text.strip():
        return []

    clauses = _split_into_clauses(text)
    criteria: list[dict] = []
    seen_signatures: set[tuple] = set()

    for clause_ref, clause_text in clauses:
        # Each pattern family is tried; a single clause can produce
        # multiple criteria (composite clauses do this routinely).
        for raw in _try_all_patterns(clause_text):
            criterion_type = raw["criterion_type"]
            mandatory, rule_number = is_gfr_mandatory(
                clause_text, criterion_type
            )

            # Deduplicate on (type, primary-key-of-threshold, clause-ref)
            signature = (
                criterion_type,
                _threshold_signature(raw.get("threshold_value")),
                clause_ref,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            criterion = {
                "id": str(uuid.uuid4()),
                "criterion_text": raw["criterion_text"],
                "criterion_type": criterion_type,
                "threshold_value": raw.get("threshold_value"),
                "gfr_override_permitted": not mandatory,
                "gfr_rule_number": rule_number,
                "is_mandatory": mandatory,
                "source_document_id": source_document_id,
                "source_clause_ref": clause_ref or "",
                "amendment_history": [],
                "acceptable_evidence_types": raw.get(
                    "acceptable_evidence_types", []
                ),
                "measurement_period": raw.get("measurement_period"),
            }
            criteria.append(criterion)

    return criteria


def classify_criterion_type(criterion_text: str) -> str:
    """Classify a single criterion string into one of the 5 types.

    Precedence: numeric_threshold > temporal_recency >
    categorical_presence > qualitative_assessment > composite fallback.

    Args:
        criterion_text: A single sentence or clause.

    Returns:
        One of: "numeric_threshold", "categorical_presence",
        "temporal_recency", "qualitative_assessment", "composite".
    """
    t = criterion_text.lower()

    for pat in NUMERIC_THRESHOLD_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            # If numeric AND temporal both match, it's still numeric
            # (threshold is the stronger constraint for evaluation).
            return "numeric_threshold"
    for pat in TEMPORAL_RECENCY_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "temporal_recency"
    for pat in CATEGORICAL_PRESENCE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "categorical_presence"
    for pat in QUALITATIVE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "qualitative_assessment"

    # Composite fallback — a sentence that lists multiple sub-criteria
    # joined by "and" after at least one strong keyword. Rare in practice
    # for a bare call; the main pipeline emits composites by aggregating.
    return "qualitative_assessment"


def is_gfr_mandatory(
    criterion_text: str,
    criterion_type: str,
) -> tuple[bool, Optional[str]]:
    """Determine if a criterion is GFR-mandatory.

    Rules (encoded from GFR 2017 and the design spec):
      - numeric_threshold over turnover / net worth / solvency / EMD
        → GFR-mandatory.
      - categorical_presence for GST / PAN / statutory registrations
        → GFR-mandatory.
      - temporal_recency (N works in last M years) → discretionary.
      - qualitative_assessment → override permitted.

    Args:
        criterion_text: The clause text (any case).
        criterion_type: The classified criterion type.

    Returns:
        Tuple ``(is_mandatory, gfr_rule_number)``. Rule number is the
        most-specific GFR reference we could identify, or None.
    """
    t = criterion_text.lower()

    # Explicit rule number cited in the clause itself ("GFR Rule 173(i)")
    cited = re.search(
        r"gfr(?:\s+rule)?\s+(\d+(?:\([a-z0-9]+\))?(?:\([a-z0-9]+\))?)",
        t,
    )
    cited_rule = f"GFR Rule {cited.group(1)}" if cited else None

    if criterion_type in {"numeric_threshold", "categorical_presence"}:
        for keyword, rule in GFR_MANDATORY_KEYWORDS.items():
            if keyword in t:
                return True, cited_rule or rule
        # Without a mandatory keyword, a numeric/categorical clause is
        # still treated as mandatory only if the NIT says so explicitly.
        if "mandatory" in t or "pre-qualification" in t:
            return True, cited_rule
        return False, cited_rule

    # temporal_recency and qualitative_assessment default to discretionary
    # unless the clause explicitly marks itself mandatory.
    if "mandatory" in t or "pre-qualification" in t:
        return True, cited_rule
    return False, cited_rule


def parse_threshold(match: re.Match, criterion_type: str) -> dict:
    """Parse ``threshold_value`` from a regex match based on type.

    Produces the JSON-serialisable dict stored in the ``criteria``
    table's ``threshold_value`` column.

    Args:
        match: The regex Match object from any pattern family.
        criterion_type: Which family matched, governs field shape.

    Returns:
        Dict with type-specific keys:
          - numeric_threshold: {value, unit, rupees, label}
          - categorical_presence: {required, document}
          - temporal_recency: {count, period, period_unit, what}
          - qualitative_assessment: {assessment_text}
    """
    groups = match.groupdict()

    if criterion_type == "numeric_threshold":
        raw_value = groups.get("value", "0").replace(",", "")
        try:
            numeric_value = float(raw_value)
        except ValueError:
            numeric_value = 0.0
        unit = (groups.get("unit") or "").lower()
        return {
            "value": numeric_value,
            "unit": unit,
            "rupees": to_rupees(numeric_value, unit),
            "label": (groups.get("label") or "").strip(),
        }

    if criterion_type == "categorical_presence":
        return {
            "required": True,
            "document": (groups.get("doc") or "").strip(),
        }

    if criterion_type == "temporal_recency":
        count_raw = (groups.get("count") or "").strip()
        count = _word_to_int(count_raw)
        try:
            period = int(groups.get("period") or 0)
        except (TypeError, ValueError):
            period = 0
        period_unit = (groups.get("period_unit") or "years").lower()
        return {
            "count": count,
            "period": period,
            "period_unit": period_unit,
            "what": (groups.get("what") or "").strip(),
        }

    return {"assessment_text": match.group(0).strip()}


def detect_amendment_indicators(text: str) -> list[str]:
    """Return which amendment-indicator phrases appear in text.

    Used by L2 to detect when an NIT references a corrigendum that was
    not uploaded. Matches are case-insensitive and substring-based.

    Args:
        text: Document text to scan.

    Returns:
        List of indicator phrases found (lowercase forms).
    """
    if not text:
        return []
    t = text.lower()
    return [ind for ind in _AMENDMENT_INDICATORS if ind in t]


def to_rupees(value: float, unit: str) -> int:
    """Convert a numeric value in crore/lakh/etc. to absolute rupees.

    Unknown units fall through with multiplier 1 so numeric literals
    without an explicit unit don't get silently amplified.

    Args:
        value: Numeric quantity (may already be a float like 12.45).
        unit: Unit string ("crore", "lakh", "lac", "cr", "l", etc.).

    Returns:
        Integer rupees. e.g. to_rupees(12.45, "crore") → 124_500_000.
    """
    u = (unit or "").lower().strip().rstrip(".")
    multipliers = {
        "crore": 10_000_000, "cr": 10_000_000,
        "lakh": 100_000, "lac": 100_000, "l": 100_000,
        "thousand": 1_000, "k": 1_000,
        "million": 1_000_000, "mn": 1_000_000,
        "billion": 1_000_000_000, "bn": 1_000_000_000,
    }
    return int(round(value * multipliers.get(u, 1)))


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _split_into_clauses(text: str) -> list[tuple[str, str]]:
    """Split text into (clause_ref, clause_text) pairs.

    Prefers numbered-clause boundaries ("4.1", "4.2(a)"). Falls back to
    paragraph splitting if no clause markers are present, then to
    sentence boundaries as a last resort.
    """
    # Try clause-number markers first.
    markers = list(_CLAUSE_MARKER_RE.finditer(text))
    if markers:
        clauses: list[tuple[str, str]] = []
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            ref = m.group("ref").strip()
            body = text[start:end].strip()
            if body:
                clauses.append((ref, body))
        if clauses:
            return clauses

    # Paragraph fallback.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        return [("", p) for p in paragraphs]

    # Sentence fallback.
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return [("", s) for s in sentences]


def _try_all_patterns(clause_text: str) -> list[dict]:
    """Run every pattern family on a clause and emit raw criterion dicts.

    A single clause can produce multiple criteria (e.g. "turnover >= 10 Cr
    AND valid GST registration AND ISO 9001" yields three).

    Returned dicts carry ``criterion_type`` and ``threshold_value`` only;
    the caller is responsible for wrapping with provenance and GFR fields.
    """
    results: list[dict] = []

    # numeric_threshold (each numeric clause is its own criterion)
    for pat in NUMERIC_THRESHOLD_PATTERNS:
        for m in re.finditer(pat, clause_text, flags=re.IGNORECASE):
            results.append(
                {
                    "criterion_text": _trim_sentence(clause_text),
                    "criterion_type": "numeric_threshold",
                    "threshold_value": parse_threshold(m, "numeric_threshold"),
                    "acceptable_evidence_types": _evidence_types_for_label(
                        m.groupdict().get("label", "")
                    ),
                    "measurement_period": _measurement_period_from(clause_text),
                }
            )

    # temporal_recency
    for pat in TEMPORAL_RECENCY_PATTERNS:
        for m in re.finditer(pat, clause_text, flags=re.IGNORECASE):
            tv = parse_threshold(m, "temporal_recency")
            results.append(
                {
                    "criterion_text": _trim_sentence(clause_text),
                    "criterion_type": "temporal_recency",
                    "threshold_value": tv,
                    "acceptable_evidence_types": [
                        "completion_certificate",
                        "work_order",
                    ],
                    "measurement_period": f"{tv.get('period', 0)} {tv.get('period_unit', 'years')}".strip(),
                }
            )

    # categorical_presence (each distinct doc mention is its own criterion)
    seen_docs: set[str] = set()
    for pat in CATEGORICAL_PRESENCE_PATTERNS:
        for m in re.finditer(pat, clause_text, flags=re.IGNORECASE):
            tv = parse_threshold(m, "categorical_presence")
            doc_key = _normalise_doc_key(tv["document"])
            if not doc_key or doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)
            results.append(
                {
                    "criterion_text": _trim_sentence(clause_text),
                    "criterion_type": "categorical_presence",
                    "threshold_value": tv,
                    "acceptable_evidence_types": ["certificate"],
                    "measurement_period": None,
                }
            )

    # qualitative_assessment — only if no stronger pattern matched.
    # Otherwise every numeric clause would also appear as qualitative
    # because "adequate" etc. often show up as framing words.
    if not results:
        for pat in QUALITATIVE_PATTERNS:
            m = re.search(pat, clause_text, flags=re.IGNORECASE)
            if m:
                results.append(
                    {
                        "criterion_text": _trim_sentence(clause_text),
                        "criterion_type": "qualitative_assessment",
                        "threshold_value": parse_threshold(
                            m, "qualitative_assessment"
                        ),
                        "acceptable_evidence_types": [],
                        "measurement_period": None,
                    }
                )
                break

    return results


def _trim_sentence(text: str, max_len: int = 400) -> str:
    """Normalise whitespace and clip to a single readable clause."""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _threshold_signature(threshold: Optional[dict]) -> str:
    """Stable signature for dedupe. Different reps → different signatures."""
    if not threshold:
        return ""
    if "rupees" in threshold:
        return f"num:{threshold.get('label', '')}:{threshold['rupees']}"
    if "document" in threshold:
        return f"doc:{threshold['document'].lower()}"
    if "count" in threshold:
        return (
            f"tmp:{threshold.get('count')}:"
            f"{threshold.get('period')}:{threshold.get('period_unit')}"
        )
    return f"qual:{threshold.get('assessment_text', '')[:60]}"


def _word_to_int(raw: str) -> int:
    """Convert digit strings OR spelled-out numerals (one..ten) to int."""
    if not raw:
        return 0
    raw = raw.strip().lower()
    if raw.isdigit():
        return int(raw)
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    return words.get(raw, 0)


def _measurement_period_from(text: str) -> Optional[str]:
    """Pull 'last N years' / 'last N financial years' out of a numeric clause."""
    m = re.search(
        r"(?:last|preceding|previous)\s+(\d+)\s+(years?|months?|financial\s+years?|fy)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return f"{m.group(1)} {m.group(2).lower()}"
    return None


def _normalise_doc_key(document: str) -> str:
    """Collapse doc-mention surface forms to a canonical key for dedupe.

    "valid GST registration certificate", "GST", "gst registration" all
    collapse to "gst" so a single clause doesn't emit three duplicate
    categorical_presence criteria for the same requirement.
    """
    if not document:
        return ""
    t = document.lower().strip()
    # Drop qualifier words
    for stop in ("valid", "registration", "certificate", "card", "number"):
        t = re.sub(rf"\b{stop}\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Canonicalise common aliases
    aliases = {
        "gst": {"gst", "goods and services tax"},
        "pan": {"pan", "permanent account"},
        "cin": {"cin", "corporate identification"},
        "tan": {"tan", "tax deduction"},
        "msme": {"msme", "udyam", "udyog aadhaar", "ssi"},
        "iso": None,  # ISO numbers are unique identifiers; keep as-is
        "epf": {"epf", "esi", "esic", "employee provident fund"},
        "fssai": {"fssai", "food safety"},
        "drug": {"drug licence", "drug license", "form 20b"},
        "solvency": {"bankers certificate", "banker s certificate", "solvency certificate"},
        "pcc": {"pcc", "pollution control certificate", "pollution under control certificate"},
        "shop": {"shop establishment", "shop and establishment"},
    }
    for canonical, forms in aliases.items():
        if forms is None:
            continue
        for form in forms:
            if form in t:
                return canonical
    # Pass-through: ISO-9001 style identifiers remain unique per number
    return t


def _evidence_types_for_label(label: str) -> list[str]:
    """Map a numeric-threshold label to its expected evidence document types."""
    l = (label or "").lower()
    if "turnover" in l or "net worth" in l:
        return ["ca_certificate", "audited_balance_sheet"]
    if "emd" in l or "earnest" in l or "bid security" in l:
        return ["emd_receipt", "bank_guarantee"]
    if "work" in l or "contract" in l or "project" in l:
        return ["work_order", "completion_certificate"]
    return []
