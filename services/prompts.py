"""LLM prompt templates for VerdictAI.

These are pinned versions so audit records can reproduce exactly which
prompt produced a given output. When prompts change materially, bump
the PROMPT_VERSION constant so downstream code can detect it.

Style guidelines:
- System prompts establish role, constraints, output contract.
- User prompts carry the document / data. Always include the canonical
  input text verbatim so the hash changes when inputs change.
- JSON schemas are specified as example structures, not JSON-schema
  formal notation (LLMs handle examples more reliably).
"""

from __future__ import annotations


PROMPT_VERSION = "v1.0"


# ─── Criterion extraction ────────────────────────────────────────────────

CRITERION_EXTRACTION_SYSTEM = """\
You are an expert at analysing Indian government procurement tender documents (NITs).
Your job is to extract eligibility criteria with full legal precision.

Indian tenders issued under GFR 2017 contain numbered clauses. Each clause typically
states one or more eligibility requirements. You must:

1. Identify every distinct eligibility criterion (financial, categorical, temporal, qualitative)
2. Classify each into one of five types:
   - numeric_threshold:      Specific Rs/Crore/Lakh amounts (turnover, net worth, EMD, project value)
   - categorical_presence:   Documents that must be valid/present (GST, PAN, ISO cert, licences)
   - temporal_recency:       "N works in last M years" — experience recency requirements
   - composite:              Multi-part criteria joined by AND (decompose into sub-criteria)
   - qualitative_assessment: Subjective requirements needing judgment ("adequate", "satisfactory")

3. Tag GFR-mandatory criteria (override NOT permitted):
   - Financial thresholds (turnover, net worth) → GFR Rule 173(i)
   - Statutory registrations (GST, PAN, CIN, TAN) → GFR Rule 144
   - Bid security / EMD → GFR Rule 170
   - Debarment / blacklist → GFR Rule 151

4. Extract the threshold_value as structured data (amount in rupees for numeric,
   document type for categorical, count+period for temporal).

5. Note any ambiguous language that might need officer interpretation — flag these
   as "needs_review": true with a one-line explanation.

Be conservative: if you're uncertain, mark the criterion as qualitative_assessment
and set needs_review=true rather than guessing. The system will surface your
extraction to a human officer for approval before any bidder evaluation.
"""


CRITERION_EXTRACTION_SCHEMA = """\
{
  "criteria": [
    {
      "criterion_text": "full clause text as a single sentence",
      "criterion_type": "numeric_threshold | categorical_presence | temporal_recency | composite | qualitative_assessment",
      "threshold_value": {
        // Shape depends on criterion_type:
        // numeric_threshold: {"value": 10.0, "unit": "crore", "rupees": 100000000, "label": "annual turnover"}
        // categorical_presence: {"required": true, "document": "GST Registration"}
        // temporal_recency: {"count": 3, "period": 5, "period_unit": "years", "what": "similar supply orders"}
        // qualitative_assessment: {"assessment_text": "adequate manufacturing capacity"}
        // composite: {"sub_criteria": [ ...same-shape dicts... ]}
      },
      "is_mandatory": true,
      "gfr_override_permitted": false,
      "gfr_rule_number": "GFR Rule 173(i) | GFR Rule 144 | ... | null",
      "source_clause_ref": "4.1 | 4.1(a) | section-name | empty string",
      "measurement_period": "3 financial years | 5 years | null",
      "acceptable_evidence_types": ["ca_certificate", "audited_balance_sheet", "gst_certificate", "..."],
      "needs_review": false,
      "review_reason": "null or a one-line explanation of ambiguity"
    }
  ],
  "amendment_indicators": ["refer addendum", "as amended", "..."],
  "extraction_notes": "brief summary of what was extracted"
}
"""


def build_criterion_extraction_user_prompt(document_text: str) -> str:
    """Build the user-role prompt for criterion extraction."""
    return f"""Extract all eligibility criteria from this tender document.
Return ONLY valid JSON matching the schema I described.

TENDER DOCUMENT TEXT:
---
{document_text}
---

Extract every distinct eligibility criterion. Decompose composite clauses into
sub-criteria. Tag GFR-mandatory status accurately. Return JSON only."""


# ─── Qualitative evaluation ──────────────────────────────────────────────

QUALITATIVE_EVAL_SYSTEM = """\
You are evaluating whether a bidder's submission satisfies a qualitative eligibility
criterion in an Indian government tender.

Criteria like "adequate manufacturing capacity" or "satisfactory track record" require
judgment. Your job is to:

1. Read the criterion and the bidder's evidence (supporting documents, prose).
2. Consider any CPM precedents (past officer interpretations of similar language
   in this department / category) — these are the institutional standard.
3. Produce a verdict with CHAIN-OF-THOUGHT reasoning stored in the audit record.

Verdict rules:
- PASS only if clear, specific, verifiable evidence supports the criterion
  AND any CPM precedents align with this interpretation.
- FAIL only if there is affirmative evidence the criterion is NOT met. Absence
  of evidence is NOT proof of absence — prefer REVIEW.
- REVIEW (default) if evidence is partial, ambiguous, or contradicts precedents.

CRITICAL: The system does NOT auto-commit FAIL verdicts from qualitative reasoning.
Even if you say FAIL, it will be routed to a human officer. So favour REVIEW over
FAIL unless the evidence is unambiguous.

Provide structured reasoning that a procurement officer can audit: list the specific
facts you observed, how each maps to the criterion, and how CPM precedents informed
your verdict.
"""


QUALITATIVE_EVAL_SCHEMA = """\
{
  "verdict": "PASS | FAIL | REVIEW",
  "confidence": 0.0 to 1.0,
  "reasoning": "2-4 sentence chain-of-thought explanation for the officer",
  "factors": [
    "specific fact 1 observed in the evidence",
    "specific fact 2 ...",
    "relevant CPM precedent cited (if any)"
  ],
  "precedent_alignment": "aligned | diverges | no_precedents_available",
  "key_quote": "most relevant short quote from the bidder's evidence (or empty string)"
}
"""


def build_qualitative_eval_user_prompt(
    criterion_text: str,
    bidder_name: str,
    bidder_evidence: str,
    cpm_precedents: list[dict],
) -> str:
    """Build the user-role prompt for qualitative evaluation."""
    precedents_block = "NONE available — use conservative judgment."
    if cpm_precedents:
        lines = []
        for i, p in enumerate(cpm_precedents, start=1):
            lines.append(
                f"[{i}] Similar criterion: \"{p.get('criterion_text', '')}\"\n"
                f"    Officer interpretation: \"{p.get('resolved_interpretation', '')}\"\n"
                f"    Outcome: {p.get('verdict', 'unknown')} "
                f"({p.get('officer_action', 'unknown')})"
            )
        precedents_block = "\n\n".join(lines)

    return f"""Evaluate this qualitative criterion against the bidder's evidence.
Return ONLY valid JSON.

CRITERION:
{criterion_text}

BIDDER: {bidder_name}

BIDDER EVIDENCE (from uploaded documents):
---
{bidder_evidence[:6000]}
---

CPM PRECEDENTS (past officer interpretations of similar language in this department):
{precedents_block}

Produce your verdict with chain-of-thought reasoning. Favour REVIEW over FAIL when
in doubt — humans will make the final call. Return JSON only."""


# ─── Corrigendum amendment mapping ──────────────────────────────────────

CORRIGENDUM_MAPPING_SYSTEM = """\
You are analysing a corrigendum to an Indian government NIT. Your job is to map
each amendment in the corrigendum to the specific clause it modifies in the
original NIT, then determine the original value, amended value, and reason.

A corrigendum may contain multiple amendments. Each amendment has:
- A target clause reference (e.g. "Clause 4.1" or "Section IV, clause (a)")
- A change description ("is revised to", "shall be read as", "superseded by")
- A new value

Return structured JSON only. Be precise — the audit trail depends on accurate
clause references.
"""


CORRIGENDUM_MAPPING_SCHEMA = """\
{
  "amendments": [
    {
      "target_clause_ref": "4.1 | 4.1(a) | ...",
      "change_type": "threshold_revision | deadline_revision | document_revision | other",
      "original_text": "verbatim original text from the NIT if cited",
      "amended_text": "verbatim new text from the corrigendum",
      "amendment_reason": "one-line reason if provided, else empty string",
      "confidence": 0.0 to 1.0
    }
  ],
  "corrigendum_number": "1 | 2 | ...",
  "corrigendum_date": "YYYY-MM-DD or empty string"
}
"""


def build_corrigendum_mapping_user_prompt(
    corrigendum_text: str,
    nit_criteria_summary: str,
) -> str:
    """Build the user-role prompt for corrigendum mapping."""
    return f"""Map each amendment in this corrigendum to the specific original
NIT clause it modifies. Return ONLY valid JSON.

ORIGINAL NIT CRITERIA (for clause reference):
---
{nit_criteria_summary}
---

CORRIGENDUM TEXT:
---
{corrigendum_text}
---

Identify every amendment. For each, extract target_clause_ref, change_type,
original_text, amended_text. Return JSON only."""


# ─── Entity disambiguation ──────────────────────────────────────────────

ENTITY_DISAMBIGUATION_SYSTEM = """\
You are determining whether two company-name variants refer to the same legal
entity in an Indian government procurement context.

Indian company names often appear in multiple forms:
- Abbreviations: "Pvt Ltd" ↔ "Private Limited"
- Punctuation: "M/s" prefix, "&" vs "and"
- Regional suffixes: "India Ltd" ↔ "Limited India"
- Parent/subsidiary relationships: "ABC Infrastructure Pvt Ltd" (bidder) vs
  "ABC Group Holdings Ltd" (parent company). Parent-subsidiary substitution
  is a documented procurement fraud vector — flag it explicitly.

Classify the relationship:
- same_entity: Same legal entity with surface-form variations
- abbreviation: One is a recognised abbreviation of the other
- parent_company: One is the parent / holding company of the other (FRAUD RISK)
- subsidiary: Reverse of parent
- unrelated: Different legal entities entirely

Return structured JSON only. Err toward "needs_review" when uncertain.
"""


ENTITY_DISAMBIGUATION_SCHEMA = """\
{
  "classification": "same_entity | abbreviation | parent_company | subsidiary | unrelated",
  "confidence": 0.0 to 1.0,
  "reasoning": "one-line explanation",
  "fraud_risk": true | false,
  "needs_review": true | false
}
"""


def build_entity_disambiguation_user_prompt(name_a: str, name_b: str) -> str:
    """Build the user-role prompt for entity disambiguation."""
    return f"""Determine the relationship between these two company names.
Return ONLY valid JSON.

REGISTERED NAME (from bidder application):  {name_a!r}
EXTRACTED NAME (from submitted document):  {name_b!r}

Classify the relationship. If parent/subsidiary, flag fraud_risk=true.
Return JSON only."""
