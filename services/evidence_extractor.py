"""Real evidence extractor for VerdictAI Layer 3.

Replaces the hash-seeded fake extraction in ``l3_evidence`` with real
regex + pdfplumber table parsing over OCR'd bidder documents.

Functions:
- extract_numeric_value:     Rs / Crore / Lakh figures, fiscal year if nearby
- extract_from_tables:       pdfplumber table rows matching search terms
- extract_fiscal_year:       "FY 2023-24", "2022-23" patterns
- extract_validity_date:     Certificate validity dates
- extract_registration_number: GSTIN / PAN / CIN / MSME patterns
- extract_company_name:      Primary bidder name from letterhead
- extract_project_list:      Project rows with value + date + description

Every function returns a ``confidence`` in [0.0, 1.0] based on how many
fields were cleanly extracted and whether the match is ambiguous. There
are no randomised values — identical inputs always produce identical
outputs.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import pdfplumber

from services.criterion_extractor import to_rupees


logger = logging.getLogger(__name__)


# ─── Numeric value extraction ────────────────────────────────────────────────


# Capture "Rs. 12.45 Crore", "INR 10 Cr", "₹ 5,00,00,000", "10.8 Lakh" etc.
_NUMERIC_DEFAULT_PATTERNS = [
    # Currency prefix then number then unit
    r"(?:rs\.?|inr|₹)\s*(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?P<unit>crore|lakh|lac|cr\.?|l\.?|million|mn|thousand|k)?",
    # Number then unit, no currency prefix required
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?P<unit>crore|lakh|lac|cr\.?|l\.?|million|mn)",
]


def extract_numeric_value(
    text: str,
    patterns: Optional[list[str]] = None,
) -> Optional[dict]:
    """Extract the first numeric Rs/Cr/Lakh figure from text.

    Preference order: values with an explicit currency prefix win over
    bare digits; values with an explicit unit (crore/lakh) win over
    unit-less numbers; earlier matches win on ties.

    Args:
        text: Any string (OCR output, table cell, prose).
        patterns: Optional custom regex list with named groups
                  ``value`` and ``unit``. Defaults to currency-aware patterns.

    Returns:
        ``{"amount": int_rupees, "unit": str, "raw_value": float,
           "fiscal_year": Optional[str], "confidence": float}``
        or None if no numeric value is present.
    """
    if not text:
        return None
    patterns = patterns or _NUMERIC_DEFAULT_PATTERNS

    best: Optional[tuple[int, re.Match, str]] = None  # (quality, match, pattern_tag)

    # First pass prefers currency-prefixed matches (quality=2)
    for m in re.finditer(patterns[0], text, flags=re.IGNORECASE):
        quality = 2 if m.group("unit") else 1
        if best is None or quality > best[0]:
            best = (quality, m, "currency")

    # Second pass — only consider if no currency-prefixed match found
    if best is None:
        for pat in patterns[1:]:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                best = (1, m, "unit")
                break
            if best:
                break

    if best is None:
        return None

    quality, match, _ = best
    raw_value_str = match.group("value").replace(",", "")
    try:
        raw_value = float(raw_value_str)
    except ValueError:
        return None

    unit = (match.group("unit") or "").lower()
    amount = to_rupees(raw_value, unit) if unit else int(round(raw_value))

    # Fiscal year if one appears within a short window around the match
    window = text[max(0, match.start() - 40): match.end() + 40]
    fiscal_year = extract_fiscal_year(window)

    # Confidence: quality=2 (currency + unit) → 0.9, quality=1 (unit only) → 0.75,
    # anything else → 0.6. Boost if fiscal_year also extracted cleanly.
    base = {2: 0.9, 1: 0.75}.get(quality, 0.6)
    confidence = min(1.0, base + (0.05 if fiscal_year else 0.0))

    return {
        "amount": amount,
        "unit": unit or "rupees",
        "raw_value": raw_value,
        "fiscal_year": fiscal_year,
        "confidence": confidence,
    }


# ─── Table extraction ────────────────────────────────────────────────────────


def extract_from_tables(
    pdf_path: str,
    search_terms: list[str],
) -> list[dict]:
    """Extract rows from PDF tables that contain any of the search terms.

    Uses pdfplumber's ``extract_tables`` on each page. Rows are joined
    into a single lowercase string for matching; matching rows are
    returned with their page number and numeric content (if any).

    Args:
        pdf_path: Path to the bidder submission PDF.
        search_terms: List of substrings to search for (case-insensitive).
                      A row matches if it contains ANY of the terms.

    Returns:
        List of dicts with keys:
            page, row_index, cells, matched_term, numeric_value, confidence
        Empty list on parse failure or zero matches.
    """
    if not pdf_path or not search_terms:
        return []

    lower_terms = [t.lower() for t in search_terms if t]
    matches: list[dict] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                except Exception as exc:  # pragma: no cover - rare parser glitch
                    logger.warning(
                        "extract_tables failed on page %d of %s: %s",
                        page_num, pdf_path, exc,
                    )
                    continue

                for t_idx, table in enumerate(tables):
                    for r_idx, row in enumerate(table):
                        cells = [
                            (c or "").strip() for c in row if c is not None
                        ]
                        row_text = " ".join(cells).lower()
                        matched_term = next(
                            (t for t in lower_terms if t in row_text), None
                        )
                        if not matched_term:
                            continue

                        joined = " ".join(cells)
                        numeric = extract_numeric_value(joined)
                        # Confidence floor 0.7 for a verified match; bump
                        # higher if a numeric value was also extracted.
                        confidence = 0.85 if numeric else 0.7

                        matches.append(
                            {
                                "page": page_num,
                                "table_index": t_idx,
                                "row_index": r_idx,
                                "cells": cells,
                                "matched_term": matched_term,
                                "numeric_value": numeric,
                                "confidence": confidence,
                            }
                        )
    except Exception as exc:
        logger.error(
            "Failed to extract tables from %s: %s: %s",
            pdf_path, type(exc).__name__, exc,
        )
        return []

    return matches


# ─── Dates and fiscal years ──────────────────────────────────────────────────


_FY_PATTERNS = [
    # "FY 2023-24" or "F.Y. 2023-2024"
    r"\b(?:f\.?y\.?|financial\s+year)\s*(?P<fy>(?:20)?\d{2}\s*[-–/]\s*(?:20)?\d{2,4})\b",
    # Standalone "2023-24" / "2022-2023"
    r"\b(?P<fy>(?:20)\d{2}\s*[-–/]\s*(?:20)?\d{2,4})\b",
]


def extract_fiscal_year(text: str) -> Optional[str]:
    """Return a canonicalised fiscal-year string or None.

    Normalises "FY 2023-24", "F.Y. 2023/2024", "2022-23" → "2023-24"-style
    "YYYY-YY" form.
    """
    if not text:
        return None
    for pat in _FY_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            raw = m.group("fy")
            # Normalise separators and whitespace
            raw = re.sub(r"\s+", "", raw).replace("/", "-").replace("–", "-")
            parts = raw.split("-")
            if len(parts) != 2:
                return raw
            start, end = parts
            if len(start) == 2:
                start = "20" + start
            if len(end) == 4:
                end = end[-2:]
            return f"{start}-{end}"
    return None


# Various DD-MM-YYYY / YYYY-MM-DD / DD Month YYYY forms
_DATE_PATTERNS = [
    (r"\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})\b", "dmy"),
    (r"\b(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})\b", "ymd"),
    (
        r"\b(\d{1,2})\s+"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+"
        r"(\d{4})\b",
        "dm_word_y",
    ),
    (
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+"
        r"(\d{4})\b",
        "m_word_y",
    ),
]

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def extract_validity_date(text: str) -> Optional[dict]:
    """Extract a certificate validity date from text.

    Looks for phrases like "valid up to 31-12-2025" / "valid till 2024".
    Falls back to any date appearing after the keyword "valid".

    Args:
        text: Raw document text.

    Returns:
        ``{"date": "YYYY-MM-DD", "is_valid": bool, "confidence": float}``
        where ``is_valid`` is True iff the date is strictly after today.
        Returns None if no parsable date is found.
    """
    if not text:
        return None

    # Prefer a date that follows a validity keyword
    validity_cue = re.search(
        r"(?:valid(?:ity)?\s*(?:up\s*to|till|until|upto)?|expires?\s+on)\s*[:\-]?\s*(.{0,60})",
        text,
        flags=re.IGNORECASE,
    )
    search_regions = [validity_cue.group(1)] if validity_cue else []
    search_regions.append(text)

    for region in search_regions:
        iso = _first_iso_date(region)
        if iso:
            try:
                parsed = datetime.strptime(iso, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                is_valid = parsed > datetime.now(timezone.utc)
            except ValueError:
                is_valid = None
            confidence = 0.85 if validity_cue else 0.7
            return {
                "date": iso,
                "is_valid": is_valid,
                "confidence": confidence,
            }
    return None


def _first_iso_date(text: str) -> Optional[str]:
    """Return the first parseable date in ISO YYYY-MM-DD form."""
    for pat, kind in _DATE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            if kind == "dmy":
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif kind == "ymd":
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif kind == "dm_word_y":
                d = int(m.group(1))
                mo = _MONTH_MAP.get(m.group(2).lower()[:3], 0)
                y = int(m.group(3))
            elif kind == "m_word_y":
                d = 1
                mo = _MONTH_MAP.get(m.group(1).lower()[:3], 0)
                y = int(m.group(2))
            else:
                continue
            if mo == 0 or d == 0:
                continue
            datetime(y, mo, d)  # validate
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except (ValueError, IndexError):
            continue
    return None


# ─── Registration numbers ────────────────────────────────────────────────────


_REG_PATTERNS = {
    # GSTIN: 2-digit state + 5 letters + 4 digits + letter + digit + Z + alnum
    "gst": re.compile(
        r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z][A-Z0-9])\b"
    ),
    # PAN: 5 letters + 4 digits + letter
    "pan": re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b"),
    # CIN: L/U + 5 digits + 2 letters + 4 digits + 3 letters + 6 digits
    "cin": re.compile(
        r"\b([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b"
    ),
    # TAN: 4 letters + 5 digits + 1 letter
    "tan": re.compile(r"\b([A-Z]{4}\d{5}[A-Z])\b"),
    # UDYAM: UDYAM-XX-NN-NNNNNNN
    "udyam": re.compile(r"\b(UDYAM[-\s]?[A-Z]{2}[-\s]?\d{2}[-\s]?\d{7})\b"),
    # ISO: "ISO 9001:2015"
    "iso": re.compile(r"\b(ISO\s*\d{4,5}(?::\d{2,4})?)\b"),
}


def extract_registration_number(
    text: str,
    cert_type: str,
) -> Optional[str]:
    """Extract a registration number of a specific type from text.

    Uses format-aware regex so only well-formed identifiers match
    (rather than any uppercase sequence). This is important because
    GSTINs embed a PAN, so a naive PAN match would pick up GSTIN
    substrings; we run cert-specific patterns only.

    Args:
        text: Any document text.
        cert_type: One of "gst", "pan", "cin", "tan", "udyam", "iso"
                   (case-insensitive). Unknown types return None.

    Returns:
        The matched identifier in canonical form, or None.
    """
    if not text or not cert_type:
        return None
    key = cert_type.lower().strip()
    # Normalise common aliases.
    if key in {"gstin", "gst_number", "gst_registration"}:
        key = "gst"
    if key in {"pan_card", "pan_number"}:
        key = "pan"

    pattern = _REG_PATTERNS.get(key)
    if pattern is None:
        return None

    # For PAN we need to exclude PANs that are actually the inner segment
    # of a GSTIN — otherwise we'd wrongly "find" a PAN inside a GSTIN.
    if key == "pan":
        # Strip all GSTINs from the text before running the PAN pattern.
        cleaned = _REG_PATTERNS["gst"].sub(" ", text.upper())
        m = pattern.search(cleaned)
        return m.group(1) if m else None

    m = pattern.search(text.upper() if key != "iso" else text)
    return m.group(1).strip() if m else None


# ─── Company name extraction ────────────────────────────────────────────────


# "M/s XYZ & Co Pvt Ltd" with a required legal suffix.
# Suffix alternatives use word boundaries so "Co" doesn't match inside
# "Corp" and "Ltd" doesn't match inside "Ltda".
_MS_NAME_RE = re.compile(
    r"M/?s\.?\s+(?P<name>[A-Z][A-Za-z0-9&'.\- ]{2,80}?"
    r"(?:\s+(?:Pvt\.?|Private)\s+Ltd\.?\b|\s+Ltd\.?\b|\s+Limited\b|\s+LLP\b|"
    r"\s+Corporation\b|\s+Corp\.?\b|\s+Co\.?\b|\s+Inc\.?\b))",
    flags=re.IGNORECASE,
)

# "M/s XYZ" without a legal suffix — catches letterhead names
_MS_NAME_SIMPLE_RE = re.compile(
    r"M/?s\.?\s+(?P<name>[A-Z][A-Za-z0-9&'.\- ]{2,80})",
    flags=re.IGNORECASE,
)

# "Something Pvt Ltd" / "Something Limited" not prefixed by M/s.
# Anchored at a word boundary; word before Pvt/Ltd/... must not be a
# common filler word ("This", "The", "By", "To"), which we strip below.
_SUFFIX_NAME_RE = re.compile(
    r"(?<![A-Za-z])(?P<name>[A-Z][A-Za-z0-9&'.\- ]{2,80}?"
    r"(?:\s+(?:Pvt\.?|Private)\s+Ltd\.?\b|\s+Ltd\.?\b|\s+Limited\b|\s+LLP\b|"
    r"\s+Corporation\b|\s+Corp\.?\b))"
)

# Common English filler words that shouldn't be part of a company name
# even if they happen to start with an uppercase letter.
_NAME_FILLER_PREFIXES = {
    "this", "that", "these", "those", "the", "by", "to", "for", "in",
    "from", "of", "and", "or", "our", "their", "his", "her", "my",
    "we", "is", "are", "was", "were", "submitted", "issued", "signed",
    "dated", "registered", "certified", "company", "firm", "bidder",
}


def extract_company_name(text: str) -> Optional[str]:
    """Extract the primary company name from document text.

    Strategy:
      1. Prefer explicit "M/s XYZ Pvt Ltd" constructions (first occurrence).
      2. Fall back to the first name ending in a recognised legal suffix.
      3. Return None if nothing matches.

    The returned name is trimmed of surrounding whitespace and its
    leading "M/s " (if present), since downstream comparison uses
    :func:`services.entity_matcher.normalise_company_name`
    which handles suffixes separately.

    Args:
        text: Document text, ideally the first page / letterhead.

    Returns:
        Company name as a plain string, or None.
    """
    if not text:
        return None

    m = _MS_NAME_RE.search(text)
    if m:
        return _clean_company_name(m.group("name"))

    m = _SUFFIX_NAME_RE.search(text)
    if m:
        return _clean_company_name(m.group("name"))

    # Last resort — accept "M/s Xyz" without a legal suffix. This is
    # common on letterheads.
    m = _MS_NAME_SIMPLE_RE.search(text)
    if m:
        # Trim at line break or obvious sentence terminator.
        raw = m.group("name")
        raw = re.split(r"[\n\r]|\s+(?:on\s+|dated|date)\b", raw, maxsplit=1)[0]
        return _clean_company_name(raw)

    return None


def _clean_company_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    # Drop stray trailing punctuation
    name = name.rstrip(",;:.")

    # If the match starts with a filler word, drop leading tokens until
    # a non-filler token is found.
    tokens = name.split()
    while tokens and tokens[0].lower() in _NAME_FILLER_PREFIXES:
        tokens.pop(0)
    return " ".join(tokens)


# ─── Project list extraction ─────────────────────────────────────────────────


_PROJECT_DESCRIPTION_HEADERS = {
    "description", "project", "work", "nature of work", "scope",
    "particulars", "supply order", "work order",
}
_PROJECT_VALUE_HEADERS = {
    "value", "order value", "amount", "contract value", "cost",
}
_PROJECT_DATE_HEADERS = {
    "date", "completion date", "completed on", "date of completion",
    "completion", "period of completion",
}


def extract_project_list(
    pdf_path: str,
    text: str,
) -> list[dict]:
    """Extract a list of past projects for the temporal_recency criterion.

    Primary strategy: look through PDF tables and identify rows that map
    a description → value → date. Fallback: scan prose for three-way
    co-occurrences of (description keyword, currency amount, date).

    Args:
        pdf_path: Path to the bidder submission PDF (may be None/empty
                  to skip table extraction).
        text: OCR'd / extracted text of the submission.

    Returns:
        List of dicts with keys:
          description (str), value (dict or None), completion_date (str or None),
          confidence (float). Ordered by appearance in the document.
    """
    projects: list[dict] = []

    # ── Table pass ──
    if pdf_path:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        tables = page.extract_tables() or []
                    except Exception:  # pragma: no cover
                        continue
                    for table in tables:
                        rows = [
                            [(c or "").strip() for c in row]
                            for row in table if row
                        ]
                        projects.extend(_projects_from_table_rows(rows))
        except Exception as exc:
            logger.warning(
                "pdfplumber failed on %s while extracting projects: %s",
                pdf_path, exc,
            )

    # ── Prose fallback: only if no structured projects were found ──
    if not projects and text:
        projects.extend(_projects_from_prose(text))

    return projects


def _projects_from_table_rows(rows: list[list[str]]) -> list[dict]:
    """Map a single table's rows into project dicts using header heuristics."""
    if not rows:
        return []

    header = [c.lower() for c in rows[0]]
    desc_col = _find_header(header, _PROJECT_DESCRIPTION_HEADERS)
    value_col = _find_header(header, _PROJECT_VALUE_HEADERS)
    date_col = _find_header(header, _PROJECT_DATE_HEADERS)

    # If we can't locate a description column, this table isn't a project list.
    if desc_col is None:
        return []

    out: list[dict] = []
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        if desc_col >= len(row):
            continue
        description = row[desc_col].strip()
        if not description or description.lower() in _PROJECT_DESCRIPTION_HEADERS:
            continue

        value_cell = row[value_col] if value_col is not None and value_col < len(row) else ""
        date_cell = row[date_col] if date_col is not None and date_col < len(row) else ""

        numeric = extract_numeric_value(value_cell) if value_cell else None
        iso_date = _first_iso_date(date_cell) if date_cell else None

        # Confidence: description-only 0.55; +0.2 for value, +0.15 for date
        confidence = 0.55
        if numeric:
            confidence += 0.2
        if iso_date:
            confidence += 0.15

        out.append(
            {
                "description": description,
                "value": numeric,
                "completion_date": iso_date,
                "confidence": round(min(confidence, 1.0), 2),
            }
        )
    return out


def _find_header(header: list[str], candidates: set[str]) -> Optional[int]:
    """Return the index of the first header cell that matches any candidate."""
    for idx, cell in enumerate(header):
        c = cell.strip().lower()
        if c in candidates:
            return idx
        for cand in candidates:
            if cand in c:
                return idx
    return None


# Used by prose fallback: find lines that look like "X - Rs Y - DATE"
_PROSE_PROJECT_RE = re.compile(
    r"(?P<desc>[^\n]{8,120}?)"
    r"[\-:–|]\s*(?:rs\.?|inr|₹)?\s*(?P<val>[\d,]+(?:\.\d+)?)\s*"
    r"(?P<unit>crore|lakh|lac|cr\.?|l\.?)?"
    r"[\s\S]{0,60}?"
    r"(?P<date>\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|"
    r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4})",
    flags=re.IGNORECASE,
)


def _projects_from_prose(text: str) -> list[dict]:
    """Best-effort project extraction from free-form text."""
    out: list[dict] = []
    for m in _PROSE_PROJECT_RE.finditer(text):
        description = re.sub(r"\s+", " ", m.group("desc")).strip(" -:–|")
        if not description:
            continue
        numeric = extract_numeric_value(
            f"Rs {m.group('val')} {m.group('unit') or ''}"
        )
        iso_date = _first_iso_date(m.group("date"))
        confidence = 0.55 + (0.2 if numeric else 0.0) + (0.1 if iso_date else 0.0)
        out.append(
            {
                "description": description,
                "value": numeric,
                "completion_date": iso_date,
                "confidence": round(min(confidence, 1.0), 2),
            }
        )
    return out
