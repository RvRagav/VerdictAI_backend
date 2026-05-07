"""Entity Matcher service for VerdictAI.

Provides fuzzy company name matching with abbreviation expansion,
suffix removal, and mismatch type classification. Used by the
Evidence Extraction layer (L3) to detect entity mismatches between
registered bidder names and names extracted from documents.

Requirements: 6.6
"""

import re
from difflib import SequenceMatcher

# Common abbreviation expansions for Indian company names
ABBREVIATIONS = {
    "pvt": "private",
    "ltd": "limited",
    "co": "company",
    "corp": "corporation",
    "engg": "engineering",
    "infra": "infrastructure",
    "const": "construction",
    "tech": "technologies",
    "govt": "government",
}


def normalise_company_name(name: str) -> str:
    """Normalise a company name for comparison.

    Steps:
    1. Lowercase and strip whitespace
    2. Replace punctuation [.,\\-()&] with space
    3. Split into tokens
    4. Expand abbreviations using ABBREVIATIONS dict
    5. Remove common suffixes that don't affect identity
    6. Join with single space

    Args:
        name: Raw company name string.

    Returns:
        Normalised company name suitable for comparison.
    """
    # Lowercase and strip
    name = name.lower().strip()

    # Replace punctuation with space
    name = re.sub(r'[.,\-()&]', ' ', name)

    # Split into tokens
    tokens = name.split()

    # Expand abbreviations
    tokens = [ABBREVIATIONS.get(t, t) for t in tokens]

    # Remove common suffixes that don't affect identity
    remove_suffixes = {"private", "limited", "pvt", "ltd", "india"}
    tokens = [t for t in tokens if t not in remove_suffixes]

    # Join with single space
    return " ".join(tokens)


def match_entity(
    registered_name: str,
    extracted_name: str,
    threshold: float = 0.85,
) -> dict:
    """Fuzzy match with abbreviation expansion.

    Compares two company names after normalisation and returns a
    result dict compatible with the EntityMatchResult model.

    Matching logic:
    1. Normalise both names
    2. If exact match after normalisation → score 1.0, is_match=True
    3. Use SequenceMatcher for similarity score
    4. Check containment (parent company detection)
    5. Classify mismatch_type: parent_company, abbreviation, different_entity
    6. requires_review = True if score < threshold

    Args:
        registered_name: The officially registered company name.
        extracted_name: The name extracted from a document.
        threshold: Minimum similarity score for a match (default 0.85).

    Returns:
        Dict with keys: registered_name, extracted_name, similarity_score,
        is_match, mismatch_type, requires_review.
    """
    norm_registered = normalise_company_name(registered_name)
    norm_extracted = normalise_company_name(extracted_name)

    # Exact match after normalisation
    if norm_registered == norm_extracted:
        return {
            "registered_name": registered_name,
            "extracted_name": extracted_name,
            "similarity_score": 1.0,
            "is_match": True,
            "mismatch_type": None,
            "requires_review": False,
        }

    # Sequence similarity
    score = SequenceMatcher(None, norm_registered, norm_extracted).ratio()

    # Containment check (parent company detection)
    is_substring = (
        norm_registered in norm_extracted or norm_extracted in norm_registered
    )

    # Classify mismatch type
    mismatch_type = None
    if score >= threshold:
        # High similarity but not exact — likely abbreviation difference
        mismatch_type = "abbreviation"
    elif is_substring:
        # One name contains the other — parent company relationship
        mismatch_type = "parent_company"
    else:
        mismatch_type = "different_entity"

    return {
        "registered_name": registered_name,
        "extracted_name": extracted_name,
        "similarity_score": score,
        "is_match": score >= threshold,
        "mismatch_type": mismatch_type,
        "requires_review": score < threshold,
    }
