"""Real semantic service for VerdictAI.

Replaces the pre-configured LLM stub with deterministic, real
computations backed by:
  - :mod:`backend.services.criterion_extractor` for criterion extraction,
  - :mod:`backend.services.embedding_service` for semantic similarity,
  - :mod:`backend.services.evidence_extractor` for evidence surfacing.

The public class :class:`LLMStub` is preserved so existing callers keep
working, but its methods now produce results from the real services. In
particular ``is_simulated`` is always False and ``model_version``
reflects the real pinned models.

Response shape (stable across prompt types):

    {
        "result":          <prompt-type-specific payload>,
        "confidence":       float in [0.0, 1.0],
        "reasoning":        human-readable string,
        "is_simulated":     False,
        "model_version":    "semantic-v1.0+regex-v1.0",
        "prompt_hash":      64-char hex sha256 of canonical request,
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import sqlite3

from backend.services import (
    criterion_extractor,
    embedding_service,
    evidence_extractor,
)


logger = logging.getLogger(__name__)


# ─── Public class ────────────────────────────────────────────────────────────


class LLMStub:
    """Real semantic service exposed under the legacy LLMStub name.

    The three supported ``prompt_type`` values remain:

      * ``criterion_extraction``   — regex / NLP extraction of criteria
      * ``qualitative_evaluation`` — semantic scoring vs CPM precedents
      * ``similarity_assessment``  — pairwise semantic similarity

    Unknown prompt types return a safe REVIEW-oriented default (never
    an auto-commit). Every invocation should be logged via
    :meth:`log_invocation` for reproducibility.
    """

    MODEL_VERSION = "semantic-v1.0+regex-v1.0"

    def invoke(self, request: dict) -> dict:
        """Dispatch a prompt to the correct real-service handler.

        Args:
            request: Dict with keys ``prompt_type``, ``context``,
                     ``tender_id`` (optional), ``scenario_hint`` (ignored).

        Returns:
            Standard response dict — see module docstring.
        """
        prompt_type = request.get("prompt_type", "")
        context = request.get("context", {}) or {}
        prompt_hash = self._compute_prompt_hash(request)

        try:
            if prompt_type == "criterion_extraction":
                result, confidence, reasoning = self._handle_extraction(context)
            elif prompt_type == "qualitative_evaluation":
                result, confidence, reasoning = self._handle_qualitative(context)
            elif prompt_type == "similarity_assessment":
                result, confidence, reasoning = self._handle_similarity(context)
            else:
                result, confidence, reasoning = self._handle_default(prompt_type)
        except Exception as exc:  # defensive — never raise to caller
            logger.exception("semantic_service failed on %s: %s", prompt_type, exc)
            result, confidence, reasoning = (
                {"error": f"{type(exc).__name__}: {exc}"},
                0.0,
                "Semantic service raised an exception; routing to review.",
            )

        return {
            "result": result,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "is_simulated": False,
            "model_version": self.MODEL_VERSION,
            "prompt_hash": prompt_hash,
        }

    # ── Logging (unchanged interface) ─────────────────────────────────────

    def log_invocation(
        self,
        conn: sqlite3.Connection,
        tender_id: str,
        request: dict,
        response: dict,
    ) -> None:
        """Persist an invocation to the ``llm_stub_log`` audit table.

        The table name is kept for schema compatibility; the stored
        ``model_version`` identifies the real model used.
        """
        conn.execute(
            """
            INSERT INTO llm_stub_log
                (id, tender_id, prompt_type, prompt_hash, prompt_content,
                 response_content, model_version, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                tender_id,
                request.get("prompt_type", "unknown"),
                response.get("prompt_hash", ""),
                json.dumps(request, sort_keys=True, separators=(",", ":"), default=str),
                json.dumps(response, sort_keys=True, separators=(",", ":"), default=str),
                self.MODEL_VERSION,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    # ── Hashing ───────────────────────────────────────────────────────────

    def _compute_prompt_hash(self, request: dict) -> str:
        """Deterministic SHA-256 over a canonicalised JSON form."""
        payload = json.dumps(
            request, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── Handler: criterion extraction ────────────────────────────────────

    def _handle_extraction(self, context: dict) -> tuple[dict, float, str]:
        """Real criterion extraction via ``criterion_extractor``."""
        text = context.get("document_text", "") or ""
        source_document_id = context.get("document_id", "") or ""

        criteria = criterion_extractor.extract_criteria_from_text(
            text=text,
            source_document_id=source_document_id,
        )
        indicators = criterion_extractor.detect_amendment_indicators(text)

        # Confidence: 0.4 baseline. Scale up with how many criteria were
        # found (we expect 3-7 in a well-formed NIT). Clipped to 0.95 —
        # never fake-high-confidence just because regex matched.
        if not criteria:
            confidence = 0.3
        else:
            confidence = min(0.4 + 0.1 * len(criteria), 0.95)

        reasoning = (
            f"Extracted {len(criteria)} criteria via rule-based NLP "
            f"({criterion_extractor.__name__}). "
            f"Amendment indicators found: {indicators or 'none'}."
        )

        return (
            {
                "criteria": criteria,
                "extraction_notes": reasoning,
                "amendment_indicators": indicators,
            },
            confidence,
            reasoning,
        )

    # ── Handler: qualitative evaluation ──────────────────────────────────

    def _handle_qualitative(self, context: dict) -> tuple[dict, float, str]:
        """Score a bidder passage vs CPM precedents using embeddings.

        Inputs consumed from ``context``:
          - ``criterion_text`` (str, required)
          - ``document_text``  (str, the bidder passage to evaluate)
          - ``cpm_precedents`` (list[str], prior officer interpretations)
        """
        criterion_text = context.get("criterion_text", "") or ""
        document_text = context.get("document_text", "") or ""
        precedents = [
            p for p in (context.get("cpm_precedents") or []) if p
        ]

        if not criterion_text:
            return (
                {
                    "verdict": "REVIEW",
                    "factors": [],
                    "similarity_to_criterion": 0.0,
                    "best_precedent_similarity": 0.0,
                },
                0.0,
                "Missing criterion_text — cannot evaluate semantically.",
            )

        # Similarity of the bidder passage to the criterion language.
        if document_text:
            crit_doc_sim = embedding_service.similarity_score(
                criterion_text, document_text
            )
        else:
            crit_doc_sim = 0.0

        # Similarity of the criterion to the best matching CPM precedent.
        best_precedent_sim = 0.0
        best_precedent: Optional[str] = None
        if precedents:
            ranked = embedding_service.rank_by_similarity(
                criterion_text, precedents, top_k=1
            )
            if ranked:
                idx, sim = ranked[0]
                best_precedent = precedents[idx]
                best_precedent_sim = sim

        # Verdict logic — deliberately conservative. Only strong evidence
        # from BOTH the bidder passage AND a precedent yields PASS; any
        # FAIL determination requires HITL (spec §2.4: "The platform does
        # not auto-commit disqualifications based on LLM interpretation").
        if crit_doc_sim >= 0.55 and best_precedent_sim >= 0.55:
            verdict = "PASS"
        elif crit_doc_sim < 0.25 and not precedents:
            verdict = "REVIEW"
        else:
            verdict = "REVIEW"

        confidence = max(
            0.0,
            min(1.0, 0.5 * crit_doc_sim + 0.5 * best_precedent_sim),
        )

        factors = [
            f"criterion↔document similarity: {crit_doc_sim:.2f}",
            f"best precedent similarity: {best_precedent_sim:.2f}",
        ]
        if best_precedent:
            factors.append(
                f"top precedent (truncated): {best_precedent[:120]}"
            )

        reasoning = (
            f"Semantic score criterion↔document={crit_doc_sim:.2f}, "
            f"criterion↔best_precedent={best_precedent_sim:.2f}. "
            f"Verdict {verdict} with confidence {confidence:.2f}. "
            f"Model: {embedding_service.MODEL_VERSION}."
        )

        return (
            {
                "verdict": verdict,
                "factors": factors,
                "similarity_to_criterion": crit_doc_sim,
                "best_precedent_similarity": best_precedent_sim,
                "best_precedent": best_precedent,
            },
            confidence,
            reasoning,
        )

    # ── Handler: similarity assessment ───────────────────────────────────

    def _handle_similarity(self, context: dict) -> tuple[dict, float, str]:
        """Pairwise cosine similarity between two texts."""
        text_a = context.get("text_a") or context.get("criterion_a") or ""
        text_b = context.get("text_b") or context.get("criterion_b") or ""

        if not text_a or not text_b:
            return (
                {
                    "similarity_score": 0.0,
                    "criterion_a": text_a,
                    "criterion_b": text_b,
                    "matching_aspects": [],
                },
                0.0,
                "Missing text_a or text_b — returning zero similarity.",
            )

        score = embedding_service.similarity_score(text_a, text_b)
        # Confidence mirrors the similarity magnitude. A near-zero
        # similarity is still a *confident* "not similar", so we use |.|
        confidence = min(1.0, abs(score))

        reasoning = (
            f"Cosine similarity {score:.3f} computed with "
            f"{embedding_service.MODEL_VERSION}."
        )
        return (
            {
                "similarity_score": score,
                "criterion_a": text_a,
                "criterion_b": text_b,
                "matching_aspects": [],
            },
            confidence,
            reasoning,
        )

    # ── Handler: default ─────────────────────────────────────────────────

    def _handle_default(self, prompt_type: str) -> tuple[dict, float, str]:
        """Unknown prompt types route to officer review with zero confidence."""
        return (
            {},
            0.0,
            f"Unknown prompt_type '{prompt_type}' — routed to review.",
        )


# ─── Evidence helper passthrough (optional convenience) ──────────────────────
#
# Kept as module-level functions so tests / scripts can import them without
# spinning up an LLMStub instance. Mirrors the evidence_extractor API.

def extract_numeric_value(*args, **kwargs):  # pragma: no cover - trivial reexport
    return evidence_extractor.extract_numeric_value(*args, **kwargs)


def extract_company_name(*args, **kwargs):  # pragma: no cover - trivial reexport
    return evidence_extractor.extract_company_name(*args, **kwargs)


def extract_registration_number(*args, **kwargs):  # pragma: no cover - trivial reexport
    return evidence_extractor.extract_registration_number(*args, **kwargs)
