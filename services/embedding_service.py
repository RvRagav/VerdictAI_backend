"""Semantic embedding service for VerdictAI.

Wraps a local sentence-transformers model (all-MiniLM-L6-v2, ~80MB)
to provide real semantic similarity for criterion disambiguation, CPM
precedent retrieval, and qualitative evaluation. The model is cached
to ~/.cache/huggingface/hub on first download and loaded lazily.

All vectors are L2-normalised at encode time so cosine similarity
reduces to a plain dot product. This is both faster and numerically
well-behaved.

Functions:
- get_model:          Lazy-load the underlying SentenceTransformer.
- encode:             Encode a list of strings into a (N, 384) matrix.
- cosine_similarity:  Similarity between two normalised vectors.
- similarity_score:   Similarity between two raw strings.
- rank_by_similarity: Top-k ranking of candidates against a query.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


# Model identifier. Small (~80MB), 384-dim, reasonable for short legal
# clauses and criterion descriptions. Pinned so the audit trail can
# record exactly which encoder produced a given similarity score.
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Public version string for audit logging / model provenance.
MODEL_VERSION = "sentence-transformers-all-MiniLM-L6-v2"

# Module-level singleton, populated on first call to get_model().
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer model on first call.

    Subsequent calls return the cached instance. Loading is deferred so
    that import of this module is cheap and unit tests that don't
    exercise semantic similarity don't pay the model-download cost.

    Returns:
        The loaded SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into L2-normalised 384-dim embeddings.

    Args:
        texts: Non-empty list of strings to encode.

    Returns:
        numpy array of shape (len(texts), 384) with dtype float32.
        Each row is L2-normalised so dot products equal cosine
        similarities in [-1.0, 1.0].
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors.

    Expects vectors produced by :func:`encode`, which are already
    L2-normalised — so this is just a dot product. Returns a float
    in [-1.0, 1.0].

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity as a Python float.
    """
    return float(np.dot(a, b))


def similarity_score(text_a: str, text_b: str) -> float:
    """Convenience wrapper: similarity between two raw strings.

    Encodes both strings and returns their cosine similarity. Suitable
    for low-volume pairwise comparisons; for batched comparisons use
    :func:`rank_by_similarity` instead.

    Args:
        text_a: First string.
        text_b: Second string.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either
        input is empty.
    """
    if not text_a or not text_b:
        return 0.0
    embeddings = encode([text_a, text_b])
    return cosine_similarity(embeddings[0], embeddings[1])


def rank_by_similarity(
    query: str,
    candidates: list[str],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Rank candidate strings by similarity to the query.

    Args:
        query: Reference string to rank candidates against.
        candidates: List of candidate strings.
        top_k: Maximum number of results to return.

    Returns:
        List of (index, similarity) tuples, sorted by similarity
        descending. Indices refer to the original candidates list.
        Empty candidates list returns []. An empty query returns
        tuples with similarity 0.0 preserving input order up to top_k.
    """
    if not candidates:
        return []
    if not query:
        return [(i, 0.0) for i in range(min(top_k, len(candidates)))]

    query_emb = encode([query])[0]
    cand_emb = encode(candidates)
    # Dot product because every vector is already unit-norm.
    sims = cand_emb @ query_emb

    k = min(top_k, len(candidates))
    # argsort on negated array gives descending order.
    top_indices = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top_indices]
