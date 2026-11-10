"""
semantic.py — Semantic similarity scoring using sentence-transformers.

Provides:
    semantic_score(output, reference) -> float

The score is the cosine similarity between the two sentence embeddings,
clipped to [0.0, 1.0].  The model is loaded lazily on first call and
cached in-process.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Load and cache the SentenceTransformer model (CPU-friendly, ~90 MB)."""
    from sentence_transformers import SentenceTransformer  # lazy import

    logger.info("Loading sentence-transformer model '%s' …", _MODEL_NAME)
    try:
        # Prefer local cached files first to avoid network timeouts/warnings
        model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
    except Exception:
        model = SentenceTransformer(_MODEL_NAME)
    logger.info("Model loaded.")
    return model


def semantic_score(output: str, reference: str) -> float:
    """Return cosine similarity between *output* and *reference*, clipped to [0, 1].

    Args:
        output:    The model's response text.
        reference: The reference / gold-standard answer.

    Returns:
        A float in [0.0, 1.0].  Returns 0.0 if either string is empty.
    """
    if not output or not reference:
        return 0.0

    model = _get_model()
    # encode returns a 2-D array of shape (n_sentences, embedding_dim)
    embeddings = model.encode([output, reference], convert_to_numpy=True)

    vec_a: np.ndarray = embeddings[0]
    vec_b: np.ndarray = embeddings[1]

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Clip: cosine similarity is theoretically in [-1, 1]; sentences rarely go
    # negative, but we guard against floating-point edge cases.
    return float(np.clip(cosine, 0.0, 1.0))


def embed_text(text: str) -> np.ndarray:
    """Return 1D float32 numpy array embedding for *text*."""
    if not text:
        return np.zeros(384, dtype=np.float32)
    model = _get_model()
    emb = model.encode(text, convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32)

