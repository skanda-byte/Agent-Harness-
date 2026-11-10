"""
cache.py — Semantic caching layer backed by SQLite.

Provides:
    lookup(query, threshold=0.9) -> CacheResponse | None
"""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

import numpy as np

from agentic_harness.db import get_all_cached, save_cache, clear_cache
from agentic_harness.semantic import embed_text

logger = logging.getLogger(__name__)


class CacheResponse(str):
    """
    String subclass holding the cached response text plus metadata.

    Behaves as a regular string for equality and printing (e.g. `resp == "408"`),
    while also supporting attribute/dict access for `similarity`, `reliability`,
    and `retries`.
    """

    def __new__(
        cls,
        response: str,
        similarity: float,
        reliability: float,
        retries: int,
        query: str,
        cache_id: int,
    ):
        obj = super().__new__(cls, response)
        obj.response = response
        obj.similarity = similarity
        obj.reliability = reliability
        obj.retries = retries
        obj.query = query
        obj.cache_id = cache_id
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str) and hasattr(self, item):
            return getattr(self, item)
        return super().__getitem__(item)


def lookup(
    query: str,
    threshold: float = 0.9,
    db_path: Path | str | None = None,
) -> CacheResponse | None:
    """Embed *query*, compute cosine similarity against all cached embeddings,
    and return the cached response if similarity > threshold (default 0.9).

    Args:
        query:     The incoming user prompt / query.
        threshold: Minimum cosine similarity to count as a cache hit (> 0.9).
        db_path:   Optional SQLite DB path override.

    Returns:
        A CacheResponse string object if hit, else None.
    """
    if not query:
        return None

    cached_rows = get_all_cached(db_path=db_path)
    if not cached_rows:
        return None

    query_vec = embed_text(query)
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm == 0.0:
        return None

    best_sim = -1.0
    best_row = None

    for row in cached_rows:
        blob = row["embedding"]
        cached_vec = np.frombuffer(blob, dtype=np.float32)
        c_norm = float(np.linalg.norm(cached_vec))
        if c_norm == 0.0:
            continue

        sim = float(np.dot(query_vec, cached_vec) / (q_norm * c_norm))
        if sim > best_sim:
            best_sim = sim
            best_row = row

    if best_row is not None and best_sim > threshold:
        logger.info(
            "Cache HIT (sim=%.4f > %.2f) for query: %r (matched: %r)",
            best_sim,
            threshold,
            query[:50],
            best_row["query"][:50],
        )
        return CacheResponse(
            response=best_row["response"],
            similarity=round(best_sim, 4),
            reliability=best_row["reliability"],
            retries=best_row["retries"],
            query=best_row["query"],
            cache_id=best_row["id"],
        )

    return None

