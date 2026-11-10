"""
db.py — SQLite persistence for benchmark runs and semantic cache.

Tables
------
1. runs:
   id, sample_id, mode, attempt, output, det_pass, semantic, critic,
   reliability, latency_ms, timestamp

2. cache:
   id, query, embedding (BLOB), response, reliability, retries, created_at
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Default database location in the project root
DEFAULT_DB_PATH = Path(
    os.environ.get(
        "AGENT_HARNESS_DB_PATH",
        Path(__file__).resolve().parent.parent.parent / "agent_harness.db",
    )
)


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a configured sqlite3 connection with Row factory."""
    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # Fast write-ahead logging
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """Create tables and indices if they do not exist."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT,
            mode TEXT,
            attempt INTEGER,
            output TEXT,
            det_pass INTEGER,
            semantic REAL,
            critic REAL,
            reliability REAL,
            latency_ms REAL,
            timestamp TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            embedding BLOB,
            response TEXT,
            reliability REAL,
            retries INTEGER,
            created_at TEXT
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_sample ON runs(sample_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(mode)")
        conn.commit()


def log_run(
    sample_id: str,
    mode: str,
    attempt: int,
    output: str | None,
    det_pass: int | bool,
    semantic: float,
    critic: float,
    reliability: float,
    latency_ms: float,
    timestamp: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Insert a single attempt record into the runs table."""
    path = db_path or DEFAULT_DB_PATH
    init_db(path)
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    det_val = 1 if det_pass else 0

    with get_connection(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (
                sample_id, mode, attempt, output, det_pass,
                semantic, critic, reliability, latency_ms, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(sample_id),
                str(mode),
                int(attempt),
                str(output) if output is not None else "",
                det_val,
                round(float(semantic), 4),
                round(float(critic), 4),
                round(float(reliability), 4),
                round(float(latency_ms), 2),
                ts,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0


def save_cache(
    query: str,
    embedding: bytes | np.ndarray,
    response: str,
    reliability: float,
    retries: int,
    created_at: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Insert a successful run record into the cache table."""
    path = db_path or DEFAULT_DB_PATH
    init_db(path)
    ts = created_at or datetime.now(timezone.utc).isoformat()

    if isinstance(embedding, np.ndarray):
        blob = embedding.astype(np.float32).tobytes()
    elif isinstance(embedding, (bytes, bytearray)):
        blob = bytes(embedding)
    else:
        raise TypeError(f"Unsupported embedding type: {type(embedding)}")

    with get_connection(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO cache (query, embedding, response, reliability, retries, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(query),
                blob,
                str(response),
                round(float(reliability), 4),
                int(retries),
                ts,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0


def get_all_cached(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    """Retrieve all cached query responses and embeddings."""
    path = db_path or DEFAULT_DB_PATH
    init_db(path)
    with get_connection(path) as conn:
        cur = conn.execute(
            "SELECT id, query, embedding, response, reliability, retries, created_at FROM cache"
        )
        return cur.fetchall()


def clear_cache(db_path: Path | str | None = None) -> int:
    """Delete all records from the cache table and reclaim space with VACUUM.

    Returns:
        The number of cached records that were removed.
    """
    path = db_path or DEFAULT_DB_PATH
    init_db(path)
    with get_connection(path) as conn:
        cur = conn.execute("SELECT count(*) FROM cache")
        count = cur.fetchone()[0]
        conn.execute("DELETE FROM cache")
        conn.commit()
    # VACUUM outside the transaction block
    with get_connection(path) as conn:
        conn.execute("VACUUM")
    return count

