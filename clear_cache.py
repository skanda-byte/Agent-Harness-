"""
clear_cache.py — Quick utility script to wipe the semantic cache in agent_harness.db.

Usage:
    python clear_cache.py        # Wipes ONLY the cache table (preserves runs telemetry)
    python clear_cache.py --all  # Wipes both cache and runs tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import sqlite3
from agentic_harness.db import DEFAULT_DB_PATH, get_connection, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe semantic cache from agent_harness.db")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also wipe the runs history table (default: False, only wipes cache)",
    )
    args = parser.parse_args()

    db_path = DEFAULT_DB_PATH
    if not db_path.exists():
        print(f"Database {db_path} does not exist. Nothing to clear.")
        return

    init_db(db_path)

    with get_connection(db_path) as conn:
        cache_count = conn.execute("SELECT count(*) FROM cache").fetchone()[0]
        conn.execute("DELETE FROM cache")
        conn.commit()

        runs_wiped = 0
        if args.all:
            runs_wiped = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
            conn.execute("DELETE FROM runs")
            conn.commit()

    # Reclaim space
    with get_connection(db_path) as conn:
        conn.execute("VACUUM")

    print(f"✓ Successfully wiped {cache_count} records from 'cache' table.")
    if args.all:
        print(f"✓ Also wiped {runs_wiped} records from 'runs' table.")
    else:
        with get_connection(db_path) as conn:
            remaining_runs = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
        print(f"  ('runs' table preserved with {remaining_runs} telemetry records.)")


if __name__ == "__main__":
    main()

