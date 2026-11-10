"""
tests/test_dashboard.py — Unit tests for dashboard.py data loading, aggregation,
and metric consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard import (
    load_sample_metadata,
    load_results_summary,
    load_db_runs,
    get_consolidated_sample_data,
)


def test_load_sample_metadata():
    meta = load_sample_metadata()
    assert isinstance(meta, dict)
    assert len(meta) >= 54  # At least the 54 samples, up to 279 with challenges
    # Check sample structure
    any_id = next(iter(meta.keys()))
    assert "category" in meta[any_id]
    assert "prompt" in meta[any_id]


def test_load_results_summary():
    raw, metrics_df, samples_df = load_results_summary()
    assert raw is not None
    assert not metrics_df.empty
    assert "mode" in metrics_df.columns
    assert "success_rate" in metrics_df.columns
    assert not samples_df.empty
    assert "passed" in samples_df.columns
    assert "reliability" in samples_df.columns
    assert "retries_used" in samples_df.columns


def test_load_db_runs():
    runs_df, cache_df = load_db_runs()
    assert not runs_df.empty
    assert "sample_id" in runs_df.columns
    assert "reliability" in runs_df.columns
    assert "det_pass" in runs_df.columns
    assert isinstance(cache_df, pd.DataFrame)


def test_consolidated_sample_data():
    _, _, samples_df = load_results_summary()
    runs_df, _ = load_db_runs()

    # From results file
    df_res = get_consolidated_sample_data(samples_df, runs_df, "Results File (summary.json)")
    assert not df_res.empty
    assert "mode" in df_res.columns
    assert "passed" in df_res.columns

    # From SQLite database
    df_db = get_consolidated_sample_data(samples_df, runs_df, "SQLite DB (agent_harness.db)")
    assert not df_db.empty
    assert "mode" in df_db.columns
    assert "reliability" in df_db.columns


if __name__ == "__main__":
    test_load_sample_metadata()
    test_load_results_summary()
    test_load_db_runs()
    test_consolidated_sample_data()
    print("ALL DASHBOARD DATA & AGGREGATION TESTS PASSED!")

