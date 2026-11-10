"""
tests/test_evaluate.py — Unit test for evaluate.py metrics calculation and table formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate import compute_metrics, format_console_table, format_markdown_table, set_reproducibility


def test_reproducibility():
    set_reproducibility(123)


def test_compute_metrics_baseline():
    mock_results = [
        {"id": "s1", "passed": True, "reliability": 1.0, "failed_attempt1": False, "recovered": False, "retries_used": 0, "latency_ms": 100.0},
        {"id": "s2", "passed": False, "reliability": 0.0, "failed_attempt1": True, "recovered": False, "retries_used": 0, "latency_ms": 200.0},
    ]
    m = compute_metrics("benchmark", "test", "baseline", mock_results)
    assert m["total"] == 2
    assert m["passed"] == 1
    assert m["success_rate"] == 0.5
    assert m["avg_reliability"] == 0.5
    assert m["error_recovery_rate"] is None
    assert m["avg_retries"] == 0.0
    assert m["avg_cold_latency_ms"] == 150.0
    assert m["avg_cached_latency_ms"] is None


def test_compute_metrics_harness():
    mock_results = [
        {"id": "s1", "passed": True, "reliability": 0.95, "failed_attempt1": False, "recovered": False, "retries_used": 0, "latency_ms": 300.0},
        {"id": "s2", "passed": True, "reliability": 0.88, "failed_attempt1": True, "recovered": True, "retries_used": 1, "latency_ms": 600.0},
        {"id": "s3", "passed": False, "reliability": 0.40, "failed_attempt1": True, "recovered": False, "retries_used": 2, "latency_ms": 900.0},
    ]
    cached_latencies = [4.2, 5.1]
    m = compute_metrics("benchmark", "test", "harness", mock_results, cached_latencies)
    assert m["total"] == 3
    assert m["passed"] == 2
    assert round(m["success_rate"], 2) == 0.67
    assert m["failed_attempt1"] == 2
    assert m["recovered"] == 1
    assert m["error_recovery_rate"] == 0.5
    assert round(m["avg_retries"], 2) == 1.0
    assert round(m["avg_cold_latency_ms"], 1) == 600.0
    assert round(m["avg_cached_latency_ms"], 1) == 4.65 or round(m["avg_cached_latency_ms"], 2) == 4.65


def test_table_formatting():
    mock_results = [
        {"id": "s1", "passed": True, "reliability": 1.0, "failed_attempt1": False, "recovered": False, "retries_used": 0, "latency_ms": 100.0},
    ]
    m_base = compute_metrics("benchmark", "test", "baseline", mock_results)
    m_harn = compute_metrics("benchmark", "test", "harness", mock_results, [3.5])

    console_tbl = format_console_table([m_base, m_harn])
    assert "benchmark" in console_tbl
    assert "baseline" in console_tbl
    assert "harness" in console_tbl

    md_tbl = format_markdown_table([m_base, m_harn])
    assert "| **benchmark** |" in md_tbl
    assert "| `baseline` |" in md_tbl
    assert "3.5 ms" in md_tbl

    print("ALL EVALUATE UNIT TESTS PASSED!")


if __name__ == "__main__":
    test_reproducibility()
    test_compute_metrics_baseline()
    test_compute_metrics_harness()
    test_table_formatting()
