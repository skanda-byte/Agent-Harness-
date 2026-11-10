"""
tests/test_stalled_loop.py — Unit test for stalled loop detection on consecutive identical output & error.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import agentic_harness.harness as h_mod


def test_stalled_loop_detection():
    # Simulate an LLM that returns the EXACT SAME bad output twice in a row
    call_count = 0

    def mock_gen(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        return "I will never remove my comma, ever."

    h_mod.generate = mock_gen
    h_mod.critique = lambda q, a: {"score": 0.5, "issues": ["Has commas."], "suggestion": "Remove commas"}
    h_mod.semantic_score = lambda a, b: 0.5

    # Constraint: no commas
    constraints = [{"type": "no_commas"}]

    # We give max_retries=5. Without stalled loop detection, it would make 6 calls (1 initial + 5 retries).
    # With stalled loop detection, on attempt 2 it sees identical output + identical error,
    # and halts immediately! So call_count should be 2.
    res = h_mod.answer(
        query="Write without commas",
        constraints=constraints,
        max_retries=5,
        enable_cache=False,
    )

    assert call_count == 2, f"Expected exactly 2 calls before halting, got {call_count}"
    assert res["passed"] is False
    assert res.get("failure_reason") == "possible checker error"
    assert res.get("error_message") == "possible checker error"
    assert res["answer"] == "I will never remove my comma, ever."
    print("STALLED LOOP DETECTION TEST PASSED!")


if __name__ == "__main__":
    test_stalled_loop_detection()
