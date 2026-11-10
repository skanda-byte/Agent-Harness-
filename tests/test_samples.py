"""
tests/test_samples.py — Verify samples.json integrity, splits, categories, and reference checker validity.
"""

from __future__ import annotations

import json
from pathlib import Path
from agentic_harness.validators import run_checker

def test_samples_integrity():
    samples_file = Path(__file__).resolve().parent.parent / "src" / "agentic_harness" / "benchmark" / "samples.json"
    with samples_file.open(encoding="utf-8") as f:
        samples = json.load(f)

    assert len(samples) == 54, f"Expected 54 samples, got {len(samples)}"

    categories = {}
    splits = {}
    for s in samples:
        cat = s["category"]
        split = s["split"]
        categories[cat] = categories.get(cat, 0) + 1
        splits[split] = splits.get(split, 0) + 1

        # Check reference answer validity
        p, err = run_checker(s["checker"]["type"], s["reference_answer"], s["checker"].get("params", {}))
        assert p, f"Checker failed on reference answer for {s['id']} ({s['category']}): {err}"

    expected_cats = [
        "json_extraction",
        "arithmetic_word_problems",
        "format_constraints",
        "factual_qa",
        "instruction_following",
        "length_limited_summarization",
    ]
    for c in expected_cats:
        assert categories.get(c) == 9, f"Category {c} count is {categories.get(c)}, expected 9"

    assert splits.get("tune") == 32, f"Expected 32 tune samples, got {splits.get('tune')}"
    assert splits.get("test") == 22, f"Expected 22 test samples, got {splits.get('test')}"

    print("ALL 54 SAMPLES VALIDATED SUCCESSFULLY!")
    print(f"Categories: {categories}")
    print(f"Splits: {splits}")

if __name__ == "__main__":
    test_samples_integrity()

