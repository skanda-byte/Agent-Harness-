"""
tests/test_challenges.py — Validate challenges.json count, splits, schema, and reference answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from agentic_harness.validators import run_checker

def test_challenges_integrity():
    challenges_file = Path(__file__).resolve().parent.parent / "src" / "agentic_harness" / "benchmark" / "challenges.json"
    assert challenges_file.exists(), f"challenges.json does not exist at {challenges_file}"

    with challenges_file.open(encoding="utf-8") as f:
        challenges = json.load(f)

    assert len(challenges) >= 200, f"Expected at least 200 challenges, got {len(challenges)}"

    categories = {}
    splits = {}
    for c in challenges:
        cat = c["category"]
        split = c["split"]
        categories[cat] = categories.get(cat, 0) + 1
        splits[split] = splits.get(split, 0) + 1

        p, err = run_checker(c["checker"]["type"], c["reference_answer"], c["checker"].get("params", {}))
        assert p, f"Checker failed for {c['id']} ({c['category']}): {err}"

    tune_pct = splits["tune"] / len(challenges)
    assert 0.55 <= tune_pct <= 0.65, f"Expected ~60% tune split, got {tune_pct:.2f}"

    print(f"ALL {len(challenges)} CHALLENGES VALIDATED SUCCESSFULLY!")
    print(f"Categories: {categories}")
    print(f"Splits: {splits}")

if __name__ == "__main__":
    test_challenges_integrity()

