"""
benchmark/run_bench.py — Run every sample in samples.json through the LLM
and report success metrics, latency, and cache hits.

Modes
-----
  --mode baseline   Single LLM call per sample, no retries.
  --mode harness    Agentic retry harness with semantic cache & multi-signal scoring.

Usage
-----
    python -m agentic_harness.benchmark.run_bench --mode baseline
    python -m agentic_harness.benchmark.run_bench --mode harness --max-retries 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is importable when the script is run directly
_HERE     = Path(__file__).resolve().parent   # …/benchmark/
_PKG_ROOT = _HERE.parent                      # …/agentic_harness/
_SRC      = _PKG_ROOT.parent                  # …/src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentic_harness.llm import get_client          # noqa: E402
from agentic_harness.validators import run_checker  # noqa: E402
from agentic_harness.harness import run as harness_run  # noqa: E402
from agentic_harness.db import log_run, init_db         # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMPLES_FILE    = _HERE / "samples.json"
CHALLENGES_FILE = _HERE / "challenges.json"
RESULTS_DIR     = _HERE / "results"


def load_samples(suite: str = "benchmark", split: str | None = None) -> list[dict]:
    """Load benchmark or challenge samples, optionally filtered by split (tune|test)."""
    target_file = CHALLENGES_FILE if suite == "challenge" else SAMPLES_FILE
    if not target_file.exists():
        raise FileNotFoundError(
            f"Suite file not found: {target_file}. "
            + ("Run `python -m agentic_harness.benchmark.make_challenges` to generate it." if suite == "challenge" else "")
        )

    with target_file.open(encoding="utf-8") as fh:
        samples = json.load(fh)

    if split is not None:
        filtered = [s for s in samples if s.get("split") == split]
        logger.info("Filtered %d/%d samples for split=%r", len(filtered), len(samples), split)
        return filtered

    return samples


def run_sample_baseline(sample: dict, temperature: float = 0.2) -> dict:
    """Single LLM call, no retries, logged to runs table."""
    sid      = sample["id"]
    category = sample["category"]
    prompt   = sample["prompt"]
    checker  = sample["checker"]

    gen_client = get_client("generator")
    logger.info("[%s] (%s) [baseline] calling generator model '%s' …", sid, category, gen_client.model)
    t0 = time.perf_counter()

    try:
        res = gen_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        elapsed = time.perf_counter() - t0
        latency_ms = round(elapsed * 1000, 2)
        if res.error:
            raise RuntimeError(f"{res.error}: {res.text}")
        response = res.text
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        latency_ms = round(elapsed * 1000, 2)
        logger.error("[%s] LLM call failed: %s", sid, exc)

        log_run(
            sample_id=sid,
            mode="baseline",
            attempt=1,
            output=None,
            det_pass=0,
            semantic=0.0,
            critic=0.0,
            reliability=0.0,
            latency_ms=latency_ms,
        )

        return {
            "id": sid,
            "category": category,
            "mode": "baseline",
            "passed": False,
            "error": str(exc),
            "final_output": None,
            "retries_used": 0,
            "cache_hit": False,
            "failed_attempt1": True,
            "recovered": False,
            "history": [],
            "latency_ms": latency_ms,
            "latency_s": round(elapsed, 3),
        }

    try:
        passed, error_msg = run_checker(
            checker["type"], response, checker.get("params", {})
        )
    except KeyError as exc:
        passed, error_msg = False, str(exc)

    det_score = 1.0 if passed else 0.0

    # Log to runs table
    log_run(
        sample_id=sid,
        mode="baseline",
        attempt=1,
        output=response,
        det_pass=passed,
        semantic=0.0,
        critic=0.0,
        reliability=det_score,
        latency_ms=latency_ms,
    )

    if passed:
        logger.info("[%s] ✓ PASSED (%.1fms)", sid, latency_ms)
    else:
        logger.warning("[%s] ✗ FAILED — %s (%.1fms)", sid, error_msg, latency_ms)

    return {
        "id": sid,
        "category": category,
        "mode": "baseline",
        "passed": passed,
        "reliability": det_score,
        "error_message": error_msg,
        "final_output": response,
        "retries_used": 0,
        "cache_hit": False,
        "failed_attempt1": not passed,
        "recovered": False,
        "history": [{
            "attempt": 1,
            "output": response,
            "passed": passed,
            "reliability": det_score,
            "error": error_msg,
            "latency_ms": latency_ms,
        }],
        "latency_ms": latency_ms,
        "latency_s": round(elapsed, 3),
    }


def run_sample_harness(
    sample: dict,
    max_retries: int,
    enable_cache: bool = True,
    temperature: float = 0.2,
) -> dict:
    """Delegate to harness.run() with caching and run logging."""
    sid      = sample["id"]
    category = sample["category"]
    checker  = sample["checker"]

    logger.info("[%s] (%s) [harness] running …", sid, category)

    result = harness_run(
        sample,
        max_retries=max_retries,
        mode="harness",
        enable_cache=enable_cache,
        temperature=temperature,
    )

    history         = result["history"]
    passed          = result["passed"]
    retries_used    = result["retries_used"]
    final_output    = result["final_output"]
    cache_hit       = result.get("cache_hit", False)
    total_latency_ms= result.get("latency_ms", 0.0)

    # Recovery: first attempt failed but final passed
    failed_attempt1 = len(history) > 0 and not history[0].get("passed", False)
    recovered = failed_attempt1 and passed

    last_error = ""
    if history:
        last_error = history[-1].get("det_error") or history[-1].get("error", "") or ""

    final_reliability = history[-1].get("reliability", 1.0 if passed else 0.0) if history else (1.0 if passed else 0.0)

    return {
        "id": sid,
        "category": category,
        "mode": "harness",
        "checker_type": checker["type"],
        "passed": passed,
        "reliability": final_reliability,
        "error_message": last_error,
        "final_output": final_output,
        "retries_used": retries_used,
        "cache_hit": cache_hit,
        "failed_attempt1": failed_attempt1,
        "recovered": recovered,
        "history": history,
        "latency_ms": total_latency_ms,
        "latency_s": round(total_latency_ms / 1000.0, 3),
    }


def print_report(
    results: list[dict],
    mode: str,
    suite: str = "benchmark",
    split: str | None = None,
) -> None:
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])

    failed_first   = [r for r in results if r["failed_attempt1"]]
    recovered      = [r for r in failed_first if r["recovered"]]
    n_failed_first = len(failed_first)
    n_recovered    = len(recovered)

    recovery_rate = (
        f"{n_recovered}/{n_failed_first}  "
        f"({100 * n_recovered / n_failed_first:.1f}%)"
        if n_failed_first > 0
        else "N/A (all passed on attempt 1)"
    )
    if mode == "baseline":
        recovery_rate = "N/A (baseline — no retries)"

    avg_retries = (
        sum(r["retries_used"] for r in results) / total if total > 0 else 0.0
    )

    cache_hits = sum(1 for r in results if r.get("cache_hit"))
    avg_latency = (
        sum(r.get("latency_ms", 0.0) for r in results) / total if total > 0 else 0.0
    )

    cat_totals: dict[str, int] = defaultdict(int)
    cat_passed: dict[str, int] = defaultdict(int)
    for r in results:
        cat = r["category"]
        cat_totals[cat] += 1
        if r["passed"]:
            cat_passed[cat] += 1

    W = 68
    print("\n" + "=" * W)
    print(f"  BENCHMARK RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Suite: {suite.upper()} | Split: {(split or 'all').upper()} | Mode: {mode.upper()}")
    print("=" * W)
    print(f"  {'Success rate':<28} {passed}/{total}  ({100*passed/total:.1f}%)")
    print(f"  {'Error recovery rate':<28} {recovery_rate}")
    print(f"  {'Avg retries / query':<28} {avg_retries:.2f}")
    print(f"  {'Cache hits':<28} {cache_hits}/{total}")
    print(f"  {'Avg query latency':<28} {avg_latency:.1f} ms")
    print()

    print("  Per-category breakdown:")
    for cat in sorted(cat_totals):
        t = cat_totals[cat]
        p = cat_passed[cat]
        bar = ("█" * p) + ("░" * (t - p))
        print(f"    {cat:<22} {p}/{t}  [{bar}]")
    print()

    print("  Sample-level detail:")
    hdr = f"  {'ID':<6} {'Cat':<18} {'P?':<3} {'R':<2} {'Hit?':<5} {'Latency':<9} Note"
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in results:
        icon  = "✓" if r["passed"] else "✗"
        hit   = "yes" if r.get("cache_hit") else "no"
        lat   = f"{r.get('latency_ms', 0):.0f}ms"
        note  = r.get("error_message") or r.get("error") or ""
        if len(note) > 22:
            note = note[:19] + "…"
        print(
            f"  {r['id']:<6} {r['category']:<18} {icon:<3} "
            f"{r['retries_used']:<2} {hit:<5} {lat:<9} {note}"
        )

    print("=" * W + "\n")


def save_results(
    results: list[dict],
    mode: str,
    suite: str = "benchmark",
    split: str | None = None,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    split_tag = split or "all"
    path = RESULTS_DIR / f"bench_{suite}_{split_tag}_{mode}_{ts}.json"

    total  = len(results)
    passed = sum(1 for r in results if r["passed"])

    failed_first = [r for r in results if r["failed_attempt1"]]
    recovered    = [r for r in failed_first if r["recovered"]]

    payload = {
        "run_at":          ts,
        "suite":           suite,
        "split":           split,
        "mode":            mode,
        "total":           total,
        "passed":          passed,
        "success_rate":    round(passed / total, 4) if total else 0,
        "failed_attempt1": len(failed_first),
        "recovered":       len(recovered),
        "recovery_rate":   (
            round(len(recovered) / len(failed_first), 4)
            if failed_first else None
        ),
        "avg_retries":     (
            round(sum(r["retries_used"] for r in results) / total, 3)
            if total else 0
        ),
        "cache_hits":      sum(1 for r in results if r.get("cache_hit")),
        "avg_latency_ms":  (
            round(sum(r.get("latency_ms", 0.0) for r in results) / total, 2)
            if total else 0.0
        ),
        "results": results,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Results saved → %s", path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Harness benchmark runner with caching & run logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=["benchmark", "challenge"],
        default="benchmark",
        help="Which dataset suite to run: 'benchmark' (samples.json) or 'challenge' (challenges.json).",
    )
    parser.add_argument(
        "--split",
        choices=["tune", "test"],
        default=None,
        help="Filter samples by split ('tune' or 'test'). Default is all splits.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "harness"],
        default="baseline",
        help="baseline: single LLM call. harness: agentic retry & evaluation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        dest="max_retries",
        help="(harness mode) maximum correction turns per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for LLM generation (default: 0.2).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable checking and populating the semantic cache.",
    )
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    mode    = args.mode
    suite   = args.suite
    split   = args.split
    temp    = args.temperature
    samples = load_samples(suite=suite, split=split)

    init_db()

    logger.info(
        "Loaded %d samples from suite=%r (split=%r). Running in [%s] mode%s (temp=%.2f, cache=%s).",
        len(samples),
        suite,
        split or "all",
        mode,
        f" (max_retries={args.max_retries})" if mode == "harness" else "",
        temp,
        not args.no_cache,
    )

    results: list[dict] = []
    for sample in samples:
        if mode == "baseline":
            result = run_sample_baseline(sample, temperature=temp)
        else:
            result = run_sample_harness(
                sample,
                max_retries=args.max_retries,
                enable_cache=not args.no_cache,
                temperature=temp,
            )
        results.append(result)
        time.sleep(3.0)

    print_report(results, mode=mode, suite=suite, split=split)
    out_path = save_results(results, mode=mode, suite=suite, split=split)
    print(f"Full results written to: {out_path}\n")


if __name__ == "__main__":
    main()
