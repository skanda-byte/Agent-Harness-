"""
chat.py — Interactive CLI for the Agentic Harness.

Usage:
    python chat.py
    python chat.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentic_harness.harness import answer


def print_verbose_history(history: list[dict]):
    print("\n" + "=" * 60)
    print("  VERBOSE ATTEMPT TRACE")
    print("=" * 60)
    for h in history:
        att = h.get("attempt", 1)
        det = h.get("det_score")
        det_str = f"{det:.2f}" if det is not None else "N/A (skipped)"
        sem = h.get("sem_score", 0.0)
        crit = h.get("crit_score", 0.0)
        rel = h.get("reliability", 0.0)
        status = "ACCEPTED ✓" if h.get("accepted") else "REJECTED ✗"
        lat = h.get("latency_ms", 0.0)

        print(f"\n[Attempt {att}] -> {status}  ({lat:.1f}ms)")
        print(f"  • Sub-scores    : det={det_str}  semantic={sem:.2f}  critic={crit:.2f}  reliability={rel:.3f}")

        if h.get("det_error"):
            print(f"  • Validator err : {h['det_error']}")

        critic_info = h.get("critic")
        if critic_info and isinstance(critic_info, dict):
            issues = critic_info.get("issues", [])
            if issues:
                print(f"  • Critic issues : {', '.join(issues)}")
            sug = critic_info.get("suggestion")
            if sug:
                print(f"  • Suggestion    : {sug}")

        feedback = h.get("feedback")
        if feedback:
            print("  • Feedback sent to model:")
            print("    " + "-" * 50)
            for line in feedback.strip().splitlines():
                print(f"    | {line}")
            print("    " + "-" * 50)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Harness Live Chat CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print every attempt, its sub-scores, and the feedback sent to the model.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable semantic cache lookup and storage.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Agentic Harness Live Chat")
    print(f"  Mode: Interactive | Verbose: {args.verbose} | Cache: {not args.no_cache}")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 60)

    while True:
        try:
            print()
            query = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("\nThinking and evaluating …")

        try:
            result = answer(
                query=query,
                verbose=args.verbose,
                enable_cache=not args.no_cache,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}")
            continue

        if args.verbose:
            print_verbose_history(result.get("history", []))

        final_answer = result.get("answer") or "(No answer produced)"
        reliability  = result.get("reliability", 0.0)
        retries      = result.get("retries_used", 0)
        from_cache   = result.get("from_cache", False)
        lat_ms       = result.get("latency_ms", 0.0)
        constraints  = result.get("constraints", [])

        print("\n" + "-" * 60)
        print("Answer:")
        print(final_answer.strip())
        print("-" * 60)
        print(f"Reliability : {reliability:.3f}")
        print(f"Retries     : {retries}")
        print(f"From Cache  : {from_cache}")
        print(f"Latency     : {lat_ms:.1f} ms")
        if constraints and args.verbose:
            print(f"Active checks: {constraints}")
        print("-" * 60)


if __name__ == "__main__":
    main()

