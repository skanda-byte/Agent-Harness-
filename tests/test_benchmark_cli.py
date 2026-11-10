"""
tests/test_benchmark_cli.py — Unit tests verifying dataset loading, suites, and split filtering.
"""

from __future__ import annotations

from agentic_harness.benchmark.run_bench import load_samples

def test_load_samples_benchmark():
    # Benchmark suite total
    all_bench = load_samples(suite="benchmark")
    assert len(all_bench) == 54, f"Expected 54 benchmark samples, got {len(all_bench)}"

    # Benchmark tune split
    bench_tune = load_samples(suite="benchmark", split="tune")
    assert len(bench_tune) == 32, f"Expected 32 tune samples, got {len(bench_tune)}"

    # Benchmark test split
    bench_test = load_samples(suite="benchmark", split="test")
    assert len(bench_test) == 22, f"Expected 22 test samples, got {len(bench_test)}"


def test_load_samples_challenge():
    # Challenge suite total
    all_chal = load_samples(suite="challenge")
    assert len(all_chal) >= 200, f"Expected >= 200 challenges, got {len(all_chal)}"

    # Challenge tune split
    chal_tune = load_samples(suite="challenge", split="tune")
    chal_test = load_samples(suite="challenge", split="test")

    assert len(chal_tune) + len(chal_test) == len(all_chal)
    tune_ratio = len(chal_tune) / len(all_chal)
    assert 0.55 <= tune_ratio <= 0.65, f"Expected ~60% tune split, got {tune_ratio:.2f}"

    print("BENCHMARK SUITE & SPLIT LOADING TESTS PASSED!")

if __name__ == "__main__":
    test_load_samples_benchmark()
    test_load_samples_challenge()

