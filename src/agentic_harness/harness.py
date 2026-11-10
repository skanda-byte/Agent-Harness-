"""
harness.py — Agentic retry harness with multi-signal reliability scoring,
SQLite runs logging, semantic caching, and live interactive answer() flow.

Flow for answer(query, constraints=None)
----------------------------------------
  Step 1: If constraints is None, call constraint_extractor.extract_constraints(query).
  Step 2: Build and evaluate validators from checks. If empty, skip deterministic layer.
  Step 3: Run retry loop. Compute semantic score against query (relevance).
          Weights: 0.5 sem + 0.5 crit if no deterministic checks; otherwise 0.4/0.3/0.3.
  Step 4: Check semantic cache first (skip calls on hit); cache successful answers (reliability >= 0.80).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentic_harness.llm import get_client, LLMClient, LLMResult
from agentic_harness.validators import run_checker
from agentic_harness.semantic import semantic_score, embed_text
from agentic_harness.critic import critique
from agentic_harness.db import log_run, save_cache
from agentic_harness.cache import lookup
from agentic_harness.constraint_extractor import extract_constraints

logger = logging.getLogger(__name__)


def generate(messages: list[dict[str, str]], **kwargs: Any) -> str:
    """Invoke the configured generator LLM client and return the generated text."""
    client = get_client("generator")
    temp = kwargs.get("temperature")
    tokens = kwargs.get("max_tokens", kwargs.get("max_completion_tokens"))
    res = client.generate(messages, temperature=temp, max_tokens=tokens)
    if res.error:
        raise RuntimeError(f"{res.error}: {res.text}")
    return res.text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEMPERATURE       = 0.2
ACCEPT_THRESHOLD   = 0.8          # reliability score required to accept

_W_DET  = 0.4                     # weight: deterministic validator
_W_SEM  = 0.3                     # weight: semantic similarity
_W_CRIT = 0.3                     # weight: LLM critic

_DET_FAIL_CAP = 0.6               # max reliability when deterministic check fails

# ---------------------------------------------------------------------------
# Direct feedback builder for generator
# ---------------------------------------------------------------------------

def build_retry_message(
    validator_errors: list[str],
    critic_issues: list[str],
    is_identical: bool = False,
) -> str:
    """Build short, direct feedback for the generator (1.5B friendly).

    Includes at most 3 errors, and includes critic issues only if there are no validator errors.
    If the answer is identical to the previous attempt, adds:
    'Your last answer was identical. Change it.'
    """
    clean_val_errs = [e.strip() for e in validator_errors if e and e.strip()]
    clean_crit_issues = [c.strip() for c in critic_issues if c and c.strip()]

    if clean_val_errs:
        selected_errors = clean_val_errs[:3]
    elif clean_crit_issues:
        selected_errors = clean_crit_issues[:3]
    else:
        selected_errors = ["Output did not satisfy required constraints."]

    rules_text = "\n".join(f"- {err}" for err in selected_errors)

    parts = ["Your previous answer broke these rules:", rules_text]
    if is_identical:
        parts.append("Your last answer was identical. Change it.")
    parts.append("Rewrite the answer fixing only these problems. Keep the same content and follow the original request exactly. Output only the answer.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# combine()
# ---------------------------------------------------------------------------

def combine(det: float | None, sem: float, crit: float) -> float:
    """Compute the reliability score from sub-scores with relevance rescaling.

    Args:
        det:  Deterministic validator score (1.0 or 0.0), or None if skipped.
        sem:  Semantic similarity in [0, 1].
        crit: LLM critic score in [0, 1].

    Returns:
        Reliability score in [0, 1].
        - relevance = min(1.0, sem / 0.5)
        - If det is None (no deterministic checks):
            reliability = 0.8 * crit + 0.2 * relevance
        - If det is provided:
            reliability = 0.4 * det + 0.3 * relevance + 0.3 * crit (capped at 0.6 if det == 0)
        - Off-topic detector: if sem < 0.15, cap reliability at 0.6.
    """
    relevance = min(1.0, max(0.0, sem) / 0.5)

    if det is None:
        raw = 0.8 * crit + 0.2 * relevance
    else:
        raw = _W_DET * det + _W_SEM * relevance + _W_CRIT * crit
        if det == 0.0:
            raw = min(raw, _DET_FAIL_CAP)

    # Off-topic detector: if similarity is below 0.15, cap reliability at 0.6
    if sem < 0.15:
        raw = min(raw, 0.6)

    return round(float(raw), 4)


# ---------------------------------------------------------------------------
# Constraint normalization & evaluation
# ---------------------------------------------------------------------------

def normalize_check(check: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Normalize a check dict to (checker_type, params) for validators.py."""
    raw_type = check.get("type", "").lower()

    if raw_type in ("sentence_count", "exact_sentence_count"):
        n = int(check.get("n", check.get("expected_count", check.get("count", 0))))
        return "sentence_count", {"n": n}

    elif raw_type in ("words_per_sentence", "sentence_word_count"):
        min_words = int(check.get("min", check.get("min_words", 0)))
        max_words = int(check.get("max", check.get("max_words", 999999)))
        return "words_per_sentence", {"min": min_words, "max": max_words}

    elif raw_type in ("words_per_bullet", "bullet_word_count"):
        params: dict[str, Any] = {}
        if "n" in check:
            params["n"] = int(check["n"])
        if "min" in check:
            params["min"] = int(check["min"])
        if "max" in check:
            params["max"] = int(check["max"])
        return "words_per_bullet", params

    elif raw_type in ("distinct_first_letters", "distinct_letters"):
        return "distinct_first_letters", {}

    elif raw_type in ("contains_phrase", "phrase"):
        return "contains_phrase", {"phrase": str(check.get("phrase", ""))}

    elif raw_type in ("banned_letters", "banned_chars"):
        return "banned_letters", {"letters": check.get("letters", [])}

    elif raw_type in ("no_commas", "no_comma"):
        return "no_commas", {}

    elif raw_type in ("total_word_count", "word_count", "word_count_range"):
        min_words = int(check.get("min", check.get("min_words", 0)))
        max_words = int(check.get("max", check.get("max_words", 999999)))
        return "total_word_count", {"min": min_words, "max": max_words}

    elif raw_type in ("bullet_count", "exact_bullet_count"):
        expected_count = int(check.get("n", check.get("expected_count", 0)))
        bullet_prefix = str(check.get("bullet_prefix", "-"))
        return "bullet_count", {"n": expected_count, "expected_count": expected_count, "bullet_prefix": bullet_prefix}

    elif raw_type in ("json_keys", "valid_json_with_keys"):
        required_keys = check.get("keys", check.get("required_keys", []))
        if isinstance(required_keys, str):
            required_keys = [required_keys]
        return "json_keys", {"required_keys": list(required_keys), "keys": list(required_keys)}

    elif raw_type in ("numeric_match", "number"):
        expected = float(check.get("expected", check.get("value", 0)))
        tolerance = float(check.get("tolerance", 0))
        return "numeric_match", {"expected": expected, "tolerance": tolerance}

    elif raw_type in ("regex_match", "regex", "pattern"):
        pattern = str(check.get("pattern", ".*"))
        return "regex", {"pattern": pattern}

    return raw_type, {k: v for k, v in check.items() if k != "type"}


def evaluate_checks(
    checks: list[dict[str, Any]],
    response: str,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Run all checks against response.

    Returns:
        (all_passed: bool, error_message: str, check_details: list[dict])
    Each check_detail has: {"type": c_type, "params": params, "passed": bool, "error": str}
    """
    if not checks:
        return True, "", []

    all_passed = True
    errors: list[str] = []
    details: list[dict[str, Any]] = []

    for check in checks:
        c_type, params = normalize_check(check)
        try:
            passed, err = run_checker(c_type, response, params)
            if not passed:
                all_passed = False
                errors.append(f"[{c_type}] {err}")
            details.append({"type": c_type, "params": params, "passed": passed, "error": err})
        except KeyError as exc:
            all_passed = False
            err_msg = f"Unknown checker type {c_type}: {exc}"
            errors.append(err_msg)
            details.append({"type": c_type, "params": params, "passed": False, "error": err_msg})

    return all_passed, "\n".join(errors), details


# ---------------------------------------------------------------------------
# run() — Benchmark runner function
# ---------------------------------------------------------------------------

def run(
    sample: dict[str, Any],
    max_retries: int = 3,
    mode: str = "harness",
    enable_cache: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run *sample* through the agentic correction loop with semantic caching
    and SQLite run logging.
    """
    sid            = sample.get("id", "?")
    prompt         = sample["prompt"]
    reference      = sample.get("reference_answer", "")
    checker        = sample["checker"]
    checker_type   = checker["type"]
    checker_params = checker.get("params", {})

    t_start = time.perf_counter()

    # ---------------------------------------------------------------------- #
    # 0. Check Semantic Cache
    # ---------------------------------------------------------------------- #
    if enable_cache:
        cached = lookup(prompt)
        if cached is not None:
            hit_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            logger.info(
                "[%s] Cache HIT (similarity=%.4f, latency=%.2fms) — skipping API calls",
                sid,
                cached.similarity,
                hit_latency_ms,
            )

            log_run(
                sample_id=sid,
                mode=f"{mode}_cache_hit",
                attempt=1,
                output=str(cached),
                det_pass=1,
                semantic=cached.similarity,
                critic=1.0,
                reliability=cached.reliability,
                latency_ms=hit_latency_ms,
            )

            return {
                "final_output": str(cached),
                "passed":       True,
                "retries_used": 0,
                "cache_hit":    True,
                "latency_ms":   hit_latency_ms,
                "history": [{
                    "attempt":     1,
                    "output":      str(cached),
                    "det_score":   1.0,
                    "sem_score":   cached.similarity,
                    "crit_score":  1.0,
                    "reliability": cached.reliability,
                    "accepted":    True,
                    "passed":      True,
                    "det_error":   "",
                    "error":       "",
                    "cache_hit":   True,
                    "latency_ms":  hit_latency_ms,
                    "critic":      {"score": 1.0, "issues": [], "suggestion": "Served from cache"},
                }],
            }

    miss_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info("[%s] Cache MISS (%.2fms) — executing evaluation loop", sid, miss_latency_ms)

    # ---------------------------------------------------------------------- #
    # Standard Evaluation Loop
    # ---------------------------------------------------------------------- #
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

    history:      list[dict[str, Any]] = []
    final_output: str | None           = None
    passed        = False
    retries_used  = 0

    for attempt in range(1, max_retries + 2):
        is_retry = attempt > 1
        label = (
            f"attempt {attempt}"
            if not is_retry
            else f"retry {attempt - 1}/{max_retries}"
        )
        gen_client = get_client("generator")
        logger.info(
            "[%s] %s — calling generator model '%s' (provider=%s) …",
            sid,
            label,
            gen_client.model,
            gen_client.provider,
        )

        t_att_start = time.perf_counter()
        gen_temp = round(float(kwargs.get("temperature", _TEMPERATURE)) + 0.1 * (attempt - 1), 2)

        try:
            output = generate(
                messages=messages,
                temperature=gen_temp,
                max_retries=1,
            )
        except Exception as exc:  # noqa: BLE001
            att_latency_ms = round((time.perf_counter() - t_att_start) * 1000, 2)
            logger.error("[%s] LLM error on %s: %s", sid, label, exc)

            log_run(
                sample_id=sid,
                mode=mode,
                attempt=attempt,
                output=None,
                det_pass=0,
                semantic=0.0,
                critic=0.0,
                reliability=0.0,
                latency_ms=att_latency_ms,
            )

            history.append({
                "attempt":     attempt,
                "output":      None,
                "det_score":   0.0,
                "sem_score":   0.0,
                "crit_score":  0.0,
                "reliability": 0.0,
                "accepted":    False,
                "passed":      False,
                "det_error":   f"LLM call failed: {exc}",
                "error":       f"LLM call failed: {exc}",
                "cache_hit":   False,
                "latency_ms":  att_latency_ms,
                "critic":      None,
            })
            break

        try:
            det_passed, det_error = run_checker(checker_type, output, checker_params)
        except KeyError as exc:
            det_passed, det_error = False, str(exc)

        det_score = 1.0 if det_passed else 0.0
        sem = semantic_score(output, reference)
        critic_result = critique(prompt, output)
        crit_score    = critic_result["score"]
        critic_issues = [issue for issue in critic_result.get("issues", []) if issue and "parse error" not in issue.lower() and "no json" not in issue.lower()]
        reliability   = combine(det_score, sem, crit_score)
        att_latency_ms = round((time.perf_counter() - t_att_start) * 1000, 2)

        # Log individual check pass/fail and three sub-scores
        logger.info("[%s] %s check [%s] -> %s", sid, label, checker_type, "PASS" if det_passed else f"FAIL: {det_error}")
        logger.info(
            "[%s] %s scores → det=%.2f  sem=%.2f  crit=%.2f  reliability=%.3f  (%.1fms)",
            sid, label, det_score, sem, crit_score, reliability, att_latency_ms,
        )

        has_validator_error = bool(det_error and det_error.strip() and det_score == 0.0)
        has_critic_issues = len(critic_issues) > 0
        accepted     = (reliability >= ACCEPT_THRESHOLD) or (not has_validator_error and not has_critic_issues)
        final_output = output

        prev_output = history[-1].get("output") if history else None
        prev_error = history[-1].get("det_error") if history else None
        is_identical = bool(is_retry and prev_output is not None and output.strip() == str(prev_output).strip())

        # Stalled loop detection on consecutive identical output & error
        if is_retry and history:
            if output == prev_output and det_error == prev_error and not accepted:
                logger.warning(
                    "[%s] Stalled loop detected on %s: identical output and validator error. "
                    "Stopping retries and marking as 'possible checker error'.",
                    sid, label,
                )
                history.append({
                    "attempt":     attempt,
                    "output":      output,
                    "det_score":   det_score,
                    "sem_score":   round(sem, 4),
                    "crit_score":  round(crit_score, 4),
                    "reliability": reliability,
                    "accepted":    False,
                    "passed":      False,
                    "det_error":   "possible checker error",
                    "error":       "possible checker error",
                    "cache_hit":   False,
                    "latency_ms":  att_latency_ms,
                    "critic":      critic_result,
                })
                best = max(history, key=lambda h: h.get("reliability", 0.0))
                final_output = best.get("output")
                total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                return {
                    "final_output":   final_output,
                    "passed":         False,
                    "retries_used":   retries_used,
                    "cache_hit":      False,
                    "latency_ms":     total_latency_ms,
                    "history":        history,
                    "failure_reason": "possible checker error",
                    "error_message":  "possible checker error",
                }

        log_run(
            sample_id=sid,
            mode=mode,
            attempt=attempt,
            output=output,
            det_pass=det_score == 1.0,
            semantic=sem,
            critic=crit_score,
            reliability=reliability,
            latency_ms=att_latency_ms,
        )

        history.append({
            "attempt":     attempt,
            "output":      output,
            "det_score":   det_score,
            "sem_score":   round(sem, 4),
            "crit_score":  round(crit_score, 4),
            "reliability": reliability,
            "accepted":    accepted,
            "passed":      accepted,
            "det_error":   det_error,
            "error":       det_error,
            "cache_hit":   False,
            "latency_ms":  att_latency_ms,
            "critic":      critic_result,
        })

        if accepted:
            passed = True
            logger.info("[%s] ✓ ACCEPTED on %s (reliability=%.3f)", sid, label, reliability)
            break

        logger.warning(
            "[%s] ✗ REJECTED on %s (reliability=%.3f < %.2f)",
            sid, label, reliability, ACCEPT_THRESHOLD,
        )

        if attempt <= max_retries and (has_validator_error or has_critic_issues):
            retries_used += 1
            val_errors = [det_error] if has_validator_error else []
            correction_msg = build_retry_message(
                validator_errors=val_errors,
                critic_issues=critic_issues,
                is_identical=is_identical,
            )
            # Keep only the original request, the last answer, and this message in context
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
                {"role": "user", "content": correction_msg},
            ]
            time.sleep(0.4)
            time.sleep(0.4)
        else:
            # No specific issues to retry with
            passed = True
            logger.info("[%s] No validator error or critic issues to fix; accepting without retry.", sid)
            break

    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    final_rel = history[-1]["reliability"] if history else 0.0
    if enable_cache and final_output is not None and final_rel >= ACCEPT_THRESHOLD:
        try:
            save_cache(
                query=prompt,
                embedding=embed_text(prompt),
                response=final_output,
                reliability=final_rel,
                retries=retries_used,
            )
            logger.info("[%s] Stored response in cache (reliability=%.3f)", sid, final_rel)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Failed to save to cache: %s", sid, exc)

    return {
        "final_output": final_output,
        "passed":       passed,
        "retries_used": retries_used,
        "cache_hit":    False,
        "latency_ms":   total_latency_ms,
        "history":      history,
    }


# ---------------------------------------------------------------------------
# answer() — Live Entry Point for arbitrary queries without reference answers
# ---------------------------------------------------------------------------

def answer(
    query: str,
    constraints: list[dict[str, Any]] | None = None,
    max_retries: int = 3,
    enable_cache: bool = True,
    sample_id: str = "live_query",
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate an answer for *query* using agentic evaluation without requiring
    a reference answer.

    Steps:
      1. If constraints is None, automatically extract constraints with LLM.
      2. Build & evaluate validators. If empty, skip deterministic layer.
      3. Evaluate: compute semantic relevance between answer and query.
         Weights: 0.5 sem + 0.5 crit (if no det checks); otherwise 0.4/0.3/0.3.
      4. Check cache first and cache successful results.

    Returns:
        dict with keys: ``answer``, ``reliability``, ``retries_used``,
        ``from_cache``, ``cache_hit``, ``latency_ms``, ``constraints``, ``history``.
    """
    t_start = time.perf_counter()

    # Step 4: Check Semantic Cache first
    if enable_cache:
        cached = lookup(query)
        if cached is not None:
            hit_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            logger.info(
                "Cache HIT (sim=%.4f, %.2fms) for query: %r",
                cached.similarity, hit_latency_ms, query[:50],
            )

            log_run(
                sample_id=sample_id,
                mode="answer_cache_hit",
                attempt=1,
                output=str(cached),
                det_pass=1,
                semantic=cached.similarity,
                critic=1.0,
                reliability=cached.reliability,
                latency_ms=hit_latency_ms,
            )

            return {
                "answer":       str(cached),
                "reliability":  cached.reliability,
                "retries_used": 0,
                "from_cache":   True,
                "cache_hit":    True,
                "latency_ms":   hit_latency_ms,
                "constraints":  [],
                "history": [{
                    "attempt":     1,
                    "output":      str(cached),
                    "det_score":   None,
                    "sem_score":   cached.similarity,
                    "crit_score":  1.0,
                    "reliability": cached.reliability,
                    "accepted":    True,
                    "det_error":   "",
                    "feedback":    None,
                    "from_cache":  True,
                    "latency_ms":  hit_latency_ms,
                }],
            }

    # Step 1: Constraint extraction
    active_constraints: list[dict[str, Any]]
    if constraints is None:
        active_constraints = extract_constraints(query)
    else:
        active_constraints = list(constraints)

    has_det = len(active_constraints) > 0
    logger.info("[answer] Extracted checks: %s", active_constraints)

    messages: list[dict[str, str]] = [{"role": "user", "content": query}]

    history:      list[dict[str, Any]] = []
    final_output: str | None           = None
    retries_used  = 0
    passed        = False

    base_temp = float(kwargs.get("temperature", _TEMPERATURE))

    # Step 3: Retry Loop
    for attempt in range(1, max_retries + 2):
        t_att_start = time.perf_counter()
        is_retry = attempt > 1
        label = f"attempt {attempt}" if not is_retry else f"retry {attempt - 1}/{max_retries}"
        gen_client = get_client("generator")
        logger.info(
            "[answer] %s — calling generator model '%s' (provider=%s) …",
            label,
            gen_client.model,
            gen_client.provider,
        )

        gen_temp = round(base_temp + 0.1 * (attempt - 1), 2)

        try:
            output = generate(
                messages=messages,
                temperature=gen_temp,
                max_retries=1,
            )
        except Exception as exc:  # noqa: BLE001
            att_latency_ms = round((time.perf_counter() - t_att_start) * 1000, 2)
            logger.error("[answer] LLM call failed on %s: %s", label, exc)

            log_run(
                sample_id=sample_id,
                mode="answer",
                attempt=attempt,
                output=None,
                det_pass=0,
                semantic=0.0,
                critic=0.0,
                reliability=0.0,
                latency_ms=att_latency_ms,
            )

            history.append({
                "attempt":     attempt,
                "output":      None,
                "det_score":   None,
                "sem_score":   0.0,
                "crit_score":  0.0,
                "reliability": 0.0,
                "accepted":    False,
                "det_error":   f"LLM call failed: {exc}",
                "feedback":    None,
                "latency_ms":  att_latency_ms,
                "critic":      None,
            })
            break

        # Step 2: Deterministic validation if constraints exist
        if has_det:
            det_passed, det_error, check_details = evaluate_checks(active_constraints, output)
            det_score = 1.0 if det_passed else 0.0
            for cd in check_details:
                status = "PASS" if cd["passed"] else f"FAIL ({cd['error']})"
                logger.info("[answer] %s check [%s] -> %s", label, cd["type"], status)
        else:
            det_passed, det_error, check_details = True, "", []
            det_score = None

        # Step 3: Semantic relevance between answer and query
        sem = semantic_score(output, query)

        # Critic evaluation
        critic_result = critique(query, output)
        crit_score    = critic_result["score"]
        critic_issues = [issue for issue in critic_result.get("issues", []) if issue and "parse error" not in issue.lower() and "no json" not in issue.lower()]

        # Combined reliability
        reliability = combine(det_score, sem, crit_score)
        att_latency_ms = round((time.perf_counter() - t_att_start) * 1000, 2)

        # Pointless retries fix:
        # If there is no validator error and the critic returns no issues,
        # accept the answer instead of retrying with no feedback.
        has_validator_error = bool(det_error and det_error.strip() and det_score == 0.0)
        has_critic_issues = len(critic_issues) > 0

        accepted = (reliability >= ACCEPT_THRESHOLD) or (not has_validator_error and not has_critic_issues)
        final_output = output

        # Log sub-scores for every attempt
        det_str = f"{det_score:.2f}" if det_score is not None else "None"
        logger.info(
            "[answer] %s scores → det=%s  sem=%.2f  crit=%.2f  reliability=%.3f  (%.1fms)",
            label, det_str, sem, crit_score, reliability, att_latency_ms,
        )

        prev_output = history[-1].get("output") if history else None
        prev_error = history[-1].get("det_error") if history else None
        is_identical = bool(is_retry and prev_output is not None and output.strip() == str(prev_output).strip())

        # Stalled loop detection on consecutive identical output & error
        if is_retry and history:
            if output == prev_output and det_error == prev_error and not accepted:
                logger.warning(
                    "[answer] Stalled loop detected on %s: identical output and validator error. "
                    "Stopping retries and marking as 'possible checker error'.",
                    label,
                )
                history.append({
                    "attempt":        attempt,
                    "output":         output,
                    "det_score":      det_score,
                    "sem_score":      round(sem, 4),
                    "crit_score":     round(crit_score, 4),
                    "reliability":    reliability,
                    "accepted":       False,
                    "passed":         False,
                    "det_error":      "possible checker error",
                    "error":          "possible checker error",
                    "failure_reason": "possible checker error",
                    "critic":         critic_result,
                    "feedback":       None,
                    "latency_ms":     att_latency_ms,
                    "check_details":  check_details,
                })
                best = max(history, key=lambda h: h.get("reliability", 0.0))
                final_output = best.get("output")
                final_rel = best.get("reliability", 0.0)
                total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                return {
                    "answer":         final_output,
                    "reliability":    final_rel,
                    "retries_used":   retries_used,
                    "from_cache":     False,
                    "cache_hit":      False,
                    "constraints":    active_constraints,
                    "latency_ms":     total_latency_ms,
                    "history":        history,
                    "failure_reason": "possible checker error",
                    "error_message":  "possible checker error",
                    "passed":         False,
                }

        # Log to runs table
        log_run(
            sample_id=sample_id,
            mode="answer",
            attempt=attempt,
            output=output,
            det_pass=1 if (det_score == 1.0 or det_score is None) else 0,
            semantic=sem,
            critic=crit_score,
            reliability=reliability,
            latency_ms=att_latency_ms,
        )

        feedback_sent: str | None = None
        # Only retry if not accepted AND there is specific feedback to send
        if not accepted and attempt <= max_retries and (has_validator_error or has_critic_issues):
            retries_used += 1
            val_errors = [cd["error"] for cd in check_details if not cd.get("passed", True) and cd.get("error")]
            if not val_errors and has_validator_error:
                val_errors = [det_error]
            feedback_sent = build_retry_message(
                validator_errors=val_errors,
                critic_issues=critic_issues,
                is_identical=is_identical,
            )
            # Keep only the original request, the last answer, and this message in context
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": output},
                {"role": "user", "content": feedback_sent},
            ]
        elif not accepted and not (has_validator_error or has_critic_issues):
            # No specific issues found, accept rather than pointless retry
            accepted = True
            logger.info("[answer] No validator error and no critic issues found; accepting answer without retry.")

        history.append({
            "attempt":     attempt,
            "output":      output,
            "det_score":   det_score,
            "sem_score":   round(sem, 4),
            "crit_score":  round(crit_score, 4),
            "reliability": reliability,
            "accepted":    accepted,
            "det_error":   det_error,
            "critic":      critic_result,
            "feedback":    feedback_sent,
            "latency_ms":  att_latency_ms,
        })

        if accepted:
            passed = True
            logger.info("[answer] ✓ ACCEPTED on %s (reliability=%.3f)", label, reliability)
            break

        logger.warning(
            "[answer] ✗ REJECTED on %s (reliability=%.3f < %.2f)",
            label, reliability, ACCEPT_THRESHOLD,
        )

        if attempt <= max_retries:
            time.sleep(0.4)

    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    final_rel = history[-1]["reliability"] if history else 0.0

    # Step 4: Cache successful answer
    if enable_cache and final_output is not None and final_rel >= ACCEPT_THRESHOLD:
        try:
            save_cache(
                query=query,
                embedding=embed_text(query),
                response=final_output,
                reliability=final_rel,
                retries=retries_used,
            )
            logger.info("Saved live query answer to cache (reliability=%.3f)", final_rel)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save live query answer to cache: %s", exc)

    return {
        "answer":       final_output,
        "reliability":  final_rel,
        "retries_used": retries_used,
        "from_cache":   False,
        "cache_hit":    False,
        "constraints":  active_constraints,
        "latency_ms":   total_latency_ms,
        "history":      history,
        "passed":       passed,
    }
