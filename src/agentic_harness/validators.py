"""
validators.py — One function per checker type used in benchmark/samples.json.

Every validator returns a (passed: bool, error_message: str) tuple.
An empty error_message means no error (but always check `passed`).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Type alias for the return value of every validator
# ---------------------------------------------------------------------------
ValidatorResult = Tuple[bool, str]


# ---------------------------------------------------------------------------
# Sentence splitting utility
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? followed by whitespace or end of text.

    Ignores empty segments and cleanly handles trailing newlines or extra spaces.
    """
    if not text:
        return []
    # Split on one or more . ! ? characters followed by whitespace or end of string
    raw_segments = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    # Ignore empty or whitespace-only segments
    sentences = [s.strip() for s in raw_segments if s.strip()]
    return sentences


# ---------------------------------------------------------------------------
# 1. valid_json_with_keys
# ---------------------------------------------------------------------------

def check_valid_json_with_keys(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* is valid JSON containing all *required_keys*.

    params:
        required_keys (list[str]): Keys that must be present in the parsed JSON object.
    """
    required_keys: list[str] = params.get("required_keys", params.get("keys", []))

    # Strip markdown code fences if the model wrapped the JSON
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop first line (``` or ```json) and last ``` line
        inner = [l for l in lines[1:] if l.strip() != "```"]
        cleaned = "\n".join(inner).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"

    if not isinstance(parsed, dict):
        return False, f"Expected a JSON object (dict), got {type(parsed).__name__}."

    missing = [k for k in required_keys if k not in parsed]
    if missing:
        return False, f"Missing required keys: {missing}"

    return True, ""


# ---------------------------------------------------------------------------
# 2. numeric_match
# ---------------------------------------------------------------------------

def check_numeric_match(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that the final number in *response* equals *expected* within *tolerance*.

    params:
        expected  (int | float): The correct numerical answer.
        tolerance (int | float): Allowed absolute deviation (default 0).
    """
    expected: float = float(params["expected"])
    tolerance: float = float(params.get("tolerance", 0))

    # Find all numbers (int or float, optionally negative) in the response
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    if not numbers:
        return False, f"No number found in response: {response!r}"

    # Use the last number found (models often put the answer at the end)
    actual = float(numbers[-1])

    if abs(actual - expected) <= tolerance:
        return True, ""
    return (
        False,
        f"Expected {expected} (±{tolerance}), got {actual}. Response: {response!r}",
    )


# ---------------------------------------------------------------------------
# 3. word_count_range
# ---------------------------------------------------------------------------

def check_word_count_range(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that the word count of *response* falls within [min_words, max_words].

    params:
        min_words (int): Minimum acceptable word count (inclusive).
        max_words (int): Maximum acceptable word count (inclusive).
    """
    min_words: int = int(params["min_words"])
    max_words: int = int(params["max_words"])

    word_count = len(response.split())
    if min_words <= word_count <= max_words:
        return True, ""
    return (
        False,
        f"Word count {word_count} is outside [{min_words}, {max_words}].",
    )


# ---------------------------------------------------------------------------
# 4. regex_match
# ---------------------------------------------------------------------------

def check_regex_match(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that the stripped *response* matches *pattern*.

    params:
        pattern (str): A Python regex pattern. The whole response (stripped)
                       must match via re.fullmatch.
    """
    pattern: str = params["pattern"]
    text = response.strip()

    try:
        match = re.fullmatch(pattern, text)
    except re.error as exc:
        return False, f"Invalid regex pattern {pattern!r}: {exc}"

    if match:
        return True, ""
    return False, f"Response {text!r} does not match pattern {pattern!r}."


# ---------------------------------------------------------------------------
# 5. exact_bullet_count
# ---------------------------------------------------------------------------

def check_exact_bullet_count(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* contains exactly *expected_count* bullet lines.

    A bullet line is any non-empty line whose stripped content starts with
    *bullet_prefix* (default "-").

    params:
        expected_count (int): The required number of bullet points.
        bullet_prefix  (str): Character(s) that start a bullet (default "-").
    """
    expected_count: int = int(params.get("expected_count", params.get("n", 0)))
    bullet_prefix: str = params.get("bullet_prefix", "-")

    lines = response.splitlines()
    bullet_lines = [l for l in lines if l.strip().startswith(bullet_prefix)]

    if len(bullet_lines) == expected_count:
        return True, ""
    return (
        False,
        f"Expected {expected_count} bullet(s) starting with {bullet_prefix!r}, "
        f"found {len(bullet_lines)}.",
    )


# ---------------------------------------------------------------------------
# 6. sentence_count
# ---------------------------------------------------------------------------

def check_sentence_count(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* contains exactly *n* sentences.

    params:
        n (int): The required number of sentences.
    """
    expected_n = int(params.get("n", params.get("expected_count", 0)))
    sentences = split_sentences(response)

    if len(sentences) == expected_n:
        return True, ""
    return (
        False,
        f"Expected {expected_n} sentence(s), found {len(sentences)}.",
    )


# ---------------------------------------------------------------------------
# 7. words_per_sentence
# ---------------------------------------------------------------------------

def check_words_per_sentence(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *every* sentence in *response* has word count within [min, max].

    params:
        min (int): Minimum word count per sentence (inclusive, default 0).
        max (int): Maximum word count per sentence (inclusive, default 999999).
    """
    min_words = int(params.get("min", params.get("min_words", 0)))
    max_words = int(params.get("max", params.get("max_words", 999999)))

    sentences = split_sentences(response)
    if not sentences:
        return False, "No sentences found in response."

    for idx, s in enumerate(sentences, 1):
        count = len(s.split())
        if count < min_words or count > max_words:
            return (
                False,
                f"Sentence {idx} has {count} words (expected {min_words}–{max_words}): {s!r}",
            )

    return True, ""


# ---------------------------------------------------------------------------
# 8. no_commas
# ---------------------------------------------------------------------------

def check_no_commas(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* does not contain any comma characters.

    params:
        None required.
    """
    if "," in response:
        return False, "Response contains commas, but no commas are allowed."
    return True, ""


# ---------------------------------------------------------------------------
# 9. total_word_count
# ---------------------------------------------------------------------------

def check_total_word_count(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that the total word count of *response* falls within [min, max].

    params:
        min (int): Minimum total words (inclusive, default 0).
        max (int): Maximum total words (inclusive, default 999999).
    """
    min_words = int(params.get("min", params.get("min_words", 0)))
    max_words = int(params.get("max", params.get("max_words", 999999)))

    count = len(response.split())
    if min_words <= count <= max_words:
        return True, ""
    return (
        False,
        f"Total word count {count} is outside [{min_words}, {max_words}].",
    )


# ---------------------------------------------------------------------------
# 10. contains_all
# ---------------------------------------------------------------------------

def check_contains_all(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* contains all specified keywords/phrases (case-insensitive)."""
    words = params.get("words", params.get("keywords", []))
    if isinstance(words, str):
        words = [words]
    lower = response.lower()
    missing = [w for w in words if w.lower() not in lower]
    if missing:
        return False, f"Missing required words/phrases: {missing}"
    return True, ""


# ---------------------------------------------------------------------------
# 11. forbidden_words
# ---------------------------------------------------------------------------

def check_forbidden_words(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* contains NONE of the specified forbidden words (case-insensitive, whole-word matching)."""
    words = params.get("words", params.get("forbidden", []))
    if isinstance(words, str):
        words = [words]
    lower = response.lower()
    found = []
    for w in words:
        if re.search(r"\b" + re.escape(w.lower()) + r"\b", lower):
            found.append(w)
    if found:
        return False, f"Found forbidden words/phrases: {found}"
    return True, ""


# ---------------------------------------------------------------------------
# 12. starts_with
# ---------------------------------------------------------------------------

def check_starts_with(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* starts with *prefix*."""
    prefix = str(params.get("prefix", ""))
    if not response.strip().startswith(prefix):
        return False, f"Response does not start with {prefix!r}"
    return True, ""


# ---------------------------------------------------------------------------
# 13. words_per_bullet
# ---------------------------------------------------------------------------

def check_words_per_bullet(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that each bullet line contains between [min, max] words, or exactly n words."""
    min_words = params.get("min")
    max_words = params.get("max")
    exact_n = params.get("n")
    if exact_n is not None:
        min_words = int(exact_n)
        max_words = int(exact_n)
    else:
        min_words = int(min_words) if min_words is not None else 0
        max_words = int(max_words) if max_words is not None else 999999

    lines = [l.strip() for l in response.splitlines() if l.strip()]
    bullet_lines = [l for l in lines if re.match(r"^[-*•\d+.]\s*", l)]
    if not bullet_lines:
        return False, "No bullet points found in response."

    for idx, b in enumerate(bullet_lines, 1):
        cleaned = re.sub(r"^[-*•\d+.]\s*", "", b).strip()
        count = len(cleaned.split())
        if count < min_words or count > max_words:
            if min_words == max_words:
                return False, f"Bullet {idx} has {count} words (expected exactly {min_words}): {b!r}"
            return False, f"Bullet {idx} has {count} words (expected {min_words}-{max_words}): {b!r}"
    return True, ""


# ---------------------------------------------------------------------------
# 14. distinct_first_letters
# ---------------------------------------------------------------------------

def check_distinct_first_letters(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that each sentence or bullet point begins with a distinct letter."""
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    bullet_lines = [l for l in lines if re.match(r"^[-*•\d+.]\s*", l)]
    if bullet_lines:
        segments = [re.sub(r"^[-*•\d+.]\s*", "", b).strip() for b in bullet_lines]
    else:
        segments = split_sentences(response)

    if not segments:
        return False, "No sentences or bullets found."

    seen: dict[str, int] = {}
    for idx, seg in enumerate(segments, 1):
        first_letter = ""
        for char in seg:
            if char.isalpha():
                first_letter = char.lower()
                break
        if not first_letter:
            continue
        if first_letter in seen:
            return (
                False,
                f"Duplicate first letter '{first_letter.upper()}' found in item {idx} "
                f"(previously in item {seen[first_letter]}): {seg!r}",
            )
        seen[first_letter] = idx

    return True, ""


# ---------------------------------------------------------------------------
# 15. contains_phrase
# ---------------------------------------------------------------------------

def check_contains_phrase(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* contains the specified phrase (case-insensitive)."""
    phrase = str(params.get("phrase", "")).strip()
    if not phrase:
        return True, ""
    if phrase.lower() not in response.lower():
        return False, f"Response is missing required phrase: {phrase!r}"
    return True, ""


# ---------------------------------------------------------------------------
# 16. banned_letters
# ---------------------------------------------------------------------------

def check_banned_letters(
    response: str,
    params: Dict[str, Any],
) -> ValidatorResult:
    """Check that *response* does not contain any of the banned letters (case-insensitive)."""
    raw_letters = params.get("letters", [])
    if isinstance(raw_letters, str):
        letters = [c.lower() for c in raw_letters if c.isalpha()]
    else:
        letters = [str(c).lower() for c in raw_letters if str(c).strip()]

    lower = response.lower()
    found = [c for c in sorted(set(letters)) if c in lower]
    if found:
        return False, f"Response contains banned letter(s): {found}"
    return True, ""


# ---------------------------------------------------------------------------
# Dispatch helper used by run_bench.py
# ---------------------------------------------------------------------------

CHECKER_REGISTRY: Dict[str, Any] = {
    "valid_json_with_keys": check_valid_json_with_keys,
    "json_keys": check_valid_json_with_keys,
    "numeric_match": check_numeric_match,
    "number": check_numeric_match,
    "word_count_range": check_word_count_range,
    "word_count": check_word_count_range,
    "total_word_count": check_total_word_count,
    "sentence_count": check_sentence_count,
    "words_per_sentence": check_words_per_sentence,
    "words_per_bullet": check_words_per_bullet,
    "distinct_first_letters": check_distinct_first_letters,
    "contains_phrase": check_contains_phrase,
    "banned_letters": check_banned_letters,
    "no_commas": check_no_commas,
    "regex": check_regex_match,
    "regex_match": check_regex_match,
    "exact_bullet_count": check_exact_bullet_count,
    "bullet_count": check_exact_bullet_count,
    "contains_all": check_contains_all,
    "forbidden_words": check_forbidden_words,
    "starts_with": check_starts_with,
}


def run_checker(checker_type: str, response: str, params: Dict[str, Any]) -> ValidatorResult:
    """Look up and call the validator for *checker_type*.

    Raises:
        KeyError: If *checker_type* is not registered.
    """
    fn = CHECKER_REGISTRY.get(checker_type)
    if fn is None:
        raise KeyError(
            f"Unknown checker type {checker_type!r}. "
            f"Available: {list(CHECKER_REGISTRY.keys())}"
        )
    return fn(response, params)

