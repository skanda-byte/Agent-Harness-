"""
tests/test_validators.py — Unit tests for sentence splitter and new validators.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentic_harness.validators import (
    split_sentences,
    check_sentence_count,
    check_words_per_sentence,
    check_no_commas,
    check_total_word_count,
    run_checker,
)


def test_split_sentences_basic():
    text = "First sentence. Second sentence! Third sentence?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence"
    assert sentences[1] == "Second sentence"
    assert sentences[2] == "Third sentence"


def test_split_sentences_trailing_newlines_and_spaces():
    text = "First sentence. Second sentence.\n\n"
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "First sentence"
    assert sentences[1] == "Second sentence"


def test_split_sentences_ignore_empty_segments():
    text = "Hello world.  ... !  How are you doing today?   "
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "Hello world"
    assert sentences[1] == "How are you doing today"


def test_split_sentences_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   \n\t  ") == []


def test_split_sentences_without_trailing_punctuation():
    text = "This is sentence one. This is sentence two without dot"
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "This is sentence one"
    assert sentences[1] == "This is sentence two without dot"


def test_check_sentence_count():
    text = "Sentence one. Sentence two! Sentence three?"
    passed, err = check_sentence_count(text, {"n": 3})
    assert passed is True
    assert err == ""

    passed, err = check_sentence_count(text, {"n": 2})
    assert passed is False
    assert "Expected 2 sentence(s), found 3" in err


def test_check_words_per_sentence():
    # 3 sentences, each has 5 words
    text = "This is a five sentence. Here is another five one. That makes three five ones."
    passed, err = check_words_per_sentence(text, {"min": 4, "max": 6})
    assert passed is True
    assert err == ""

    # Sentence 2 has 3 words
    bad_text = "This is a five sentence. Too short now. That makes three five ones."
    passed, err = check_words_per_sentence(bad_text, {"min": 4, "max": 6})
    assert passed is False
    assert "Sentence 2 has 3 words" in err


def test_check_no_commas():
    text_clean = "This is a clean sentence without any punctuation except a period."
    passed, err = check_no_commas(text_clean, {})
    assert passed is True

    text_comma = "This sentence, unfortunately, contains commas."
    passed, err = check_no_commas(text_comma, {})
    assert passed is False
    assert "Response contains commas" in err


def test_check_total_word_count():
    text = "one two three four five"
    passed, err = check_total_word_count(text, {"min": 3, "max": 6})
    assert passed is True

    passed, err = check_total_word_count(text, {"min": 6, "max": 10})
    assert passed is False
    assert "Total word count 5 is outside [6, 10]" in err


def test_checker_registry():
    passed, _ = run_checker("no_commas", "No commas here", {})
    assert passed is True

    passed, _ = run_checker("sentence_count", "One. Two.", {"n": 2})
    assert passed is True


if __name__ == "__main__":
    test_split_sentences_basic()
    test_split_sentences_trailing_newlines_and_spaces()
    test_split_sentences_ignore_empty_segments()
    test_split_sentences_empty_input()
    test_split_sentences_without_trailing_punctuation()
    test_check_sentence_count()
    test_check_words_per_sentence()
    test_check_no_commas()
    test_check_total_word_count()
    test_checker_registry()
    print("ALL VALIDATOR & SPLITTER UNIT TESTS PASSED!")

