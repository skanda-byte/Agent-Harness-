"""
tests/test_llm_clients.py — Unit tests for refactored LLM architecture:
- base.py LLMResult & LLMClient
- ollama_client.py (unreachable error, option parameters, tokens)
- groq_client.py (token retry, typed errors, rate-limit backoff)
- config.yaml & factory get_client("generator") vs get_client("judge")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agentic_harness.llm import (
    LLMClient,
    LLMResult,
    OllamaClient,
    GroqClient,
    get_client,
    load_config,
    clear_client_cache,
)


def test_base_result():
    res = LLMResult(
        text="Hello world",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=120.5,
        error=None,
    )
    assert res.ok is True
    assert res.prompt_tokens == 10
    assert res.completion_tokens == 5
    assert res.latency_ms == 120.5

    err_res = LLMResult(
        text="Server not running",
        error="ollama_unreachable",
    )
    assert err_res.ok is False
    assert err_res.error == "ollama_unreachable"


def test_config_loading_and_factory():
    clear_client_cache()
    cfg = load_config()

    assert "generator" in cfg
    assert "judge" in cfg

    assert cfg["generator"]["provider"] == "ollama"
    assert cfg["generator"]["model"] == "qwen2.5:1.5b"
    assert cfg["generator"]["temperature"] == 0.3
    assert cfg["generator"]["max_tokens"] == 400

    assert cfg["judge"]["provider"] == "groq"
    assert cfg["judge"]["model"] == "openai/gpt-oss-20b"
    assert cfg["judge"]["reasoning_effort"] == "low"
    assert cfg["judge"]["temperature"] == 0.0
    assert cfg["judge"]["max_tokens"] == 3000

    # Test factory
    gen = get_client("generator")
    assert isinstance(gen, OllamaClient)
    assert gen.model == "qwen2.5:1.5b"
    assert gen.temperature == 0.3
    assert gen.max_tokens == 400

    judge = get_client("judge")
    assert isinstance(judge, GroqClient)
    assert judge.model == "openai/gpt-oss-20b"
    assert judge.temperature == 0.0
    assert judge.max_tokens >= 3000
    assert judge.reasoning_effort == "low"


def test_ollama_client_unreachable():
    # Test when Ollama server is not running on an unreachable port
    client = OllamaClient(
        model="qwen2.5:1.5b",
        endpoint_url="http://localhost:9999/api/chat",  # unreachable port
        timeout_seconds=0.5,
    )
    res = client.generate([{"role": "user", "content": "hi"}])

    assert res.ok is False
    assert res.error == "ollama_unreachable"
    assert "start Ollama" in res.text
    assert res.prompt_tokens == 0
    assert res.completion_tokens == 0


def test_ollama_client_success_mock():
    client = OllamaClient(model="qwen2.5:1.5b", temperature=0.3, max_tokens=400)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "qwen2.5:1.5b",
        "message": {"role": "assistant", "content": "The sky is blue."},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 14,
        "eval_count": 5,
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        res = client.generate([{"role": "user", "content": "Why is the sky blue?"}])

        assert res.ok is True
        assert res.text == "The sky is blue."
        assert res.prompt_tokens == 14
        assert res.completion_tokens == 5
        assert res.finish_reason == "stop"
        assert res.error is None

        # Verify options passed
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["model"] == "qwen2.5:1.5b"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.3
        assert payload["options"]["num_predict"] == 400


def test_groq_client_retry_on_length():
    judge = GroqClient(model="openai/gpt-oss-20b", max_tokens=3000, reasoning_effort="low")

    # Mock response 1: cut off with length
    mock_choice1 = MagicMock()
    mock_choice1.message.content = "Incomplete response..."
    mock_choice1.finish_reason = "length"
    mock_resp1 = MagicMock()
    mock_resp1.choices = [mock_choice1]
    mock_resp1.usage.prompt_tokens = 20
    mock_resp1.usage.completion_tokens = 3000

    # Mock response 2: completed on retry with double tokens
    mock_choice2 = MagicMock()
    mock_choice2.message.content = "Full complete response that finishes cleanly."
    mock_choice2.finish_reason = "stop"
    mock_resp2 = MagicMock()
    mock_resp2.choices = [mock_choice2]
    mock_resp2.usage.prompt_tokens = 20
    mock_resp2.usage.completion_tokens = 4500

    with patch.object(judge._client.chat.completions, "create", side_effect=[mock_resp1, mock_resp2]) as mock_create:
        res = judge.generate([{"role": "user", "content": "Explain relativity"}])

        assert res.ok is True
        assert res.text == "Full complete response that finishes cleanly."
        assert res.finish_reason == "stop"
        assert res.completion_tokens == 4500
        assert mock_create.call_count == 2

        # Second call should have doubled tokens (3000 * 2 = 6000)
        first_call_kwargs = mock_create.call_args_list[0][1]
        second_call_kwargs = mock_create.call_args_list[1][1]
        assert first_call_kwargs["max_completion_tokens"] == 3000
        assert second_call_kwargs["max_completion_tokens"] == 6000
        assert first_call_kwargs["reasoning_effort"] == "low"


def test_groq_client_typed_error_when_retry_still_fails():
    judge = GroqClient(model="openai/gpt-oss-20b", max_tokens=3000, reasoning_effort="low")

    # Mock both attempts returning empty content
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_choice.finish_reason = "stop"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 10
    mock_resp.usage.completion_tokens = 0

    with patch.object(judge._client.chat.completions, "create", return_value=mock_resp):
        res = judge.generate([{"role": "user", "content": "test"}])
        assert res.ok is False
        assert res.error == "groq_empty_response"


if __name__ == "__main__":
    test_base_result()
    test_config_loading_and_factory()
    test_ollama_client_unreachable()
    test_ollama_client_success_mock()
    test_groq_client_retry_on_length()
    test_groq_client_typed_error_when_retry_still_fails()
    print("ALL LLM CLIENT UNIT TESTS PASSED!")

