"""
llm/ollama_client.py — Ollama provider implementation.

POSTs to http://localhost:11434/api/chat with model, messages, stream=False,
options={"temperature": ..., "num_predict": max_tokens}, and a 120-second timeout.
If the server is unreachable, returns typed error "ollama_unreachable".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from agentic_harness.llm.base import LLMClient, LLMResult

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_TIMEOUT_SECONDS = 120.0


class OllamaClient(LLMClient):
    """Client for local Ollama chat completions endpoint."""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.3,
        max_tokens: int = 400,
        endpoint_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider="ollama",
            **kwargs,
        )
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        """Call Ollama /api/chat and return a structured LLMResult."""
        temp = self.temperature if temperature is None else temperature
        tokens = self.max_tokens if max_tokens is None else max_tokens

        logger.info(
            "[ollama] Calling model '%s' with %d messages (temp=%.2f, max_tokens=%d) …",
            self.model,
            len(messages),
            temp,
            tokens,
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temp),
                "num_predict": int(tokens),
            },
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(self.endpoint_url, json=payload)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            msg = (
                f"Ollama server is unreachable at {self.endpoint_url}. "
                f"Please start Ollama ('ollama serve') and ensure model '{self.model}' "
                f"is available ('ollama pull {self.model}'). Error: {exc}"
            )
            logger.error("[ollama] %s", msg)
            return LLMResult(
                text=msg,
                finish_reason=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                error="ollama_unreachable",
                raw_response={"error": str(exc)},
            )
        except httpx.TimeoutException as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            msg = f"Ollama call timed out after {self.timeout_seconds}s for model '{self.model}': {exc}"
            logger.error("[ollama] %s", msg)
            return LLMResult(
                text=msg,
                finish_reason=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                error="ollama_timeout",
                raw_response={"error": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            msg = f"Unexpected error during Ollama invocation ({self.model}): {exc}"
            logger.error("[ollama] %s", msg)
            return LLMResult(
                text=msg,
                finish_reason=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                error="ollama_error",
                raw_response={"error": str(exc)},
            )

        if response.status_code != 200:
            error_type = "ollama_model_not_found" if response.status_code == 404 else "ollama_http_error"
            msg = f"Ollama returned HTTP {response.status_code}: {response.text}"
            logger.error("[ollama] %s", msg)
            return LLMResult(
                text=msg,
                finish_reason=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                error=error_type,
                raw_response=response.text,
            )

        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to parse Ollama JSON response: {exc}"
            logger.error("[ollama] %s", msg)
            return LLMResult(
                text=msg,
                finish_reason=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                error="ollama_invalid_json",
                raw_response=response.text,
            )

        content = data.get("message", {}).get("content", "")
        finish_reason = data.get("done_reason") or ("stop" if data.get("done") else None)
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)

        logger.info(
            "[ollama] Model '%s' generated %d tokens (latency=%.1fms)",
            self.model,
            completion_tokens,
            elapsed_ms,
        )

        return LLMResult(
            text=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed_ms,
            error=None,
            raw_response=data,
        )

