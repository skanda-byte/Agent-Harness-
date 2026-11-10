"""
llm/groq_client.py — Groq provider implementation.

Targeted for the judge role: model "openai/gpt-oss-20b", reasoning_effort "low",
max_completion_tokens at least 3000, reading only the final message content.
If content is empty or finish_reason is "length", retries once with double token limit,
then returns typed error. Adds exponential backoff for rate limits and reads GROQ_API_KEY from .env.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIStatusError

from agentic_harness.llm.base import LLMClient, LLMResult

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_MIN_COMPLETION_TOKENS = 3000
MIN_API_CALL_GAP_SECONDS: float = 3.0

_LAST_GROQ_CALL_TIMESTAMP: float = 0.0


def _throttle_call(gap_seconds: float = MIN_API_CALL_GAP_SECONDS) -> None:
    """Ensure at least `gap_seconds` have elapsed since the last API call."""
    global _LAST_GROQ_CALL_TIMESTAMP
    if _LAST_GROQ_CALL_TIMESTAMP > 0:
        elapsed = time.perf_counter() - _LAST_GROQ_CALL_TIMESTAMP
        if elapsed < gap_seconds:
            sleep_dur = gap_seconds - elapsed
            logger.debug("[groq] Enforcing API gap: sleeping %.2fs", sleep_dur)
            time.sleep(sleep_dur)


class GroqClient(LLMClient):
    """Client for Groq cloud API endpoints."""

    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MIN_COMPLETION_TOKENS,
        reasoning_effort: str = "low",
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max(max_tokens, DEFAULT_MIN_COMPLETION_TOKENS),
            provider="groq",
            **kwargs,
        )
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env file or export it in your shell."
            )
        self._client = Groq(api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        """Call Groq API with exponential backoff on rate limits, handling truncation retries."""
        temp = self.temperature if temperature is None else temperature
        base_tokens = self.max_tokens if max_tokens is None else max(max_tokens, DEFAULT_MIN_COMPLETION_TOKENS)

        logger.info(
            "[groq] Calling model '%s' with %d messages (temp=%.2f, max_tokens=%d, reasoning_effort=%s) …",
            self.model,
            len(messages),
            temp,
            base_tokens,
            self.reasoning_effort,
        )

        t_start = time.perf_counter()

        def _execute_api_call(token_limit: int):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": float(temp),
                "max_completion_tokens": int(token_limit),
            }
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort

            delay = self.base_delay
            for attempt in range(1, self.max_retries + 1):
                _throttle_call(MIN_API_CALL_GAP_SECONDS)
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                    return resp, None
                except RateLimitError as exc:
                    if attempt == self.max_retries:
                        logger.error("[groq] Rate limit hit on all %d attempts: %s", self.max_retries, exc)
                        return None, f"groq_rate_limit_exceeded: {exc}"
                    logger.warning(
                        "[groq] Rate limit (attempt %d/%d). Backing off %.1fs …",
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, self.max_delay)
                except APIStatusError as exc:
                    if exc.status_code is not None and exc.status_code >= 500:
                        if attempt == self.max_retries:
                            logger.error("[groq] Server error %s on all attempts: %s", exc.status_code, exc)
                            return None, f"groq_server_error_{exc.status_code}: {exc}"
                        logger.warning(
                            "[groq] Server error %s (attempt %d/%d). Backing off %.1fs …",
                            exc.status_code,
                            attempt,
                            self.max_retries,
                            delay,
                        )
                        time.sleep(delay)
                        delay = min(delay * 2, self.max_delay)
                    else:
                        logger.error("[groq] API status error %s: %s", exc.status_code, exc)
                        return None, f"groq_api_error_{exc.status_code}: {exc}"
                except Exception as exc:  # noqa: BLE001
                    logger.error("[groq] Unexpected error: %s", exc)
                    return None, f"groq_call_failed: {exc}"
                finally:
                    global _LAST_GROQ_CALL_TIMESTAMP
                    _LAST_GROQ_CALL_TIMESTAMP = time.perf_counter()

            return None, "groq_retries_exhausted"

        # 1. Primary API call
        raw_resp, err = _execute_api_call(base_tokens)
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        if err or raw_resp is None:
            return LLMResult(
                text="",
                finish_reason=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                error=err or "groq_call_failed",
            )

        choice = raw_resp.choices[0] if raw_resp.choices else None
        content = (choice.message.content or "") if choice and choice.message else ""
        finish_reason = getattr(choice, "finish_reason", None) if choice else None

        usage = getattr(raw_resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        # 2. Check for empty response or 'length' cutoff -> retry once with double token limit
        if finish_reason == "length" or not content.strip():
            doubled_tokens = base_tokens * 2
            logger.warning(
                "[groq] Content empty or finish_reason=%r (limit=%d). Retrying once with doubled tokens=%d …",
                finish_reason,
                base_tokens,
                doubled_tokens,
            )
            raw_resp_retry, err_retry = _execute_api_call(doubled_tokens)
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            if err_retry or raw_resp_retry is None:
                typed_err = "groq_length_exceeded" if finish_reason == "length" else "groq_empty_response"
                return LLMResult(
                    text=content,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=elapsed_ms,
                    error=typed_err,
                    raw_response=raw_resp,
                )

            choice_r = raw_resp_retry.choices[0] if raw_resp_retry.choices else None
            content_r = (choice_r.message.content or "") if choice_r and choice_r.message else ""
            finish_reason_r = getattr(choice_r, "finish_reason", None) if choice_r else None
            usage_r = getattr(raw_resp_retry, "usage", None)
            if usage_r:
                prompt_tokens = getattr(usage_r, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage_r, "completion_tokens", completion_tokens)

            if finish_reason_r == "length" or not content_r.strip():
                typed_err = "groq_length_exceeded" if finish_reason_r == "length" else "groq_empty_response"
                logger.error("[groq] Retry still failed with %s. Returning typed error.", typed_err)
                return LLMResult(
                    text=content_r,
                    finish_reason=finish_reason_r,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=elapsed_ms,
                    error=typed_err,
                    raw_response=raw_resp_retry,
                )

            logger.info(
                "[groq] Retry succeeded (tokens=%d, latency=%.1fms)",
                completion_tokens,
                elapsed_ms,
            )
            return LLMResult(
                text=content_r,
                finish_reason=finish_reason_r,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=elapsed_ms,
                error=None,
                raw_response=raw_resp_retry,
            )

        logger.info(
            "[groq] Model '%s' generated %d tokens (latency=%.1fms)",
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
            raw_response=raw_resp,
        )

