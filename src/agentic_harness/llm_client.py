"""
llm_client.py — Backwards-compatibility shim delegating to agentic_harness.llm.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from agentic_harness.llm import get_client, LLMResult, LLMClient

logger = logging.getLogger(__name__)


class LLMResponseError(RuntimeError):
    """Raised when an LLM call fails, returns empty content, or cuts off."""
    def __init__(self, message: str, finish_reason: str | None = None, raw_response: Any = None):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.raw_response = raw_response


def generate(
    messages: List[dict],
    temperature: Optional[float] = None,
    *,
    role: str = "generator",
    **kwargs: Any,
) -> str:
    """Delegates generation to the configured client for `role` ('generator' or 'judge')."""
    client = get_client(role)
    max_tokens = kwargs.get("max_tokens", kwargs.get("max_completion_tokens"))
    res: LLMResult = client.generate(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if res.error:
        raise LLMResponseError(
            f"[{client.provider}:{client.model}] {res.error}: {res.text}",
            finish_reason=res.finish_reason,
            raw_response=res.raw_response,
        )
    return res.text


__all__ = ["generate", "LLMResponseError", "get_client", "LLMClient", "LLMResult"]
