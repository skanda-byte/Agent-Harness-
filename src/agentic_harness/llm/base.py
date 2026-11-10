"""
llm/base.py — Abstract LLM client interface and standard LLMResult structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMResult:
    """Standardized response from any LLM provider.

    Attributes:
        text: Generated completion text.
        finish_reason: Reason generation finished ('stop', 'length', etc.) or None.
        prompt_tokens: Number of input prompt tokens evaluated.
        completion_tokens: Number of output completion tokens generated.
        latency_ms: Request round-trip duration in milliseconds.
        error: None on success, or a typed error string (e.g. 'ollama_unreachable', 'groq_length_exceeded').
        raw_response: Underlying provider response object or dict, if available.
    """
    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    raw_response: Any = None

    @property
    def ok(self) -> bool:
        """True if the generation completed without error and produced non-empty text."""
        return self.error is None and bool(self.text.strip())


class LLMClient(ABC):
    """Abstract base class for all LLM client implementations."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        provider: str = "custom",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.extra_config = kwargs

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        """Generate an assistant reply for the given list of chat messages.

        Args:
            messages: List of chat messages (e.g. [{'role': 'user', 'content': '...'}]).
            temperature: Sampling temperature override (or default to client setting).
            max_tokens: Maximum token limit override (or default to client setting).

        Returns:
            An LLMResult instance containing text, tokens, latency, and error state.
        """
        pass

