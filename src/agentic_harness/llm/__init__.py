"""
agentic_harness.llm — Multi-provider LLM factory and clients.

Provides:
    LLMResult: Standardized result object.
    LLMClient: Abstract base client.
    OllamaClient: Ollama local chat client.
    GroqClient: Groq cloud client.
    get_client(role): Factory building the client for role ("generator" or "judge") from config.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from agentic_harness.llm.base import LLMClient, LLMResult
from agentic_harness.llm.groq_client import GroqClient
from agentic_harness.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "generator": {
        "provider": "ollama",
        "model": "qwen2.5:1.5b",
        "temperature": 0.3,
        "max_tokens": 400,
    },
    "judge": {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "reasoning_effort": "low",
        "temperature": 0.0,
        "max_tokens": 3000,
    },
}

_CLIENT_CACHE: Dict[str, LLMClient] = {}


def _find_config_file(explicit_path: Path | str | None = None) -> Path | None:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p

    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path(__file__).resolve().parent.parent.parent.parent / "config.yaml",
        Path(__file__).resolve().parent.parent.parent / "config.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration dictionary from config.yaml or return defaults."""
    cfg_file = _find_config_file(config_path)
    if cfg_file is not None:
        try:
            with cfg_file.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read config from %s: %s. Using defaults.", cfg_file, exc)

    return DEFAULT_CONFIG


def get_client(
    role: str = "generator",
    config_path: Path | str | None = None,
    use_cache: bool = True,
) -> LLMClient:
    """Build and return an LLMClient for the requested role ('generator' or 'judge').

    Args:
        role: Logical role in the harness, typically 'generator' or 'judge'.
        config_path: Optional explicit path to config.yaml.
        use_cache: Whether to return a cached client instance for this role.

    Returns:
        An LLMClient instance (OllamaClient, GroqClient, etc.).
    """
    if use_cache and role in _CLIENT_CACHE:
        return _CLIENT_CACHE[role]

    cfg = load_config(config_path)
    if role not in cfg:
        raise KeyError(
            f"Role {role!r} not found in configuration. Available roles: {list(cfg.keys())}"
        )

    role_cfg = cfg[role]
    provider = str(role_cfg.get("provider", "")).lower()
    model = role_cfg.get("model")
    temperature = float(role_cfg.get("temperature", 0.7))
    max_tokens = int(role_cfg.get("max_tokens", 1000))
    extra_kwargs = {k: v for k, v in role_cfg.items() if k not in ("provider", "model", "temperature", "max_tokens")}

    client: LLMClient
    if provider == "ollama":
        client = OllamaClient(
            model=model or "qwen2.5:1.5b",
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
    elif provider == "groq":
        client = GroqClient(
            model=model or "openai/gpt-oss-20b",
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider {provider!r} for role {role!r}. Supported providers: 'ollama', 'groq'."
        )

    if use_cache:
        _CLIENT_CACHE[role] = client

    return client


def set_client(role: str, client: LLMClient) -> None:
    """Explicitly register a client instance for a given role (useful for tests/mocking)."""
    _CLIENT_CACHE[role] = client


def clear_client_cache() -> None:
    """Clear all cached client instances."""
    _CLIENT_CACHE.clear()


__all__ = [
    "LLMClient",
    "LLMResult",
    "OllamaClient",
    "GroqClient",
    "get_client",
    "set_client",
    "clear_client_cache",
    "load_config",
]

