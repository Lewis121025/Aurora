"""LLM configuration module.

Defines LLM configuration and runtime parameters:
- LLM configuration (environment variable loading)
- Distillation threshold configuration
- Session timeout configuration
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM configuration.

    Attributes:
        base_url: LLM API base URL.
        model: Model name.
        api_key: API key.
        timeout_s: Request timeout in seconds.
    """

    base_url: str
    model: str
    api_key: str
    timeout_s: float = 30.0
    max_tokens: int = 1024
    enable_thinking: bool | None = None


DISTILL_THRESHOLD_TURNS = 20
"""Distillation trigger threshold: triggers when session turn count reaches this value."""

SESSION_IDLE_TIMEOUT_MINUTES = 30
"""Session idle timeout in minutes: triggers distillation after this duration without interaction."""

EMBEDDING_DIM = 384
"""Embedding dimension (MiniLM all-MiniLM-L6-v2)."""


def _load_dotenv() -> dict[str, str]:
    path = Path(".env")
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized = value.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
            normalized = normalized[1:-1]
        values[key.strip()] = normalized
    return values


def _pick_value(dotenv_values: dict[str, str], *names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        fallback = dotenv_values.get(name, "")
        if fallback:
            return fallback
    return ""


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {raw}")


def load_llm_config() -> LLMConfig | None:
    """Load LLM configuration from environment variables.

    Required environment variables:
        - AURORA_LLM_BASE_URL
        - AURORA_LLM_API_KEY

    Optional environment variables:
        - AURORA_LLM_MODEL: defaults to "gpt-4o-mini"
        - AURORA_LLM_TIMEOUT_S: defaults to 30.0

    Returns:
        LLMConfig: configuration object; None if required variables are missing.
    """
    dotenv_values = _load_dotenv()
    provider = _pick_value(dotenv_values, "AURORA_LLM_PROVIDER").strip().lower()
    provider_prefix = f"AURORA_{provider.upper()}_LLM" if provider else ""
    provider_names = (
        f"{provider_prefix}_BASE_URL",
        f"{provider_prefix}_MODEL",
        f"{provider_prefix}_API_KEY",
    ) if provider_prefix else ("", "", "")

    base_url = _pick_value(dotenv_values, "AURORA_LLM_BASE_URL", provider_names[0])
    model = _pick_value(dotenv_values, "AURORA_LLM_MODEL", provider_names[1])
    api_key = _pick_value(dotenv_values, "AURORA_LLM_API_KEY", provider_names[2])
    timeout_s = _pick_value(dotenv_values, "AURORA_LLM_TIMEOUT_S", "AURORA_LLM_TIMEOUT") or "30.0"
    max_tokens = _pick_value(dotenv_values, "AURORA_LLM_MAX_TOKENS") or "1024"
    enable_thinking_raw = _pick_value(
        dotenv_values,
        "AURORA_LLM_ENABLE_THINKING",
        f"{provider_prefix}_ENABLE_THINKING" if provider_prefix else "",
    )
    enable_thinking = _parse_bool(enable_thinking_raw) if enable_thinking_raw else None
    if enable_thinking is None and provider == "bailian":
        enable_thinking = False

    if not base_url or not api_key:
        return None

    return LLMConfig(
        base_url=base_url,
        model=model or "gpt-4o-mini",
        api_key=api_key,
        timeout_s=float(timeout_s),
        max_tokens=int(max_tokens),
        enable_thinking=enable_thinking,
    )
