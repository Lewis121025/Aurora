"""LLM settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Provider-specific LLM config."""

    base_url: str
    model: str
    api_key: str
    timeout_s: float = 30.0
    max_tokens: int = 1024
    enable_thinking: bool | None = None


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Public LLM settings shape."""

    provider: str
    config: LLMConfig


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


def _build_settings(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: str,
    max_tokens: str,
    enable_thinking_raw: str,
) -> LLMSettings | None:
    normalized_provider = provider.strip().lower()
    normalized_base_url = base_url.strip()
    normalized_api_key = api_key.strip()
    if not normalized_provider or not normalized_base_url or not normalized_api_key:
        return None

    enable_thinking = _parse_bool(enable_thinking_raw) if enable_thinking_raw else None
    if enable_thinking is None and normalized_provider == "bailian":
        enable_thinking = False

    return LLMSettings(
        provider=normalized_provider,
        config=LLMConfig(
            base_url=normalized_base_url,
            model=model.strip() or "gpt-4o-mini",
            api_key=normalized_api_key,
            timeout_s=float(timeout_s),
            max_tokens=int(max_tokens),
            enable_thinking=enable_thinking,
        ),
    )


def load_llm_settings() -> LLMSettings | None:
    """Load `AURORA_LLM_PROVIDER` and `AURORA_LLM_CONFIG_*` settings."""
    dotenv_values = _load_dotenv()
    provider = _pick_value(dotenv_values, "AURORA_LLM_PROVIDER").strip().lower()
    base_url = _pick_value(dotenv_values, "AURORA_LLM_CONFIG_BASE_URL")
    model = _pick_value(dotenv_values, "AURORA_LLM_CONFIG_MODEL")
    api_key = _pick_value(dotenv_values, "AURORA_LLM_CONFIG_API_KEY")
    timeout_s = _pick_value(dotenv_values, "AURORA_LLM_CONFIG_TIMEOUT_S") or "30.0"
    max_tokens = _pick_value(dotenv_values, "AURORA_LLM_CONFIG_MAX_TOKENS") or "1024"
    enable_thinking_raw = _pick_value(
        dotenv_values,
        "AURORA_LLM_CONFIG_ENABLE_THINKING",
    )
    return _build_settings(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        enable_thinking_raw=enable_thinking_raw,
    )


def coerce_llm_settings(raw: LLMSettings | Mapping[str, object]) -> LLMSettings:
    """Normalize a nested `llm_settings` mapping."""
    if isinstance(raw, LLMSettings):
        return raw

    provider = raw.get("provider")
    config = raw.get("config")
    if not isinstance(provider, str) or not isinstance(config, Mapping):
        raise ValueError("llm_settings must provide provider and config")

    base_url = config.get("base_url")
    model = config.get("model", "")
    api_key = config.get("api_key")
    timeout_s = config.get("timeout_s", 30.0)
    max_tokens = config.get("max_tokens", 1024)
    enable_thinking = config.get("enable_thinking", "")
    if not isinstance(base_url, str) or not isinstance(api_key, str):
        raise ValueError("llm_settings.config must provide base_url and api_key")
    if model is not None and not isinstance(model, str):
        raise ValueError("llm_settings.config.model must be a string")
    if enable_thinking not in {"", None} and not isinstance(enable_thinking, bool):
        raise ValueError("llm_settings.config.enable_thinking must be a boolean")

    settings = _build_settings(
        provider=provider,
        base_url=base_url,
        model=model or "",
        api_key=api_key,
        timeout_s=str(timeout_s),
        max_tokens=str(max_tokens),
        enable_thinking_raw="" if enable_thinking in {"", None} else str(enable_thinking),
    )
    if settings is None:
        raise ValueError("llm_settings must provide provider, base_url, and api_key")
    return settings
