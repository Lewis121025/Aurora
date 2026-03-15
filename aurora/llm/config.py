from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: float = 30.0


def load_llm_config() -> LLMConfig | None:
    base_url = os.environ.get("AURORA_LLM_BASE_URL", "")
    model = os.environ.get("AURORA_LLM_MODEL", "")
    api_key = os.environ.get("AURORA_LLM_API_KEY", "")
    if not base_url or not api_key:
        return None
    return LLMConfig(
        base_url=base_url,
        model=model or "gpt-4o-mini",
        api_key=api_key,
        timeout_s=float(os.environ.get("AURORA_LLM_TIMEOUT_S", "30.0")),
    )
