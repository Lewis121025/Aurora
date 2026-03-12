from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _first_env(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value is not None and value != "":
            return value
    return default


@dataclass(frozen=True)
class AuroraSettings:
    data_dir: str = ".aurora_seed_v1"
    sealed_state_filename: str = "sealed_state.blob"
    alarm_filename: str = "next_wake.txt"
    provider_name: str = "openai-compatible"
    provider_base_url: str = "https://api.openai.com/v1"
    provider_model: str = "gpt-4o-mini"
    provider_api_key: str | None = None
    provider_timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "AuroraSettings":
        _load_env_file(".env")
        _load_env_file(".env.local")
        bailian_present = any(
            _first_env(key) is not None
            for key in (
                "AURORA_BAILIAN_LLM_BASE_URL",
                "AURORA_BAILIAN_LLM_MODEL",
                "AURORA_BAILIAN_LLM_API_KEY",
            )
        )
        provider_name = _first_env(
            "AURORA_PROVIDER_NAME",
            default="bailian" if bailian_present else "openai-compatible",
        )
        return cls(
            data_dir=_first_env("AURORA_DATA_DIR", default=".aurora_seed_v1") or ".aurora_seed_v1",
            sealed_state_filename=_first_env(
                "AURORA_SEALED_STATE_FILE", default="sealed_state.blob"
            )
            or "sealed_state.blob",
            alarm_filename=_first_env("AURORA_ALARM_FILE", default="next_wake.txt")
            or "next_wake.txt",
            provider_name=provider_name or "openai-compatible",
            provider_base_url=_first_env(
                "AURORA_PROVIDER_BASE_URL",
                "AURORA_BAILIAN_LLM_BASE_URL",
                default="https://api.openai.com/v1",
            )
            or "https://api.openai.com/v1",
            provider_model=_first_env(
                "AURORA_PROVIDER_MODEL",
                "AURORA_BAILIAN_LLM_MODEL",
                default="gpt-4o-mini",
            )
            or "gpt-4o-mini",
            provider_api_key=_first_env(
                "AURORA_PROVIDER_API_KEY",
                "AURORA_BAILIAN_LLM_API_KEY",
            ),
            provider_timeout_s=float(
                _first_env(
                    "AURORA_PROVIDER_TIMEOUT_S",
                    "AURORA_LLM_TIMEOUT",
                    default="30.0",
                )
                or "30.0"
            ),
        )

    def provider_fingerprint(self) -> str:
        payload = f"{self.provider_name}|{self.provider_base_url}|{self.provider_model}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
