from __future__ import annotations

from pathlib import Path

import pytest

from aurora.llm.config import coerce_llm_settings, load_llm_settings
from aurora.runtime import AuroraSystem


def test_load_llm_settings_reads_nested_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "AURORA_LLM_PROVIDER",
        "AURORA_LLM_CONFIG_BASE_URL",
        "AURORA_LLM_CONFIG_MODEL",
        "AURORA_LLM_CONFIG_API_KEY",
        "AURORA_LLM_CONFIG_TIMEOUT_S",
        "AURORA_LLM_CONFIG_MAX_TOKENS",
        "AURORA_LLM_CONFIG_ENABLE_THINKING",
    ):
        monkeypatch.delenv(name, raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            (
                "AURORA_LLM_PROVIDER=openai_compatible",
                "AURORA_LLM_CONFIG_BASE_URL=https://example.test/v1",
                "AURORA_LLM_CONFIG_MODEL=test-model",
                "AURORA_LLM_CONFIG_API_KEY=test-key",
                "AURORA_LLM_CONFIG_TIMEOUT_S=12.5",
                "AURORA_LLM_CONFIG_MAX_TOKENS=256",
                "AURORA_LLM_CONFIG_ENABLE_THINKING=true",
            )
        ),
        encoding="utf-8",
    )

    settings = load_llm_settings()

    assert settings is not None
    assert settings.provider == "openai_compatible"
    assert settings.config.base_url == "https://example.test/v1"
    assert settings.config.model == "test-model"
    assert settings.config.api_key == "test-key"
    assert settings.config.timeout_s == 12.5
    assert settings.config.max_tokens == 256
    assert settings.config.enable_thinking is True


def test_coerce_llm_settings_reads_nested_mapping() -> None:
    settings = coerce_llm_settings(
        {
            "provider": "openai",
            "config": {
                "base_url": "https://example.test/v1",
                "model": "gpt-test",
                "api_key": "secret",
                "timeout_s": 6,
                "max_tokens": 128,
            },
        }
    )

    assert settings.provider == "openai"
    assert settings.config.base_url == "https://example.test/v1"
    assert settings.config.model == "gpt-test"
    assert settings.config.api_key == "secret"
    assert settings.config.timeout_s == 6.0
    assert settings.config.max_tokens == 128
    assert settings.config.enable_thinking is None


def test_coerce_llm_settings_rejects_missing_config_fields() -> None:
    with pytest.raises(ValueError, match="base_url and api_key"):
        coerce_llm_settings(
            {
                "provider": "openai",
                "config": {
                    "model": "gpt-test",
                },
            }
        )


def test_system_create_accepts_llm_settings_mapping_lazily(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[object] = []

    class FakeProvider:
        def __init__(self, config: object) -> None:
            captured.append(config)

        def complete(
            self,
            messages: list[dict[str, str]],
            *,
            max_tokens: int | None = None,
            temperature: float | None = None,
        ) -> str:
            del messages, max_tokens, temperature
            return "ok"

    monkeypatch.setattr("aurora.runtime.system.OpenAICompatProvider", FakeProvider)

    system = AuroraSystem.create(
        data_dir=str(tmp_path / ".aurora"),
        llm_settings={
            "provider": "openai",
            "config": {
                "base_url": "https://example.test/v1",
                "model": "gpt-test",
                "api_key": "secret",
            },
        },
    )
    try:
        assert len(captured) == 0
        assert system._get_responder() is not None
        assert len(captured) == 1
    finally:
        system.close()


def test_system_create_defers_invalid_env_provider_until_respond(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AURORA_LLM_PROVIDER", "unsupported")
    monkeypatch.setenv("AURORA_LLM_CONFIG_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("AURORA_LLM_CONFIG_API_KEY", "secret")

    system = AuroraSystem.create(data_dir=str(tmp_path / ".aurora"))
    try:
        assert system.inject("hello").packet_ids
        with pytest.raises(RuntimeError, match="Unsupported AURORA_LLM_PROVIDER"):
            system.respond({"payload": "hello", "session_id": "session-a"})
    finally:
        system.close()


def test_system_create_defers_invalid_env_settings_until_respond(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AURORA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("AURORA_LLM_CONFIG_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("AURORA_LLM_CONFIG_API_KEY", "secret")
    monkeypatch.setenv("AURORA_LLM_CONFIG_TIMEOUT_S", "not-a-float")

    system = AuroraSystem.create(data_dir=str(tmp_path / ".aurora"))
    try:
        assert system.inject("hello").packet_ids
        with pytest.raises(ValueError, match="could not convert string to float"):
            system.respond({"payload": "hello", "session_id": "session-a"})
    finally:
        system.close()


def test_system_create_rejects_llm_and_llm_settings_together(tmp_path: Path) -> None:
    class DummyLLM:
        def complete(
            self,
            messages: list[dict[str, str]],
            *,
            max_tokens: int | None = None,
            temperature: float | None = None,
        ) -> str:
            del messages, max_tokens, temperature
            return "ok"

    with pytest.raises(ValueError, match="either llm or llm_settings"):
        AuroraSystem.create(
            data_dir=str(tmp_path / ".aurora"),
            llm=DummyLLM(),
            llm_settings={
                "provider": "openai",
                "config": {
                    "base_url": "https://example.test/v1",
                    "api_key": "secret",
                },
            },
        )
