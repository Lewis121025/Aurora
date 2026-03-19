from __future__ import annotations

from pathlib import Path

import pytest

from aurora.llm.config import coerce_llm_settings, load_llm_settings
from aurora.runtime.engine import AuroraKernel


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


def test_load_llm_settings_applies_bailian_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "AURORA_LLM_PROVIDER",
        "AURORA_LLM_CONFIG_BASE_URL",
        "AURORA_LLM_CONFIG_MODEL",
        "AURORA_LLM_CONFIG_API_KEY",
        "AURORA_LLM_CONFIG_ENABLE_THINKING",
    ):
        monkeypatch.delenv(name, raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            (
                "AURORA_LLM_PROVIDER=bailian",
                "AURORA_LLM_CONFIG_BASE_URL=https://example.provider/v1",
                "AURORA_LLM_CONFIG_MODEL=bailian-model",
                "AURORA_LLM_CONFIG_API_KEY=provider-key",
            )
        ),
        encoding="utf-8",
    )

    settings = load_llm_settings()

    assert settings is not None
    assert settings.provider == "bailian"
    assert settings.config.base_url == "https://example.provider/v1"
    assert settings.config.model == "bailian-model"
    assert settings.config.api_key == "provider-key"
    assert settings.config.enable_thinking is False


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


def test_kernel_create_accepts_llm_settings_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[object] = []

    class FakeProvider:
        def __init__(self, config: object) -> None:
            captured.append(config)

        def complete(self, messages: list[dict[str, str]]) -> str:
            return "ok"

    monkeypatch.setattr("aurora.runtime.engine.OpenAICompatProvider", FakeProvider)

    kernel = AuroraKernel.create(
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
        assert len(captured) == 1
    finally:
        kernel.close()


def test_kernel_create_rejects_llm_and_llm_settings_together(tmp_path: Path) -> None:
    class DummyLLM:
        def complete(self, messages: list[dict[str, str]]) -> str:
            return "ok"

    with pytest.raises(ValueError, match="either llm or llm_settings"):
        AuroraKernel.create(
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
