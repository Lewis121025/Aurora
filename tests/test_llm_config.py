from __future__ import annotations

from pathlib import Path

import pytest

from aurora.llm.config import load_llm_config


def test_load_llm_config_reads_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AURORA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AURORA_LLM_MODEL", raising=False)
    monkeypatch.delenv("AURORA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("AURORA_LLM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("AURORA_LLM_MAX_TOKENS", raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            (
                "AURORA_LLM_BASE_URL=https://example.test/v1",
                "AURORA_LLM_MODEL=test-model",
                "AURORA_LLM_API_KEY=test-key",
                "AURORA_LLM_TIMEOUT_S=12.5",
                "AURORA_LLM_MAX_TOKENS=256",
            )
        ),
        encoding="utf-8",
    )

    config = load_llm_config()

    assert config is not None
    assert config.base_url == "https://example.test/v1"
    assert config.model == "test-model"
    assert config.api_key == "test-key"
    assert config.timeout_s == 12.5
    assert config.max_tokens == 256
    assert config.enable_thinking is None


def test_load_llm_config_reads_provider_scoped_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AURORA_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AURORA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AURORA_LLM_MODEL", raising=False)
    monkeypatch.delenv("AURORA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("AURORA_LLM_TIMEOUT", raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            (
                "AURORA_LLM_PROVIDER=bailian",
                "AURORA_BAILIAN_LLM_BASE_URL=https://example.provider/v1",
                "AURORA_BAILIAN_LLM_MODEL=bailian-model",
                "AURORA_BAILIAN_LLM_API_KEY=provider-key",
                "AURORA_LLM_TIMEOUT=18.0",
            )
        ),
        encoding="utf-8",
    )

    config = load_llm_config()

    assert config is not None
    assert config.base_url == "https://example.provider/v1"
    assert config.model == "bailian-model"
    assert config.api_key == "provider-key"
    assert config.timeout_s == 18.0
    assert config.enable_thinking is False
