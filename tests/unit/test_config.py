from __future__ import annotations

from aurora.host_runtime.config import AuroraSettings


def test_settings_support_generic_provider_vars(monkeypatch) -> None:
    monkeypatch.setenv("AURORA_PROVIDER_NAME", "openai-compatible")
    monkeypatch.setenv("AURORA_PROVIDER_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("AURORA_PROVIDER_MODEL", "model-a")
    monkeypatch.setenv("AURORA_PROVIDER_API_KEY", "token-a")
    settings = AuroraSettings.from_env()
    assert settings.provider_base_url == "https://example.com/v1"
    assert settings.provider_model == "model-a"
    assert settings.provider_api_key == "token-a"


def test_settings_fallback_to_bailian_vars(monkeypatch) -> None:
    monkeypatch.delenv("AURORA_PROVIDER_BASE_URL", raising=False)
    monkeypatch.delenv("AURORA_PROVIDER_MODEL", raising=False)
    monkeypatch.delenv("AURORA_PROVIDER_API_KEY", raising=False)
    monkeypatch.setenv("AURORA_BAILIAN_LLM_BASE_URL", "https://dashscope.example/v1")
    monkeypatch.setenv("AURORA_BAILIAN_LLM_MODEL", "qwen-seed")
    monkeypatch.setenv("AURORA_BAILIAN_LLM_API_KEY", "token-b")
    settings = AuroraSettings.from_env()
    assert settings.provider_name == "bailian"
    assert settings.provider_base_url == "https://dashscope.example/v1"
    assert settings.provider_model == "qwen-seed"
    assert settings.provider_api_key == "token-b"
