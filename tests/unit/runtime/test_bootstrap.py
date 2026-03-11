from __future__ import annotations

import pytest

from aurora.system.errors import ConfigurationError
from aurora.integrations.llm.bailian import BailianLLM
from aurora.runtime.bootstrap import (
    create_embedding_provider,
    create_llm_provider,
    create_query_analyzer,
)
from aurora.runtime.settings import AuroraSettings
from aurora.soul.query import QueryAnalyzer
from tests.helpers.query_router import build_test_llm


def test_create_llm_provider_returns_none_when_explicitly_disabled():
    settings = AuroraSettings(llm_provider=None)
    provider = create_llm_provider(settings)
    assert provider is None


def test_default_llm_provider_is_bailian():
    settings = AuroraSettings()
    assert settings.llm_provider == "bailian"


def test_create_llm_provider_requires_bailian_api_key():
    settings = AuroraSettings(llm_provider="bailian", bailian_llm_api_key=None)
    with pytest.raises(ConfigurationError):
        create_llm_provider(settings)


def test_create_llm_provider_uses_bailian_provider():
    settings = AuroraSettings(llm_provider="bailian", bailian_llm_api_key="dummy")
    provider = create_llm_provider(settings)
    assert isinstance(provider, BailianLLM)


def test_create_embedding_provider_requires_bailian_api_key():
    settings = AuroraSettings(embedding_provider="bailian", bailian_embedding_api_key=None)
    with pytest.raises(ConfigurationError):
        create_embedding_provider(settings)


def test_create_query_analyzer_requires_live_llm():
    settings = AuroraSettings(llm_provider=None)
    with pytest.raises(ConfigurationError):
        create_query_analyzer(settings=settings, llm=None)


def test_create_query_analyzer_uses_injected_llm():
    settings = AuroraSettings(llm_provider=None)
    analyzer = create_query_analyzer(settings=settings, llm=build_test_llm())
    assert isinstance(analyzer, QueryAnalyzer)
