from __future__ import annotations

import pytest

from aurora.system.errors import ConfigurationError
from aurora.integrations.llm.mock import MockLLM
from aurora.runtime.bootstrap import create_embedding_provider, create_llm_provider
from aurora.runtime.settings import AuroraSettings


def test_create_llm_provider_uses_mock_by_default():
    settings = AuroraSettings(llm_provider="mock")
    provider = create_llm_provider(settings)
    assert isinstance(provider, MockLLM)


def test_create_llm_provider_requires_bailian_api_key():
    settings = AuroraSettings(llm_provider="bailian", bailian_llm_api_key=None)
    with pytest.raises(ConfigurationError):
        create_llm_provider(settings)


def test_create_embedding_provider_requires_bailian_api_key():
    settings = AuroraSettings(embedding_provider="bailian", bailian_embedding_api_key=None)
    with pytest.raises(ConfigurationError):
        create_embedding_provider(settings)
