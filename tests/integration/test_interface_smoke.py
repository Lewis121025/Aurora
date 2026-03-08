from __future__ import annotations

import pytest

from aurora import AuroraMemory, MemoryConfig
from aurora.interfaces import cli
from aurora.interfaces.mcp import AuroraMCPServer


def test_cli_module_imports():
    assert hasattr(cli, 'main')


def test_mcp_module_imports():
    assert AuroraMCPServer is not None


def test_default_memory_uses_semantic_embedder():
    memory = AuroraMemory(cfg=MemoryConfig(dim=64, max_plots=10), seed=42)
    assert not memory.is_using_hash_embedding()
    plot = memory.ingest('用户：你好 助手：你好', actors=('user', 'assistant'))
    assert plot.embedding.shape == (64,)


def test_api_app_imports_when_fastapi_available():
    pytest.importorskip('fastapi')
    from aurora.interfaces.api.app import app

    assert app.title == 'AURORA Memory API'
