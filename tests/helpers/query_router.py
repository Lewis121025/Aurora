from __future__ import annotations

import os
from typing import Dict

import pytest

from aurora.integrations.llm.bailian import BailianLLM
from aurora.integrations.llm.provider import LLMProvider
from aurora.soul.query import BaseQueryAnalyzer, QueryAnalysis, QueryAnalyzer


def build_test_llm() -> LLMProvider:
    pytest.importorskip("openai")
    api_key = os.environ.get("AURORA_BAILIAN_LLM_API_KEY")
    if not api_key:
        pytest.skip("Real Bailian LLM tests require AURORA_BAILIAN_LLM_API_KEY")
    return BailianLLM(
        api_key=api_key,
        model=os.environ.get("AURORA_BAILIAN_LLM_MODEL", "qwen3.5-plus"),
        base_url=os.environ.get(
            "AURORA_BAILIAN_LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        max_retries=1,
        timeout=min(float(os.environ.get("AURORA_TEST_LLM_TIMEOUT", "20")), 20.0),
    )


class CachedQueryAnalyzer(BaseQueryAnalyzer):
    """Wrap a live analyzer to keep real-network test calls bounded."""

    def __init__(self, inner: BaseQueryAnalyzer) -> None:
        self._inner = inner
        self._cache: Dict[str, QueryAnalysis] = {}

    def analyze(self, query_text: str) -> QueryAnalysis:
        cached = self._cache.get(query_text)
        if cached is not None:
            return cached
        analysis = self._inner.analyze(query_text)
        self._cache[query_text] = analysis
        return analysis


def build_test_query_analyzer() -> BaseQueryAnalyzer:
    return CachedQueryAnalyzer(QueryAnalyzer(llm=build_test_llm(), timeout_s=5.0, max_retries=1))
