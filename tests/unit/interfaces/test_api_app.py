from __future__ import annotations

from types import SimpleNamespace

import pytest

from aurora.runtime.results import (
    ChatTimings,
    ChatTurnResult,
    EvidenceRef,
    IngestResult,
    RetrievalTraceSummary,
    StructuredMemoryContext,
)


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from aurora.interfaces.api import app as api_module


class FakeRuntime:
    def respond(self, **kwargs):
        return ChatTurnResult(
            reply="memory-first reply",
            event_id="evt_api_001",
            memory_context=StructuredMemoryContext(
                known_facts=["user prefers 长期记忆系统"],
                preferences=["偏好长期记忆系统"],
                relationship_state=[],
                active_narratives=[],
                temporal_context=[],
                cautions=[],
                evidence_refs=[EvidenceRef(id="plot-1", kind="plot", score=0.9, role="current_fact")],
            ),
            rendered_memory_brief="[Known Facts]\n- user prefers 长期记忆系统",
            system_prompt="system prompt",
            user_prompt="Memory Brief:\n[Known Facts]\n- user prefers 长期记忆系统",
            retrieval_trace_summary=RetrievalTraceSummary(
                query="你记得我喜欢什么吗？",
                query_type="FACTUAL",
                attractor_path_len=1,
                hit_count=1,
                timeline_count=0,
                standalone_count=0,
                abstain=False,
            ),
            ingest_result=IngestResult(
                event_id="evt_api_001",
                plot_id="plot-1",
                story_id="story-1",
                encoded=True,
                tension=0.4,
                surprise=0.2,
                pred_error=0.1,
                redundancy=0.05,
            ),
            timings=ChatTimings(
                retrieval_ms=1.0,
                generation_ms=2.0,
                ingest_ms=3.0,
                total_ms=6.0,
            ),
            llm_error=None,
        )

    def query(self, *, text: str, k: int = 8):
        return SimpleNamespace(
            query=text,
            attractor_path_len=1,
            hits=[],
        )


def test_respond_endpoint_returns_structured_memory_context(monkeypatch):
    api_module.get_runtime.cache_clear()
    monkeypatch.setattr(api_module, "get_runtime", lambda: FakeRuntime())
    client = TestClient(api_module.app)

    response = client.post(
        "/v1/memory/respond",
        json={
            "session_id": "chat_api",
            "user_message": "你记得我喜欢什么吗？",
            "k": 6,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "memory-first reply"
    assert payload["memory_context"]["known_facts"]
    assert payload["memory_context"]["evidence_refs"][0]["id"] == "plot-1"
    assert "USER:" not in payload["user_prompt"]


def test_query_endpoint_remains_low_level(monkeypatch):
    api_module.get_runtime.cache_clear()
    monkeypatch.setattr(api_module, "get_runtime", lambda: FakeRuntime())
    client = TestClient(api_module.app)

    response = client.post(
        "/v1/memory/query",
        json={"text": "长期记忆系统", "k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "长期记忆系统"
    assert payload["hits"] == []
