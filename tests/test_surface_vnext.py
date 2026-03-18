from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import asdict
import json
import sys
from typing import Any, TypeVar, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import aurora.surface.api as api_module
import aurora.surface.cli as cli_module

pytest.importorskip("mcp.server.fastmcp")

import aurora.surface.mcp as mcp_module
from aurora.runtime.contracts import (
    ActiveCognition,
    AffectMarker,
    AffectiveState,
    EpisodeMemoryItem,
    NarrativeArc,
    NarrativeState,
    RecallHit,
    RecallMode,
    RecallResult,
    SemanticMemoryItem,
    SubjectMemoryState,
    TimeSpan,
    TurnOutput,
)

T = TypeVar("T")


def _run(awaitable: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(awaitable)


def _json_resource_payload(uri: str) -> dict[str, Any]:
    contents = cast(list[Any], _run(mcp_module.mcp.read_resource(uri)))
    assert len(contents) == 1
    return cast(dict[str, Any], json.loads(contents[0].content))


def _jsonable(value: object) -> Any:
    return json.loads(json.dumps(asdict(cast(Any, value)), ensure_ascii=False))


def _subject_state(subject_id: str) -> SubjectMemoryState:
    return SubjectMemoryState(
        subject_id=subject_id,
        semantic_self_model=(
            SemanticMemoryItem(
                subject="我",
                attribute="location.current",
                value="杭州",
                text="我现在住在杭州",
            ),
        ),
        semantic_world_model=(),
        procedural_memory=(),
        active_cognition=ActiveCognition(
            beliefs=("我住在杭州",),
            goals=("补完部署文档",),
            conflicts=("时间不够",),
            intentions=("今晚先写",),
            commitments=("明天同步团队",),
        ),
        affective_state=AffectiveState(
            mood="positive",
            valence=0.7,
            intensity=0.8,
            active_feelings=("开心",),
        ),
        narrative_state=NarrativeState(
            arcs=(
                NarrativeArc(
                    theme="在杭州重建生活",
                    storyline="逐步适应新的工作和生活节奏",
                    status="active",
                    episode_count=2,
                    unresolved_threads=("工作节奏",),
                    role_changes=("new resident",),
                    updated_at=1.0,
                ),
            ),
            active_themes=("在杭州重建生活",),
        ),
        recent_episodes=(
            EpisodeMemoryItem(
                title="杭州近况",
                summary="用户在杭州工作并感到开心",
                actors=("user", "aurora"),
                setting="杭州",
                time_span=TimeSpan(start=1.0, end=1.0),
                emotion_markers=(AffectMarker(label="开心", intensity=0.8, valence=0.7),),
                text="用户在杭州工作并感到开心",
            ),
        ),
    )


class _Store:
    def __init__(self, subjects: int = 1) -> None:
        self._subjects = subjects

    def subject_count(self) -> int:
        return self._subjects


class _StrictKernel:
    def __init__(self, *, subjects: int = 1) -> None:
        self.closed = False
        self.store = _Store(subjects)
        self.turn_calls: list[tuple[str, str, float | None]] = []
        self.recall_calls: list[tuple[str, str, str, int, RecallMode]] = []
        self.state_calls: list[str] = []

    def turn(self, subject_id: str, text: str, now_ts: float | None = None) -> TurnOutput:
        self.turn_calls.append((subject_id, text, now_ts))
        return TurnOutput(
            turn_id="turn-1",
            subject_id=subject_id,
            response_text="ack",
            recall_used=False,
            applied_atom_ids=("atom-1",),
        )

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        temporal_scope: str,
        limit: int = 5,
        mode: RecallMode = "blended",
    ) -> RecallResult:
        self.recall_calls.append((subject_id, query, temporal_scope, limit, mode))
        hits_by_scope = {
            "current": (
                RecallHit(
                    memory_kind="semantic",
                    content="我住在杭州",
                    score=0.91,
                    why_recalled="lexical+current",
                ),
                RecallHit(
                    memory_kind="episode",
                    content="用户在杭州工作并感到开心",
                    score=0.84,
                    why_recalled="vector+current",
                ),
            ),
            "historical": (
                RecallHit(
                    memory_kind="semantic",
                    content="我以前住在上海",
                    score=0.90,
                    why_recalled="lexical+historical",
                ),
            ),
            "both": (
                RecallHit(
                    memory_kind="semantic",
                    content="我住在杭州",
                    score=0.91,
                    why_recalled="lexical+current",
                ),
                RecallHit(
                    memory_kind="semantic",
                    content="我以前住在上海",
                    score=0.88,
                    why_recalled="lexical",
                ),
            ),
        }
        return RecallResult(
            subject_id=subject_id,
            query=query,
            temporal_scope=cast(Any, temporal_scope),
            mode=mode,
            hits=hits_by_scope[temporal_scope][:limit],
        )

    def state(self, subject_id: str) -> SubjectMemoryState:
        self.state_calls.append(subject_id)
        return _subject_state(subject_id)

    def close(self) -> None:
        self.closed = True


def test_surface_contract_registers_tools_and_resources() -> None:
    tools = cast(list[Any], _run(mcp_module.mcp.list_tools()))
    assert [tool.name for tool in tools] == ["aurora_turn", "aurora_recall"]

    turn_tool = next(tool for tool in tools if tool.name == "aurora_turn")
    assert turn_tool.inputSchema["required"] == ["subject_id", "text"]
    assert "now_ts" in turn_tool.inputSchema["properties"]
    assert turn_tool.outputSchema["required"] == [
        "turn_id",
        "subject_id",
        "response_text",
        "recall_used",
    ]
    assert "recalled_ids" not in turn_tool.outputSchema["properties"]
    assert "applied_atom_ids" in turn_tool.outputSchema["properties"]

    recall_tool = next(tool for tool in tools if tool.name == "aurora_recall")
    assert recall_tool.inputSchema["required"] == ["subject_id", "query", "temporal_scope"]
    assert "limit" in recall_tool.inputSchema["properties"]
    assert "mode" in recall_tool.inputSchema["properties"]
    assert recall_tool.outputSchema["required"] == [
        "subject_id",
        "query",
        "temporal_scope",
        "mode",
        "hits",
    ]

    templates = cast(list[Any], _run(mcp_module.mcp.list_resource_templates()))
    assert [template.name for template in templates] == ["aurora_subject_state"]
    assert [template.uriTemplate for template in templates] == ["aurora://subject/{subject_id}/state"]
    assert _run(mcp_module.mcp.list_resources()) == []


def test_surface_contract_turn_recall_and_resources_share_subject_state(monkeypatch: pytest.MonkeyPatch) -> None:
    subject_id = "subject-mcp-vnext"
    kernel = _StrictKernel()
    monkeypatch.setattr(mcp_module, "_get_kernel", lambda: kernel)

    turn_blocks, turn_result = cast(
        tuple[list[Any], dict[str, Any]],
        _run(
            mcp_module.mcp.call_tool(
                "aurora_turn",
                {"subject_id": subject_id, "text": "我在杭州工作，也喜欢爵士乐。", "now_ts": 1.0},
            )
        ),
    )
    assert turn_blocks
    assert turn_result == {
        "turn_id": "turn-1",
        "subject_id": subject_id,
        "response_text": "ack",
        "recall_used": False,
        "applied_atom_ids": ["atom-1"],
    }

    recall_blocks, recall_result = cast(
        tuple[list[Any], dict[str, Any]],
        _run(
            mcp_module.mcp.call_tool(
                "aurora_recall",
                {
                    "subject_id": subject_id,
                    "query": "我以前住在哪里？",
                    "temporal_scope": "historical",
                    "limit": 5,
                    "mode": "blended",
                },
            )
        ),
    )
    assert recall_blocks
    assert recall_result["subject_id"] == subject_id
    assert recall_result["temporal_scope"] == "historical"
    assert recall_result["hits"] == [
        {
            "memory_kind": "semantic",
            "content": "我以前住在上海",
            "score": 0.9,
            "why_recalled": "lexical+historical",
        }
    ]

    state_payload = _json_resource_payload(f"aurora://subject/{subject_id}/state")
    assert state_payload == _jsonable(_subject_state(subject_id))
    assert "memory_atoms" not in state_payload
    assert "atom_id" not in json.dumps(state_payload, ensure_ascii=False)


def test_surface_main_runs_stdio_and_closes_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MainKernel:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_kernel = _MainKernel()
    calls: list[str] = []

    monkeypatch.setattr(mcp_module, "_kernel", None)
    monkeypatch.setattr(
        mcp_module.AuroraKernel,
        "create",
        staticmethod(lambda *args, **kwargs: fake_kernel),
    )
    monkeypatch.setattr(mcp_module.mcp, "run", lambda transport="stdio": calls.append(transport))

    mcp_module.main()

    assert calls == ["stdio"]
    assert fake_kernel.closed is True
    assert mcp_module._kernel is None


def test_api_request_models_require_temporal_scope_and_reject_whitespace() -> None:
    with pytest.raises(ValidationError):
        api_module.TurnRequest(subject_id="   ", text="ok")
    with pytest.raises(ValidationError):
        api_module.TurnRequest(subject_id="subject", text="   ")
    with pytest.raises(ValidationError):
        api_module.RecallRequest.model_validate({"subject_id": "subject", "query": "杭州"})
    with pytest.raises(ValidationError):
        api_module.RecallRequest(subject_id="   ", query="杭州", temporal_scope="current")
    with pytest.raises(ValidationError):
        api_module.RecallRequest(subject_id="subject", query="   ", temporal_scope="current")


def test_http_state_surface_uses_kernel_contract_dataclasses() -> None:
    client = TestClient(api_module.build_app(cast(api_module.SurfaceKernel, _StrictKernel())))

    response = client.get("/state/subject-http")

    assert response.status_code == 200
    assert response.json() == _jsonable(_subject_state("subject-http"))


def test_http_recall_surface_requires_scope_and_passes_it_through() -> None:
    kernel = _StrictKernel()
    client = TestClient(api_module.build_app(cast(api_module.SurfaceKernel, kernel)))

    missing_scope = client.post(
        "/recall",
        json={"subject_id": "subject-http", "query": "我住在哪里？"},
    )
    response = client.post(
        "/recall",
        json={
            "subject_id": "subject-http",
            "query": "我以前住在哪里？",
            "temporal_scope": "historical",
            "limit": 3,
            "mode": "blended",
        },
    )

    assert missing_scope.status_code == 422
    assert response.status_code == 200
    assert kernel.recall_calls == [("subject-http", "我以前住在哪里？", "historical", 3, "blended")]
    assert response.json() == _jsonable(
        RecallResult(
            subject_id="subject-http",
            query="我以前住在哪里？",
            temporal_scope="historical",
            mode="blended",
            hits=(
                RecallHit(
                    memory_kind="semantic",
                    content="我以前住在上海",
                    score=0.9,
                    why_recalled="lexical+historical",
                ),
            ),
        )
    )


def test_http_auth_guard_keeps_health_open_and_protects_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURORA_API_KEY", "secret-token")
    client = TestClient(api_module.build_app(cast(api_module.SurfaceKernel, _StrictKernel(subjects=3))))

    health = client.get("/health")
    unauthorized = client.get("/state/protected")
    authorized = client.get("/state/protected", headers={"Authorization": "Bearer secret-token"})

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "subjects": 3}
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


@pytest.mark.parametrize(
    "argv",
    [
        ["aurora", "turn", "hello"],
        ["aurora", "state"],
        ["aurora", "recall", "杭州"],
        ["aurora", "recall", "杭州", "--subject-id", "subject-cli"],
    ],
)
def test_cli_requires_explicit_subject_id_and_temporal_scope(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        cli_module.main()


def test_cli_uses_contract_dataclasses_for_state_and_recall(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    kernel = _StrictKernel()
    monkeypatch.setattr(cli_module.AuroraKernel, "create", staticmethod(lambda: kernel))

    monkeypatch.setattr(sys, "argv", ["aurora", "state", "--subject-id", "subject-cli"])
    cli_module.main()
    state_output = json.loads(capsys.readouterr().out)
    assert state_output == _jsonable(_subject_state("subject-cli"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aurora",
            "recall",
            "杭州",
            "--subject-id",
            "subject-cli",
            "--temporal-scope",
            "both",
        ],
    )
    cli_module.main()
    recall_output = json.loads(capsys.readouterr().out)
    assert recall_output == _jsonable(
        RecallResult(
            subject_id="subject-cli",
            query="杭州",
            temporal_scope="both",
            mode="blended",
            hits=(
                RecallHit(
                    memory_kind="semantic",
                    content="我住在杭州",
                    score=0.91,
                    why_recalled="lexical+current",
                ),
                RecallHit(
                    memory_kind="semantic",
                    content="我以前住在上海",
                    score=0.88,
                    why_recalled="lexical",
                ),
            ),
        )
    )
