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
from aurora.runtime.contracts import ActivatedAtom, ActivatedEdge, RecallResult, SubjectMemoryState, TurnOutput

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
        summary="[CURRENT_MEMORY_FIELD]\n- memory (0.910): 我现在住在杭州",
        atoms=(
            ActivatedAtom(
                atom_id="atom-memory-1",
                atom_kind=cast(Any, "memory"),
                text="我现在住在杭州",
                activation=0.91,
                confidence=0.92,
                salience=0.88,
                created_at=1.0,
            ),
            ActivatedAtom(
                atom_id="atom-episode-1",
                atom_kind=cast(Any, "episode"),
                text="在杭州重新适应生活节奏",
                activation=0.74,
                confidence=0.78,
                salience=0.80,
                created_at=2.0,
            ),
        ),
        edges=(
            ActivatedEdge(
                source_atom_id="atom-memory-1",
                target_atom_id="atom-episode-1",
                influence=0.42,
                confidence=0.76,
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
        self.recall_calls: list[tuple[str, str, int]] = []
        self.state_calls: list[str] = []

    def turn(self, subject_id: str, text: str, now_ts: float | None = None) -> TurnOutput:
        self.turn_calls.append((subject_id, text, now_ts))
        return TurnOutput(
            turn_id="turn-1",
            subject_id=subject_id,
            response_text="ack",
            recall_used=True,
            created_atom_ids=("atom-1", "atom-2"),
            created_edge_ids=("edge-1",),
        )

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        limit: int = 8,
    ) -> RecallResult:
        self.recall_calls.append((subject_id, query, limit))
        return RecallResult(
            subject_id=subject_id,
            query=query,
            summary="[QUERY_MEMORY_FIELD]\n- memory (0.930): 我现在住在杭州",
            atoms=(
                ActivatedAtom(
                    atom_id="atom-memory-1",
                    atom_kind=cast(Any, "memory"),
                    text="我现在住在杭州",
                    activation=0.93,
                    confidence=0.92,
                    salience=0.88,
                    created_at=1.0,
                ),
            ),
            edges=(
                ActivatedEdge(
                    source_atom_id="atom-memory-1",
                    target_atom_id="atom-episode-1",
                    influence=0.38,
                    confidence=0.75,
                ),
            ),
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
    assert "created_atom_ids" in turn_tool.outputSchema["properties"]
    assert "created_edge_ids" in turn_tool.outputSchema["properties"]

    recall_tool = next(tool for tool in tools if tool.name == "aurora_recall")
    assert recall_tool.inputSchema["required"] == ["subject_id", "query"]
    assert "limit" in recall_tool.inputSchema["properties"]
    assert "summary" in recall_tool.outputSchema["properties"]
    assert "atoms" in recall_tool.outputSchema["properties"]
    assert "edges" in recall_tool.outputSchema["properties"]

    templates = cast(list[Any], _run(mcp_module.mcp.list_resource_templates()))
    assert [template.name for template in templates] == ["aurora_subject_memory_field"]
    assert [template.uriTemplate for template in templates] == ["aurora://subject/{subject_id}/memory-field"]
    assert _run(mcp_module.mcp.list_resources()) == []


def test_surface_contract_turn_recall_and_resources_share_memory_field(monkeypatch: pytest.MonkeyPatch) -> None:
    subject_id = "subject-mcp-field"
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
        "recall_used": True,
        "created_atom_ids": ["atom-1", "atom-2"],
        "created_edge_ids": ["edge-1"],
    }

    recall_blocks, recall_result = cast(
        tuple[list[Any], dict[str, Any]],
        _run(
            mcp_module.mcp.call_tool(
                "aurora_recall",
                {
                    "subject_id": subject_id,
                    "query": "我住在哪里？",
                    "limit": 5,
                },
            )
        ),
    )
    assert recall_blocks
    assert recall_result == _jsonable(kernel.recall(subject_id, "我住在哪里？", limit=5))

    state_payload = _json_resource_payload(f"aurora://subject/{subject_id}/memory-field")
    assert state_payload == _jsonable(_subject_state(subject_id))
    assert "active_cognition" not in json.dumps(state_payload, ensure_ascii=False)


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


def test_api_request_models_reject_whitespace() -> None:
    with pytest.raises(ValidationError):
        api_module.TurnRequest(subject_id="   ", text="ok")
    with pytest.raises(ValidationError):
        api_module.TurnRequest(subject_id="subject", text="   ")
    with pytest.raises(ValidationError):
        api_module.RecallRequest.model_validate({"subject_id": "subject"})
    with pytest.raises(ValidationError):
        api_module.RecallRequest(subject_id="   ", query="杭州")
    with pytest.raises(ValidationError):
        api_module.RecallRequest(subject_id="subject", query="   ")


def test_http_state_surface_uses_field_contract_dataclasses() -> None:
    client = TestClient(api_module.build_app(cast(api_module.SurfaceKernel, _StrictKernel())))

    response = client.get("/state/subject-http")

    assert response.status_code == 200
    assert response.json() == _jsonable(_subject_state("subject-http"))


def test_http_recall_surface_passes_limit_through() -> None:
    kernel = _StrictKernel()
    client = TestClient(api_module.build_app(cast(api_module.SurfaceKernel, kernel)))

    missing_query = client.post(
        "/recall",
        json={"subject_id": "subject-http"},
    )
    response = client.post(
        "/recall",
        json={
            "subject_id": "subject-http",
            "query": "我住在哪里？",
            "limit": 3,
        },
    )

    assert missing_query.status_code == 422
    assert response.status_code == 200
    assert kernel.recall_calls == [("subject-http", "我住在哪里？", 3)]
    assert response.json() == _jsonable(
        kernel.recall("subject-http", "我住在哪里？", limit=3)
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
    ],
)
def test_cli_requires_explicit_subject_id(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        cli_module.main()


def test_cli_uses_field_contract_dataclasses(
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
        ],
    )
    cli_module.main()
    recall_output = json.loads(capsys.readouterr().out)
    assert recall_output == _jsonable(
        kernel.recall("subject-cli", "杭州", limit=8)
    )
