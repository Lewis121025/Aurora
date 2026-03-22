from __future__ import annotations

import importlib
import importlib.util
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import aurora.surfaces.cli as cli_module
import aurora.surfaces.mcp as mcp_module
import aurora.surfaces.http as http_module

from tests.conftest import QueueLLM, SystemFactory


def test_release_surface_uses_new_field_api(system_factory: SystemFactory) -> None:
    system = system_factory(llm=QueueLLM("I will remind you tomorrow.", repeat_last=True))

    inject = getattr(system, "inject")
    read_workspace = getattr(system, "read_workspace")
    maintenance_cycle = getattr(system, "maintenance_cycle")
    respond = getattr(system, "respond")

    inject(
        {
            "payload": "I live in Hangzhou.",
            "session_id": "session-a",
            "turn_id": "turn-1",
            "source": "user",
            "metadata": {"role": "user", "turn_index": 1},
        }
    )
    inject(
        {
            "payload": "I do not live in Shanghai.",
            "session_id": "session-a",
            "turn_id": "turn-2",
            "source": "user",
            "metadata": {"role": "user", "turn_index": 2},
        }
    )
    inject(
        {
            "payload": "I will remind you tomorrow.",
            "session_id": "session-a",
            "turn_id": "turn-3",
            "source": "assistant",
            "metadata": {"role": "assistant", "turn_index": 3},
        }
    )

    stats = maintenance_cycle(ms_budget=12)
    workspace = read_workspace({"payload": "Where do I live?", "session_id": "session-a"}, k=6)
    packet_count_before = len(system.field.anchor_store.packets)
    response = respond(
        {
            "payload": "Please remind me tomorrow.",
            "session_id": "session-a",
            "turn_id": "turn-4",
            "source": "user",
            "metadata": {"role": "user"},
        }
    )
    packet_count_after = len(system.field.anchor_store.packets)

    assert stats.elapsed_ms >= 0
    assert workspace.active_trace_ids
    assert workspace.anchor_refs
    assert response.response_text == "I will remind you tomorrow."
    assert response.workspace.active_trace_ids
    assert response.anchor_ids
    assert response.trace_ids
    assert packet_count_after >= packet_count_before + 2
    assert not hasattr(system.field, "respond")
    assert not hasattr(system.field, "local_decoder")


def test_release_root_exports_are_minimal() -> None:
    aurora_pkg = importlib.import_module("aurora")

    assert hasattr(aurora_pkg, "AuroraSystem")
    assert hasattr(aurora_pkg, "AuroraSystemConfig")
    assert hasattr(aurora_pkg, "AuroraField")
    assert hasattr(aurora_pkg, "FieldConfig")
    assert hasattr(aurora_pkg, "build_app")
    assert not hasattr(aurora_pkg, "build_llm_provider")
    assert not hasattr(aurora_pkg, "to_dict")
    assert not hasattr(aurora_pkg, "RespondRequest")


def test_release_old_top_level_modules_are_gone() -> None:
    assert importlib.util.find_spec("aurora.api") is None
    assert importlib.util.find_spec("aurora.system") is None
    assert importlib.util.find_spec("aurora.cli") is None
    assert importlib.util.find_spec("aurora.mcp") is None
    assert importlib.util.find_spec("aurora.projections") is None
    assert importlib.util.find_spec("aurora.field") is None


def test_release_surface_cli_names_are_new_only() -> None:
    parser = cli_module.build_parser()
    subparsers = cast(dict[str, Any], cast(Any, parser._subparsers)._group_actions[0].choices)

    assert "inject" in subparsers
    assert "read-workspace" in subparsers
    assert "maintenance-cycle" in subparsers
    assert "respond" in subparsers
    assert "snapshot" in subparsers
    assert "field-stats" in subparsers

    assert "append-material" not in subparsers
    assert "build-workspace" not in subparsers
    assert "run-scheduler" not in subparsers
    assert "respond-turn" not in subparsers
    parser.parse_args(["snapshot"])
    with pytest.raises(SystemExit):
        parser.parse_args(["snapshot", "--path", "custom.json"])


def test_release_surface_mcp_tools_are_new_only(system_factory: SystemFactory) -> None:
    system = system_factory(llm=QueueLLM("ok", repeat_last=True))
    mcp_module._system = system

    try:
        assert hasattr(mcp_module, "aurora_inject")
        assert hasattr(mcp_module, "aurora_read_workspace")
        assert hasattr(mcp_module, "aurora_maintenance_cycle")
        assert hasattr(mcp_module, "aurora_respond")
        assert hasattr(mcp_module, "aurora_snapshot")
        assert hasattr(mcp_module, "aurora_field_stats")
        assert not hasattr(mcp_module, "aurora_append_material")
        assert not hasattr(mcp_module, "aurora_build_workspace")
        assert not hasattr(mcp_module, "aurora_run_scheduler")
        assert not hasattr(mcp_module, "aurora_respond_turn")
    finally:
        mcp_module._system = None


def test_release_http_surface_uses_new_routes_only(system_factory: SystemFactory) -> None:
    system = system_factory(llm=QueueLLM("I will remind you tomorrow.", repeat_last=True))
    client = TestClient(http_module.build_app(system))

    inject_response = client.post(
        "/inject",
        json={"payload": "I like tea.", "session_id": "session-a", "turn_id": "turn-1", "source": "user", "metadata": {"role": "user"}},
    )
    workspace_response = client.post(
        "/read-workspace",
        json={"cue": "What do I like?", "session_id": "session-a", "k": 6},
    )
    maintenance_response = client.post("/maintenance-cycle", json={"ms_budget": 12})
    respond_response = client.post(
        "/respond",
        json={
            "cue": "Please remind me tomorrow.",
            "session_id": "session-a",
            "metadata": {"role": "user"},
        },
    )
    snapshot_response = client.post("/snapshot")
    stats_response = client.get("/field-stats")

    assert inject_response.status_code == 200
    assert workspace_response.status_code == 200
    assert maintenance_response.status_code == 200
    assert respond_response.status_code == 200
    assert snapshot_response.status_code == 200
    assert stats_response.status_code == 200
    assert "memory_brief" not in respond_response.text

    for old_route in ("/append-material", "/build-workspace", "/run-scheduler", "/respond-turn"):
        response = client.post(old_route, json={})
        assert response.status_code == 404
