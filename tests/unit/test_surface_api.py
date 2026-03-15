from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aurora.runtime.engine import AuroraEngine
from aurora.surface.api import build_app

from tests.conftest import StubLLM


def test_health_and_state_endpoints_expose_runtime_projection(tmp_path: Path) -> None:
    app = build_app(engine=AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM()))
    client = TestClient(app)

    health = client.get("/health")
    state = client.get("/state")

    assert health.status_code == 200
    assert state.status_code == 200
    assert health.json()["status"] == "ok"
    assert state.json()["phase"] == "awake"
    assert "sleep_need" in state.json()
    assert "memory_threads" in state.json()


def test_turn_doze_sleep_endpoints_follow_final_surface_paths(tmp_path: Path) -> None:
    app = build_app(engine=AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM()))
    client = TestClient(app)

    turn = client.post("/turn", json={"session_id": "s1", "text": "我想知道为什么"})
    doze = client.post("/doze")
    sleep = client.post("/sleep")

    assert turn.status_code == 200
    assert doze.status_code == 200
    assert sleep.status_code == 200
    assert turn.json()["aurora_move"]
    assert isinstance(turn.json()["dominant_channels"], list)
    assert doze.json()["phase"] == "doze"
    assert sleep.json()["phase"] == "sleep"
