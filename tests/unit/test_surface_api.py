from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aurora.runtime.engine import AuroraEngine
from aurora.surface.api import build_app


def test_health_and_state_endpoints(tmp_path: Path) -> None:
    app = build_app(engine=AuroraEngine.create(data_dir=str(tmp_path)))
    client = TestClient(app)

    health = client.get("/health")
    state = client.get("/state")

    assert health.status_code == 200
    assert state.status_code == 200
    assert health.json()["status"] == "ok"
    assert state.json()["phase"] == "awake"
    assert "sleep_need" in state.json()
    assert state.json()["sleep_cycles"] == 0
    assert "last_reweave_delta" in state.json()
    assert "trust" in state.json()
    assert "boundary_tension" in state.json()
    assert isinstance(state.json()["active_thread_ids"], list)


def test_turn_and_phase_endpoints(tmp_path: Path) -> None:
    app = build_app(engine=AuroraEngine.create(data_dir=str(tmp_path)))
    client = TestClient(app)

    turn = client.post(
        "/turn",
        json={"session_id": "s1", "text": "I learned something"},
    )
    doze = client.post("/doze")
    sleep = client.post("/sleep")

    assert turn.status_code == 200
    assert doze.status_code == 200
    assert sleep.status_code == 200
    assert turn.json()["turn_id"]
    assert "touch_channels" in turn.json()
    assert doze.json()["phase"] == "doze"
    assert sleep.json()["phase"] == "sleep"
