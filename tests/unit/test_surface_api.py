from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aurora.runtime.engine import AuroraEngine
from aurora.surface.api import build_app


def test_health_and_state_endpoints(tmp_path: Path) -> None:
    app = build_app(engine=AuroraEngine.create(data_dir=str(tmp_path)))
    client = TestClient(app)

    health = client.get("/v1/health")
    state = client.get("/v1/state")

    assert health.status_code == 200
    assert state.status_code == 200
    assert health.json()["status"] == "ok"
    assert state.json()["phase"] == "awake"
    assert state.json()["sleep_cycles"] == 0
    assert "last_reweave_delta" in state.json()
    assert "relation_tone" in state.json()
    assert "relation_strength" in state.json()
    assert "avg_narrative_weight" in state.json()
    assert "narrative_pressure" in state.json()
    assert 0.0 <= float(state.json()["relation_strength"]) <= 1.0


def test_turn_and_phase_endpoints(tmp_path: Path) -> None:
    app = build_app(engine=AuroraEngine.create(data_dir=str(tmp_path)))
    client = TestClient(app)

    turn = client.post(
        "/v1/turn",
        json={"session_id": "s1", "text": "I learned something"},
    )
    doze = client.post("/v1/doze")
    sleep = client.post("/v1/sleep")

    assert turn.status_code == 200
    assert doze.status_code == 200
    assert sleep.status_code == 200
    assert turn.json()["turn_id"]
    assert doze.json()["phase"] == "doze"
    assert sleep.json()["phase"] == "sleep"
