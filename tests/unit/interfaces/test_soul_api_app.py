from __future__ import annotations

from fastapi.testclient import TestClient

import aurora.interfaces.api.app as api_app
from aurora.interfaces.api.app import app, get_runtime


def test_v4_identity_and_respond_endpoints(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AURORA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AURORA_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("AURORA_AXIS_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("AURORA_MEANING_PROVIDER", "heuristic")
    monkeypatch.setenv("AURORA_NARRATIVE_PROVIDER", "heuristic")
    get_runtime.cache_clear()

    with TestClient(app) as client:
        identity_resp = client.get("/v4/identity")
        assert identity_resp.status_code == 200
        identity_payload = identity_resp.json()
        assert identity_payload["identity"]["current_mode"]
        assert "axis_state" in identity_payload["identity"]
        assert "narrative_summary" in identity_payload

        respond_resp = client.post(
            "/v4/respond",
            json={"session_id": "api", "user_message": "你好"},
        )
        assert respond_resp.status_code == 200
        payload = respond_resp.json()
        assert payload["memory_context"]["mode"]
        assert "identity" in payload["memory_context"]
        assert "narrative_summary" in payload["memory_context"]
        assert payload["ingest_result"]["mode"]

        stats_resp = client.get("/v4/stats")
        assert stats_resp.status_code == 200
        stats_payload = stats_resp.json()
        assert stats_payload["architecture_mode"] == "graph_first"
        assert "graph_metrics" in stats_payload
        assert "background_evolver" in stats_payload

        evolve_resp = client.post("/v4/evolve", json={"dreams": 1})
        assert evolve_resp.status_code == 200
        evolve_payload = evolve_resp.json()
        assert {"dreams", "repairs", "total"} <= set(evolve_payload.keys())

    get_runtime.cache_clear()


def test_api_shutdown_closes_runtime(monkeypatch) -> None:
    observed = {"closed": 0}

    class FakeRuntime:
        def close(self) -> None:
            observed["closed"] += 1

    monkeypatch.setattr(api_app, "get_runtime", lambda: FakeRuntime())

    api_app.shutdown_runtime()

    assert observed["closed"] == 1
