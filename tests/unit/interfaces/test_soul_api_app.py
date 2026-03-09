from __future__ import annotations

from fastapi.testclient import TestClient

from aurora.interfaces.api.app import app, get_runtime


def test_v3_identity_and_respond_endpoints(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AURORA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AURORA_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("AURORA_MEANING_EXTRACTOR", "heuristic")
    get_runtime.cache_clear()

    client = TestClient(app)

    identity_resp = client.get("/v3/identity")
    assert identity_resp.status_code == 200
    identity_payload = identity_resp.json()
    assert identity_payload["identity"]["phase"] == "dependent_child"
    assert "narrative_summary" in identity_payload

    respond_resp = client.post(
        "/v3/respond",
        json={"session_id": "api", "user_message": "你好"},
    )
    assert respond_resp.status_code == 200
    payload = respond_resp.json()
    assert payload["memory_context"]["phase"] == "dependent_child"
    assert "identity" in payload["memory_context"]
    assert "narrative_summary" in payload["memory_context"]

    legacy_resp = client.post("/v2/memory/respond", json={"session_id": "api", "user_message": "你好"})
    assert legacy_resp.status_code == 404

    get_runtime.cache_clear()
