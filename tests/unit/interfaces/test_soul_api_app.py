from __future__ import annotations

from fastapi.testclient import TestClient

import aurora.interfaces.api.app as api_app
from aurora.interfaces.api.app import app
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings
from tests.helpers.query_router import build_test_llm


def _messages(text: str) -> list[dict[str, object]]:
    return [{"role": "user", "parts": [{"type": "text", "text": text}]}]


def test_v6_identity_query_and_reply_endpoints(monkeypatch, tmp_path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            content_embedding_provider="hash",
            text_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        ),
        llm=build_test_llm(),
    )
    monkeypatch.setattr(api_app, "get_runtime", lambda: runtime)

    with TestClient(app) as client:
        identity_resp = client.get("/v6/identity")
        assert identity_resp.status_code == 200
        identity_payload = identity_resp.json()
        assert identity_payload["identity"]["current_mode"]
        assert "axis_state" in identity_payload["identity"]

        interaction_resp = client.post(
            "/v6/interactions",
            json={
                "session_id": "api",
                "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "请记住我喜欢本地优先"}]},
                    {"role": "assistant", "parts": [{"type": "text", "text": "我会记住这个偏好"}]},
                ],
            },
        )
        assert interaction_resp.status_code == 200
        interaction_payload = interaction_resp.json()
        assert interaction_payload["status"] == "accepted"
        assert interaction_payload["job_id"]

        query_resp = client.post(
            "/v6/query",
            json={"messages": _messages("本地优先"), "session_id": "api"},
        )
        assert query_resp.status_code == 200
        query_payload = query_resp.json()
        assert "hits" in query_payload
        assert "overlay_hit_count" in query_payload

        reply_resp = client.post(
            "/v6/chat/replies",
            json={"session_id": "api", "messages": _messages("你好")},
        )
        assert reply_resp.status_code == 200
        reply_payload = reply_resp.json()
        assert reply_payload["memory_context"]["mode"]
        assert reply_payload["persistence"]["status"] == "accepted"
        assert reply_payload["reply_message"]["role"] == "assistant"

        stats_resp = client.get("/v6/stats")
        assert stats_resp.status_code == 200
        stats_payload = stats_resp.json()
        assert stats_payload["architecture_mode"] == "graph_first"
        assert "queue_depth" in stats_payload

        event_resp = client.get(f"/v6/events/{interaction_payload['event_id']}")
        assert event_resp.status_code == 200
        assert event_resp.json()["event_id"] == interaction_payload["event_id"]

        job_resp = client.get(f"/v6/jobs/{interaction_payload['job_id']}")
        assert job_resp.status_code == 200
        assert job_resp.json()["job_id"] == interaction_payload["job_id"]
    runtime.close()


def test_api_shutdown_closes_runtime(monkeypatch) -> None:
    observed = {"closed": 0}

    class FakeRuntime:
        def close(self) -> None:
            observed["closed"] += 1

    monkeypatch.setattr(api_app, "get_runtime", lambda: FakeRuntime())
    api_app.shutdown_runtime()
    assert observed["closed"] == 1
