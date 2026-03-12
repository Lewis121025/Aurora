from __future__ import annotations

import json
import urllib.request

from aurora.core_math.contracts import CollapseRequest, ReleasedTrace
from aurora.host_runtime.provider import OpenAICompatibleCollapseProvider


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_provider_without_reply_skips_network(monkeypatch) -> None:
    def fake_urlopen(req: urllib.request.Request, timeout: float):
        raise AssertionError("urlopen should not be called when emit_reply is false")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OpenAICompatibleCollapseProvider(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="test-key",
    )
    request = CollapseRequest(
        user_text="hello",
        released_traces=[ReleasedTrace(text="greeting", source="trace")],
        language="en",
        emit_reply=False,
        boundary_budget=0.3,
        verbosity_budget=0.4,
    )
    result = provider.collapse(request)
    assert result.output_text is None


def test_provider_collapse_sends_correct_payload(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float):
        captured.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse({"choices": [{"message": {"content": "hello back"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OpenAICompatibleCollapseProvider(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="test-key",
    )
    request = CollapseRequest(
        user_text="hello",
        released_traces=[ReleasedTrace(text="a memory", source="user")],
        language="en",
        emit_reply=True,
        boundary_budget=0.4,
        verbosity_budget=0.5,
    )
    result = provider.collapse(request)
    assert result.output_text == "hello back"
    assert len(captured) == 1
    user_content = json.loads(captured[0]["messages"][1]["content"])
    assert "released_traces" in user_content
    assert "released_virtual_traces" not in user_content
