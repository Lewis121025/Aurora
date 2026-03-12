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


def _sample_request() -> CollapseRequest:
    return CollapseRequest(
        user_text="hello",
        released_traces=[ReleasedTrace(text="greeting", source="trace")],
        released_virtual_traces=[],
        language="en",
        emit_reply=True,
        boundary_budget=0.3,
        verbosity_budget=0.4,
    )


def test_provider_retries_once_on_timeout(monkeypatch) -> None:
    attempts = {"count": 0}

    def fake_urlopen(req: urllib.request.Request, timeout: float):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("slow provider")
        assert timeout >= 60.0
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OpenAICompatibleCollapseProvider(
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="test-key",
        timeout_s=10.0,
    )

    result = provider.collapse(_sample_request())

    assert result.output_text == "ok"
    assert attempts["count"] == 2


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
        released_virtual_traces=[],
        language="en",
        emit_reply=False,
        boundary_budget=0.3,
        verbosity_budget=0.4,
    )

    result = provider.collapse(request)

    assert result.output_text is None
