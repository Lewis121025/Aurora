from __future__ import annotations

import sys
import types

from aurora.integrations.llm.ark import ArkLLM
from aurora.integrations.llm.bailian import BailianLLM
from aurora.integrations.llm.schemas import MeaningFramePayloadV4


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = None


class _FakeCompletions:
    def __init__(self, owner: "_FakeClient", content: str):
        self._owner = owner
        self._content = content

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, owner: "_FakeClient", content: str):
        self.completions = _FakeCompletions(owner, content)


class _FakeClient:
    def __init__(self, content: str):
        self.calls = []
        self.chat = _FakeChat(self, content)


def test_ark_client_disables_sdk_retries(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    llm = ArkLLM(api_key="test-key", base_url="https://example.com/v1")
    llm._get_client()

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["max_retries"] == 0


def test_bailian_complete_disables_thinking():
    llm = BailianLLM(api_key="test-key", base_url="https://example.com/v1")
    fake_client = _FakeClient("hello")
    llm._client = fake_client

    reply = llm.complete("hi", system="sys", max_tokens=64, timeout_s=12.0)

    assert reply == "hello"
    assert fake_client.calls[0]["extra_body"] == {"enable_thinking": False}
    assert fake_client.calls[0]["max_tokens"] == 64
    assert fake_client.calls[0]["timeout"] == 12.0


def test_bailian_complete_json_uses_structured_schema_and_no_thinking():
    llm = BailianLLM(api_key="test-key", base_url="https://example.com/v1")
    fake_client = _FakeClient(
        """
        {
          "axis_evidence": {
            "trust": 0.4,
            "openness": 0.1
          },
          "valence": 0.8,
          "arousal": 0.2,
          "tags": ["warmth", "care"],
          "threat": 0.0,
          "care": 0.7,
          "control": 0.0,
          "abandonment": 0.0,
          "agency_signal": 0.1,
          "shame": 0.0
        }
        """.strip()
    )
    llm._client = fake_client

    result = llm.complete_json(
        system="sys",
        user="user",
        schema=MeaningFramePayloadV4,
        timeout_s=9.0,
    )

    call = fake_client.calls[0]
    assert result.axis_evidence["trust"] == 0.4
    assert call["extra_body"] == {"enable_thinking": False}
    assert call["max_tokens"] == 512
    assert call["timeout"] == 9.0
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == "MeaningFramePayloadV4"
