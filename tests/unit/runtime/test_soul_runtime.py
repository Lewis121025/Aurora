from __future__ import annotations

from pathlib import Path

import pytest

from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.storage.runtime_store import StoredEvent
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings
from aurora.soul.models import Message, TextPart, messages_to_text
from aurora.system.errors import ConfigurationError


class CountingLLM(LLMProvider):
    def __init__(self, *, reply_text: str = "我记得这轮对话，也可以继续往下聊。") -> None:
        self.complete_calls = 0
        self.complete_json_calls = 0
        self._reply_text = reply_text

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop=None,
        metadata=None,
        max_retries=None,
    ):
        self.complete_calls += 1
        return Message(
            role="assistant",
            parts=(TextPart(text=self._reply_text),),
        )

    def complete_json(
        self,
        *,
        messages,
        schema,
        temperature: float = 0.2,
        timeout_s: float = 30.0,
        metadata=None,
        max_retries=None,
    ):
        self.complete_json_calls += 1
        query_text = messages[-1].parts[0].text if messages else ""
        keywords = [token for token in str(query_text).replace("\n", " ").split() if token][:3]
        return schema.model_validate(
            {
                "query_type": "FACTUAL",
                "temporal_plan": {"relation": "any", "relative_window": "none"},
                "is_aggregation": False,
                "aggregation_entities": [],
                "query_keywords": keywords,
                "query_type_confidence": 0.9,
                "temporal_confidence": 0.0,
            }
        )


class StreamingLLM(CountingLLM):
    def stream_complete(
        self,
        messages,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop=None,
        metadata=None,
        max_retries=None,
    ):
        self.complete_calls += 1
        yield "我记得"
        yield "这轮对话。"


class ReplyFailingLLM(CountingLLM):
    def complete(self, messages, **kwargs):
        self.complete_calls += 1
        raise RuntimeError("reply generation failed")


class StreamFailingLLM(CountingLLM):
    def stream_complete(self, messages, **kwargs):
        self.complete_calls += 1
        raise RuntimeError("stream generation failed")
        yield  # pragma: no cover


class NullLLM(LLMProvider):
    def complete(self, messages, **kwargs):
        raise NotImplementedError

    def complete_json(self, *, messages, schema, **kwargs):
        raise NotImplementedError


def _settings(tmp_path: Path) -> AuroraSettings:
    return AuroraSettings(
        data_dir=str(tmp_path),
        content_embedding_provider="hash",
        text_embedding_provider="hash",
        meaning_provider="heuristic",
        narrative_provider="heuristic",
        worker_count=1,
        job_poll_interval_ms=25,
        evolve_every_seconds=3600,
        fade_every_seconds=3600,
    )


def test_runtime_respond_returns_persistence_receipt(tmp_path: Path) -> None:
    llm = CountingLLM()
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=llm)

    result = runtime.respond(
        session_id="s1",
        user_messages=[Message(role="user", parts=(TextPart(text="你会什么"),))],
    )

    assert llm.complete_calls == 1
    assert llm.complete_json_calls >= 1
    assert messages_to_text((result.reply_message,)).strip()
    assert result.persistence.status == "accepted"
    assert result.persistence.job_id
    assert result.memory_context.identity is not None
    assert result.memory_context.mode == result.memory_context.identity.current_mode
    runtime.close()


def test_runtime_respond_stream_emits_persist_accept_stage(tmp_path: Path) -> None:
    llm = StreamingLLM()
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=llm)

    events = list(
        runtime.respond_stream(
            session_id="s1",
            user_messages=[Message(role="user", parts=(TextPart(text="你会什么"),))],
        )
    )

    assert llm.complete_calls == 1
    assert any(event.kind == "reply_delta" for event in events)
    assert any(event.stage == "persist_accept" for event in events if event.kind == "status")
    assert events[-1].kind == "done"
    assert events[-1].result is not None
    assert messages_to_text((events[-1].result.reply_message,)).strip()
    runtime.close()


def test_runtime_respond_raises_when_generation_fails(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=ReplyFailingLLM())

    with pytest.raises(RuntimeError, match="reply generation failed"):
        runtime.respond(
            session_id="s1",
            user_messages=[Message(role="user", parts=(TextPart(text="你会什么"),))],
        )

    assert runtime.recent_events(limit=5) == []
    runtime.close()


def test_runtime_respond_stream_raises_when_generation_fails(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=StreamFailingLLM())

    events = []
    with pytest.raises(RuntimeError, match="stream generation failed"):
        for event in runtime.respond_stream(
            session_id="s1",
            user_messages=[Message(role="user", parts=(TextPart(text="你会什么"),))],
        ):
            events.append(event)

    assert [event.stage for event in events if event.kind == "status"] == ["retrieval", "generation"]
    assert runtime.recent_events(limit=5) == []
    runtime.close()


def test_accept_interaction_is_immediately_queryable_via_overlay(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=CountingLLM())

    receipt = runtime.accept_interaction(
        event_id="evt_overlay",
        session_id="chat",
        messages=[
            Message(role="user", parts=(TextPart(text="我喜欢本地优先记忆"),)),
            Message(role="assistant", parts=(TextPart(text="我会记住这个偏好"),)),
        ],
    )
    result = runtime.query(
        messages=[Message(role="user", parts=(TextPart(text="本地优先记忆"),))],
        k=5,
        session_id="chat",
    )

    assert receipt.status == "accepted"
    assert result.overlay_hit_count >= 1
    assert any(hit.kind == "event" and hit.id == "evt_overlay" for hit in result.hits)
    runtime.close()


def test_projected_event_replaces_overlay_hit(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=CountingLLM())

    runtime.accept_interaction(
        event_id="evt_project",
        session_id="chat",
        messages=[
            Message(role="user", parts=(TextPart(text="我喜欢图优先记忆"),)),
            Message(role="assistant", parts=(TextPart(text="我会记住图优先这个偏好"),)),
        ],
    )
    assert runtime.wait_for_idle(timeout=3.0)

    result = runtime.query(
        messages=[Message(role="user", parts=(TextPart(text="图优先"),))],
        k=5,
        session_id="chat",
    )

    assert any(hit.kind == "plot" for hit in result.hits)
    assert not any(hit.kind == "event" and hit.id == "evt_project" for hit in result.hits)
    runtime.close()


def test_runtime_exposes_event_and_job_status(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=CountingLLM())

    receipt = runtime.accept_interaction(
        event_id="evt_status",
        session_id="chat",
        messages=[
            Message(role="user", parts=(TextPart(text="记录一下队列状态"),)),
            Message(role="assistant", parts=(TextPart(text="收到"),)),
        ],
    )
    event_status = runtime.get_event_status("evt_status")
    job_status = runtime.get_job_status(receipt.job_id)

    assert event_status["event_id"] == "evt_status"
    assert job_status["job_id"] == receipt.job_id
    runtime.close()


def test_runtime_stats_expose_queue_and_summary_metrics(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=CountingLLM())

    stats = runtime.get_stats()

    assert stats["architecture_mode"] == "graph_first"
    assert "summary_count" in stats
    assert "queue_depth" in stats
    runtime.close()


def test_runtime_rejects_old_event_schema_version(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=NullLLM())

    old_event = StoredEvent(
        seq=1,
        event_id="evt_old_schema",
        event_type="interaction",
        session_id="chat",
        ts=0.0,
        payload={
            "runtime_schema_version": "aurora-runtime-v5",
            "messages": [],
            "search_text": "",
        },
    )

    with pytest.raises(ConfigurationError, match="Unsupported runtime event schema version"):
        runtime._apply_event_to_memory(old_event)

    runtime.close()
