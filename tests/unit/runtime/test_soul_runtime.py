from __future__ import annotations

from pathlib import Path

from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings
from tests.helpers.query_router import build_test_llm


class CountingLLM(LLMProvider):
    def __init__(self) -> None:
        self._inner = build_test_llm()
        self.complete_calls = 0
        self.complete_json_calls = 0

    def complete(
        self,
        prompt: str,
        *,
        system=None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop=None,
        metadata=None,
        max_retries=None,
    ) -> str:
        self.complete_calls += 1
        return self._inner.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            stop=stop,
            metadata=metadata,
            max_retries=max_retries,
        )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema,
        temperature: float = 0.2,
        timeout_s: float = 30.0,
        metadata=None,
        max_retries=None,
    ):
        self.complete_json_calls += 1
        return self._inner.complete_json(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            timeout_s=timeout_s,
            metadata=metadata,
            max_retries=max_retries,
        )


class StreamingLLM(CountingLLM):
    def stream_complete(
        self,
        prompt: str,
        *,
        system=None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop=None,
        metadata=None,
        max_retries=None,
    ):
        self.complete_calls += 1
        yield from self._inner.stream_complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            stop=stop,
            metadata=metadata,
            max_retries=max_retries,
        )


def _settings(tmp_path: Path) -> AuroraSettings:
    return AuroraSettings(
        data_dir=str(tmp_path),
        embedding_provider="hash",
        axis_embedding_provider="hash",
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

    result = runtime.respond(session_id="s1", user_message="你会什么")

    assert llm.complete_calls == 1
    assert llm.complete_json_calls >= 1
    assert result.reply.strip()
    assert result.persistence.status == "accepted"
    assert result.persistence.job_id
    assert result.memory_context.identity is not None
    assert result.memory_context.mode == result.memory_context.identity.current_mode
    runtime.close()


def test_runtime_respond_stream_emits_persist_accept_stage(tmp_path: Path) -> None:
    llm = StreamingLLM()
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=llm)

    events = list(runtime.respond_stream(session_id="s1", user_message="你会什么"))

    assert llm.complete_calls == 1
    assert any(event.kind == "reply_delta" for event in events)
    assert any(event.stage == "persist_accept" for event in events if event.kind == "status")
    assert events[-1].kind == "done"
    assert events[-1].result is not None
    assert events[-1].result.reply.strip()
    runtime.close()
def test_accept_interaction_is_immediately_queryable_via_overlay(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=CountingLLM())

    receipt = runtime.accept_interaction(
        event_id="evt_overlay",
        session_id="chat",
        user_message="我喜欢本地优先记忆",
        agent_message="我会记住这个偏好",
    )
    result = runtime.query(text="本地优先记忆", k=5, session_id="chat")

    assert receipt.status == "accepted"
    assert result.overlay_hit_count >= 1
    assert any(hit.kind == "event" and hit.id == "evt_overlay" for hit in result.hits)
    runtime.close()


def test_projected_event_replaces_overlay_hit(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=CountingLLM())

    runtime.accept_interaction(
        event_id="evt_project",
        session_id="chat",
        user_message="我喜欢图优先记忆",
        agent_message="我会记住图优先这个偏好",
    )
    assert runtime.wait_for_idle(timeout=3.0)

    result = runtime.query(text="图优先", k=5, session_id="chat")

    assert any(hit.kind == "plot" for hit in result.hits)
    assert not any(hit.kind == "event" and hit.id == "evt_project" for hit in result.hits)
    runtime.close()


def test_runtime_exposes_event_and_job_status(tmp_path: Path) -> None:
    runtime = AuroraRuntime(settings=_settings(tmp_path), llm=CountingLLM())

    receipt = runtime.accept_interaction(
        event_id="evt_status",
        session_id="chat",
        user_message="记录一下队列状态",
        agent_message="收到",
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
