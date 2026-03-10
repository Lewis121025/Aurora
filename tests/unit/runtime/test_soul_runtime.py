from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aurora.integrations.storage.doc_store import Document, SQLiteDocStore
from aurora.integrations.storage.event_log import Event, SQLiteEventLog
from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings
from aurora.system.errors import ConfigurationError


class CountingLLM(LLMProvider):
    def __init__(self) -> None:
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
        return "我会陪你一起把这个问题展开。"

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
        raise AssertionError("heuristic v4 runtime path should not need auxiliary JSON LLM calls")


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
        yield "我会"
        yield "陪你一起"
        yield "把这个问题展开。"


def test_runtime_respond_uses_single_generation_call(tmp_path: Path) -> None:
    llm = CountingLLM()
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        ),
        llm=llm,
    )

    result = runtime.respond(session_id="s1", user_message="你会什么")

    assert llm.complete_calls == 1
    assert llm.complete_json_calls == 0
    assert result.memory_context.mode == result.memory_context.identity.current_mode
    assert result.memory_context.identity is not None
    assert result.ingest_result.mode == result.memory_context.identity.current_mode


def test_runtime_respond_stream_emits_reply_chunks_and_done(tmp_path: Path) -> None:
    llm = StreamingLLM()
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        ),
        llm=llm,
    )

    events = list(runtime.respond_stream(session_id="s1", user_message="你会什么"))

    assert llm.complete_calls == 1
    assert [event.kind for event in events[:2]] == ["status", "status"]
    assert any(event.kind == "reply_delta" for event in events)
    assert events[-1].kind == "done"
    assert events[-1].result is not None
    assert events[-1].result.reply == "我会陪你一起把这个问题展开。"


def test_runtime_respond_without_llm_uses_runtime_fallback(tmp_path: Path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
            llm_provider=None,
        )
    )

    result = runtime.respond(session_id="s1", user_message="你在吗")

    assert result.llm_error == "llm_not_configured"
    assert result.reply
    assert "语言模型" in result.reply


def test_runtime_rejects_legacy_snapshot(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True)
    (snapshots_dir / "snapshot_1.json").write_text(
        json.dumps({"last_seq": 1, "state": {"schema_version": "v3"}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        AuroraRuntime(
            settings=AuroraSettings(
                data_dir=str(tmp_path),
                embedding_provider="hash",
                axis_embedding_provider="hash",
                meaning_provider="heuristic",
                narrative_provider="heuristic",
            )
        )


def test_runtime_rejects_buried_legacy_docs(tmp_path: Path) -> None:
    store = SQLiteDocStore(str(tmp_path / "docs.sqlite3"))
    store.upsert(
        Document(
            id="plot_legacy",
            kind="plot",
            ts=1.0,
            body={"runtime_schema_version": "aurora-runtime-v3"},
        )
    )
    for index in range(6):
        store.upsert(
            Document(
                id=f"plot_new_{index}",
                kind="plot",
                ts=100.0 + index,
                body={"runtime_schema_version": "aurora-runtime-v4"},
            )
        )
    store.close()

    with pytest.raises(ConfigurationError):
        AuroraRuntime(
            settings=AuroraSettings(
                data_dir=str(tmp_path),
                embedding_provider="hash",
                axis_embedding_provider="hash",
                meaning_provider="heuristic",
                narrative_provider="heuristic",
            )
        )


def test_runtime_replay_writes_bootstrap_snapshot(tmp_path: Path) -> None:
    log = SQLiteEventLog(str(tmp_path / "events.sqlite3"))
    log.append(
        Event(
            id="evt_bootstrap",
            ts=1.0,
            session_id="s1",
            type="interaction",
            payload={
                "runtime_schema_version": "aurora-runtime-v4",
                "user_message": "你今天看起来有点远。",
                "agent_message": "我在这里，会继续听你说。",
                "actors": ["user", "agent"],
                "context": "你今天看起来有点远。",
            },
        )
    )
    log.close()

    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
            snapshot_every_events=0,
        )
    )

    snapshot_files = sorted((tmp_path / "snapshots").glob("snapshot_*.json"))
    assert runtime.last_seq == 1
    assert snapshot_files


def test_runtime_stats_expose_architecture_mode_and_graph_metrics(tmp_path: Path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        )
    )

    stats = runtime.get_stats()

    assert stats["architecture_mode"] == "shadow"
    assert isinstance(stats["graph_metrics"], dict)
    assert isinstance(stats["background_evolver"], dict)


def test_runtime_background_evolver_can_be_disabled(tmp_path: Path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
            background_evolver_enabled=False,
            evolve_every_seconds=0.05,
        )
    )

    stats = runtime.get_stats()

    assert stats["background_evolver"]["enabled"] is False
    assert runtime._background_thread is None
    runtime.close()
    runtime.close()


def test_runtime_background_evolver_snapshots_authoritative_mutations(tmp_path: Path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
            evolve_every_seconds=0.05,
            snapshot_every_events=0,
        )
    )
    evolved = threading.Event()

    def fake_evolve(*, dreams=None):
        runtime.mem.step += 1
        evolved.set()
        return [SimpleNamespace(source="dream")]

    runtime.mem.evolve = fake_evolve  # type: ignore[method-assign]

    assert evolved.wait(timeout=1.0)

    deadline = time.time() + 1.0
    snapshot_files = []
    while time.time() < deadline:
        snapshot_files = sorted((tmp_path / "snapshots").glob("snapshot_*.json"))
        if snapshot_files:
            break
        time.sleep(0.02)

    runtime.close()
    runtime.close()

    assert snapshot_files
    assert runtime._background_thread is not None
    assert not runtime._background_thread.is_alive()
