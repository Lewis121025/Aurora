from __future__ import annotations

import threading
import shutil
import tempfile
from pathlib import Path
from typing import List

import pytest

from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings


@pytest.fixture
def runtime_settings(temp_data_dir: Path) -> AuroraSettings:
    return AuroraSettings(
        data_dir=str(temp_data_dir),
        llm_provider="mock",
        embedding_provider="hash",
        dim=64,
        pii_redaction_enabled=False,
        snapshot_every_events=0,
        memory_seed=7,
    )


@pytest.fixture
def runtime(runtime_settings: AuroraSettings) -> AuroraRuntime:
    return AuroraRuntime(settings=runtime_settings)


def test_runtime_uses_single_storage_directory(runtime: AuroraRuntime, runtime_settings: AuroraSettings):
    assert runtime.settings is runtime_settings
    assert Path(runtime.event_log.path).parent == Path(runtime_settings.data_dir)
    assert Path(runtime.doc_store.path).parent == Path(runtime_settings.data_dir)


def test_runtime_persists_and_queries_single_conversation(runtime: AuroraRuntime):
    runtime.ingest_interaction(
        event_id="evt_001",
        session_id="chat_a",
        user_message="我喜欢把记忆系统做成叙事结构。",
        agent_message="可以把 plot、story、theme 分层建模。",
    )

    result = runtime.query(text="叙事结构的记忆系统", k=5)

    assert result.query == "叙事结构的记忆系统"
    assert result.hits


def test_runtime_replays_existing_history(runtime_settings: AuroraSettings):
    first = AuroraRuntime(settings=runtime_settings)
    created = first.ingest_interaction(
        event_id="evt_001",
        session_id="chat_a",
        user_message="我想让聊天记忆能持续演化。",
        agent_message="可以定期合并 plot 成更稳定的 story。",
    )

    second = AuroraRuntime(settings=runtime_settings)

    if created.encoded:
        assert created.plot_id in second.mem.plots


def test_runtime_replay_deterministic_ids():
    tmp = tempfile.mkdtemp()
    try:
        settings = AuroraSettings(
            data_dir=tmp,
            llm_provider="mock",
            embedding_provider="hash",
            snapshot_every_events=0,
            memory_seed=7,
        )
        runtime1 = AuroraRuntime(settings=settings)
        first = runtime1.ingest_interaction(
            event_id="evt1",
            session_id="chat_deterministic",
            user_message="hello",
            agent_message="world",
        )
        second = runtime1.ingest_interaction(
            event_id="evt2",
            session_id="chat_deterministic",
            user_message="foo",
            agent_message="bar",
        )

        replayed = AuroraRuntime(settings=settings)

        if first.encoded:
            assert first.plot_id in replayed.mem.plots
        if second.encoded:
            assert second.plot_id in replayed.mem.plots
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runtime_ingest_idempotent():
    tmp = tempfile.mkdtemp()
    try:
        settings = AuroraSettings(
            data_dir=tmp,
            llm_provider="mock",
            embedding_provider="hash",
            snapshot_every_events=0,
        )
        runtime = AuroraRuntime(settings=settings)
        first = runtime.ingest_interaction(
            event_id="evt1",
            session_id="chat_idempotent",
            user_message="hello",
            agent_message="world",
        )
        second = runtime.ingest_interaction(
            event_id="evt1",
            session_id="chat_idempotent",
            user_message="hello",
            agent_message="world",
        )

        assert first.plot_id == second.plot_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runtime_is_thread_safe(runtime: AuroraRuntime):
    results: List[str] = []
    errors: List[Exception] = []

    def worker(idx: int) -> None:
        try:
            outcome = runtime.ingest_interaction(
                event_id=f"evt_{idx}",
                session_id="chat_parallel",
                user_message=f"message {idx}",
                agent_message=f"reply {idx}",
            )
            results.append(outcome.event_id)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 5
