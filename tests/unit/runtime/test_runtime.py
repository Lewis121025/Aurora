from __future__ import annotations

import threading
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

import aurora.runtime.runtime as runtime_module
from aurora.core.memory import AuroraMemory
from aurora.core.personality import load_personality_profile
from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.schemas import PlotExtraction
from aurora.runtime.bootstrap import build_memory_config
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings

PROFILE_ID = "aurora-v2-child-elara"


class TrackingEmbedding(EmbeddingProvider):
    def __init__(self, dim: int, *, fail_on_call: bool = False):
        self._delegate = HashEmbedding(dim=dim, seed=42)
        self._fail_on_call = fail_on_call
        self.calls = 0

    def embed(self, text: str) -> np.ndarray:
        if self._fail_on_call:
            raise AssertionError("embed() should not be called")
        self.calls += 1
        return self._delegate.embed(text)


class SpyLLM(LLMProvider):
    def __init__(self):
        self.json_calls = 0
        self.last_kwargs = {}

    def complete(self, prompt: str, **kwargs) -> str:
        return prompt

    def complete_json(self, *, system: str, user: str, schema, **kwargs):
        self.json_calls += 1
        self.last_kwargs = kwargs
        return schema.model_validate(
            PlotExtraction(
                actors=["user", "agent"],
                action="capture preference",
                outcome="user prefers narrative memory",
                context="test",
            ).model_dump()
        )


class ExplodingLLM(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        raise AssertionError("complete() should not be called")

    def complete_json(self, *, system: str, user: str, schema, **kwargs):
        raise AssertionError("complete_json() should not be called")


def build_test_memory(settings: AuroraSettings, embedder: EmbeddingProvider) -> AuroraMemory:
    cfg = build_memory_config(settings)
    return AuroraMemory(
        cfg=cfg,
        seed=int(settings.memory_seed),
        embedder=embedder,
        benchmark_mode=cfg.benchmark_mode,
        bootstrap_profile=True,
    )


def expected_profile_embedding_calls() -> int:
    profile = load_personality_profile(PROFILE_ID)
    return len(profile.seed_plots) + len(profile.intuition_anchors) + len(profile.subconscious_seeds)


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
    assert runtime.mem.self_narrative_engine.narrative.profile_id == PROFILE_ID
    assert runtime.mem.self_narrative_engine.narrative.seed_plot_ids
    assert len(runtime.mem.subconscious_state.dark_matter_pool) == 2


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

    if created.memory_layer == "explicit":
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

        if first.memory_layer == "explicit":
            assert first.plot_id in replayed.mem.plots
        if second.memory_layer == "explicit":
            assert second.plot_id in replayed.mem.plots
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runtime_persists_canonical_replay_payload(runtime_settings: AuroraSettings, monkeypatch: pytest.MonkeyPatch):
    embedder = TrackingEmbedding(dim=runtime_settings.dim)
    monkeypatch.setattr(
        runtime_module,
        "create_memory",
        lambda *, settings: build_test_memory(settings, embedder),
    )
    llm = SpyLLM()
    runtime = AuroraRuntime(settings=runtime_settings, llm=llm)

    runtime.ingest_interaction(
        event_id="evt_payload",
        session_id="chat_payload",
        user_message="我喜欢长期记忆系统。",
        agent_message="我会持续保留你的偏好。",
        context="memory_preference",
    )

    events = list(runtime.event_log.iter_events())
    assert len(events) == 1

    payload = events[0][1].payload
    assert payload["runtime_schema_version"] == "aurora-runtime-v2"
    assert payload["plot_extraction"]["outcome"] == "user prefers narrative memory"
    assert payload["interaction_text"].startswith("USER: 我喜欢长期记忆系统。")
    assert payload["resolved_actors"] == ["user", "agent"]
    assert len(payload["interaction_embedding"]) == runtime_settings.dim
    assert len(payload["context_embedding"]) == runtime_settings.dim
    assert llm.json_calls == 1
    assert llm.last_kwargs["max_retries"] == runtime_module.PLOT_EXTRACTION_MAX_RETRIES
    assert llm.last_kwargs["timeout_s"] == runtime_module.PLOT_EXTRACTION_TIMEOUT_S
    assert embedder.calls == 2 + expected_profile_embedding_calls()


def test_runtime_replay_uses_persisted_payload_without_model_calls(
    runtime_settings: AuroraSettings,
    monkeypatch: pytest.MonkeyPatch,
):
    first_embedder = TrackingEmbedding(dim=runtime_settings.dim)
    monkeypatch.setattr(
        runtime_module,
        "create_memory",
        lambda *, settings: build_test_memory(settings, first_embedder),
    )
    runtime = AuroraRuntime(settings=runtime_settings, llm=SpyLLM())
    created = runtime.ingest_interaction(
        event_id="evt_replay_local",
        session_id="chat_replay_local",
        user_message="帮我记住我喜欢叙事化结构。",
        agent_message="我会记住你偏爱叙事化结构。",
        context="memory_preference",
    )

    second_embedder = TrackingEmbedding(dim=runtime_settings.dim)
    monkeypatch.setattr(
        runtime_module,
        "create_memory",
        lambda *, settings: build_test_memory(settings, second_embedder),
    )
    replayed = AuroraRuntime(settings=runtime_settings, llm=ExplodingLLM())

    assert created.plot_id in replayed.mem.plots
    assert second_embedder.calls == expected_profile_embedding_calls()


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
