from __future__ import annotations

import pytest

from aurora.core.memory import AuroraMemory
from aurora.core.memory.ingestion import StorageDecision
from aurora.core.models.config import MemoryConfig
from aurora.core.personality import load_personality_profile
from aurora.core.retrieval.query_analysis import QueryType


@pytest.fixture
def seeded_memory() -> AuroraMemory:
    memory = AuroraMemory(
        cfg=MemoryConfig(dim=64, metric_rank=16, personality_profile_id="aurora-v2-native"),
        seed=21,
        benchmark_mode=False,
        bootstrap_profile=True,
    )
    return memory


def test_shadow_plot_enters_subconscious_pool(monkeypatch: pytest.MonkeyPatch):
    memory = AuroraMemory(cfg=MemoryConfig(dim=64, metric_rank=16), seed=21)
    monkeypatch.setattr(
        memory,
        "_compute_storage_decision",
        lambda plot: StorageDecision(memory_layer="shadow", probability=0.1, reason="test-shadow"),
    )

    plot = memory.ingest("用户：这句话应该落入潜意识层。助理：收到。")

    assert plot.exposure == "shadow"
    assert plot.id in memory.plots
    assert plot.id not in memory.graph.g
    assert memory.subconscious_state.dark_matter_pool
    assert memory.subconscious_state.dark_matter_pool[0].source_plot_id == plot.id


def test_identity_query_returns_seed_plots_only_for_identity(seeded_memory: AuroraMemory):
    profile = load_personality_profile("aurora-v2-native")
    seed_text = profile.seed_plots[0].text

    identity_trace = seeded_memory.query(
        seed_text,
        k=5,
        query_type=QueryType.IDENTITY,
    )
    factual_trace = seeded_memory.query(
        seed_text,
        k=5,
        query_type=QueryType.FACTUAL,
    )

    assert any(
        seeded_memory.plots[nid].source == "seed"
        for nid, _score, kind in identity_trace.ranked
        if kind == "plot"
    )
    assert all(
        seeded_memory.plots[nid].source != "seed"
        for nid, _score, kind in factual_trace.ranked
        if kind == "plot"
    )
