from __future__ import annotations

import numpy as np

from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.soul.engine import AuroraSoul, SoulConfig
from aurora.soul.extractors import CombinatorialNarrativeProvider, HeuristicMeaningProvider


def build_memory(seed: int, architecture_mode: str = "shadow") -> AuroraSoul:
    embedder = HashEmbedding(dim=64, seed=seed)
    return AuroraSoul(
        cfg=SoulConfig(
            dim=64,
            metric_rank=16,
            max_plots=128,
            architecture_mode=architecture_mode,
        ),
        seed=seed,
        event_embedder=embedder,
        axis_embedder=embedder,
        meaning_provider=HeuristicMeaningProvider(),
        narrator=CombinatorialNarrativeProvider(),
    )


def test_soul_memory_round_trip_preserves_identity_and_plots() -> None:
    mem = build_memory(seed=7)

    first = None
    for text in ["对方冷淡地拒绝我。", "你又一次忽视我。", "我有点害怕被丢下。"]:
        first = mem.ingest(text, actors=("user", "agent"))
        if first.story_id is not None:
            break
    assert first is not None
    assert first.story_id is not None
    assert mem.identity.current_mode_label

    state = mem.to_state_dict()
    restored = AuroraSoul.from_state_dict(
        state,
        event_embedder=HashEmbedding(dim=64, seed=7),
        axis_embedder=HashEmbedding(dim=64, seed=7),
        meaning_provider=HeuristicMeaningProvider(),
        narrator=CombinatorialNarrativeProvider(),
    )

    assert restored.identity.current_mode_label == mem.identity.current_mode_label
    snapshot = restored.snapshot_identity().to_state_dict()
    assert snapshot["current_mode"] == restored.identity.current_mode_label
    assert "axis_state" in snapshot
    assert len(restored.plots) >= 1
    assert len(restored.stories) >= 1
    assert restored.narrative_summary().current_mode == restored.identity.current_mode_label


def test_soul_memory_feedback_and_evolve_keep_new_observables_available() -> None:
    mem = build_memory(seed=11)

    mem.ingest("你总是不理我，让我很难过。", actors=("user", "agent"))
    trace = mem.query("她现在是什么状态？", k=4)
    assert trace.ranked

    top_id, _, _ = trace.ranked[0]
    mem.feedback_retrieval("她现在是什么状态？", top_id, True)
    dreams = mem.evolve(dreams=1)

    snapshot = mem.snapshot_identity()
    assert snapshot.current_mode
    assert isinstance(snapshot.axis_aliases, dict)
    assert isinstance(snapshot.modes, dict)
    assert snapshot.dream_count >= len(dreams)


def test_restore_does_not_call_remote_bootstrap_hooks() -> None:
    class BootstrapOnlyMeaning(HeuristicMeaningProvider):
        def extract_persona_axes(self, profile_text: str):
            raise AssertionError("restore should not invoke persona-axis extraction")

        def bootstrap_persona_axes(self, profile_text: str):
            return []

    class BootstrapOnlyNarrator(CombinatorialNarrativeProvider):
        def label_mode(self, prototype_axes, schema, support):
            raise AssertionError("restore should not invoke mode labeling")

        def bootstrap_mode_label(self, prototype_axes, schema, support):
            return "origin"

    embedder = HashEmbedding(dim=64, seed=19)
    mem = AuroraSoul(
        cfg=SoulConfig(dim=64, metric_rank=16, max_plots=128),
        seed=19,
        event_embedder=embedder,
        axis_embedder=embedder,
        meaning_provider=HeuristicMeaningProvider(),
        narrator=CombinatorialNarrativeProvider(),
    )
    mem.ingest("你突然安静下来，让我有点不确定。", actors=("user", "agent"))

    restored = AuroraSoul.from_state_dict(
        mem.to_state_dict(),
        event_embedder=embedder,
        axis_embedder=embedder,
        meaning_provider=BootstrapOnlyMeaning(),
        narrator=BootstrapOnlyNarrator(),
    )

    assert restored.identity.current_mode_label == mem.identity.current_mode_label
    assert np.linalg.norm(restored.identity.self_vector) >= 0.0


def test_shadow_projection_records_query_and_evolve_metrics() -> None:
    mem = build_memory(seed=23, architecture_mode="shadow")

    for text in [
        "你刚才突然沉默，让我有点不安。",
        "但你后来又试着解释，我们继续靠近了一点。",
        "我还是会反复想起那些忽远忽近的时刻。",
    ]:
        mem.ingest(text, actors=("user", "agent"))

    trace = mem.query("我们之前围绕什么情绪来回摆动过", k=5)
    evolved = mem.evolve(dreams=1)

    assert trace.ranked
    assert mem.shadow_projection is not None
    assert "query" in mem.shadow_metrics
    assert "evolve" in mem.shadow_metrics
    assert "top_k_overlap" in mem.shadow_metrics["query"]
    assert "dream_candidates" in mem.shadow_metrics["evolve"]
    assert isinstance(evolved, list)


def test_shadow_restore_rebuilds_projection_from_authoritative_plots() -> None:
    mem = build_memory(seed=27, architecture_mode="shadow")
    mem.ingest("你反复确认我是不是还会留下。", actors=("user", "agent"))
    mem.ingest("我说我在，但你还是有点不确定。", actors=("user", "agent"))

    restored = AuroraSoul.from_state_dict(
        mem.to_state_dict(),
        event_embedder=HashEmbedding(dim=64, seed=27),
        axis_embedder=HashEmbedding(dim=64, seed=27),
        meaning_provider=HeuristicMeaningProvider(),
        narrator=CombinatorialNarrativeProvider(),
    )

    assert restored.shadow_projection is not None
    assert len(restored.shadow_projection.plots) == len(
        [node_id for node_id, kind in zip(restored.vindex.ids, restored.vindex.kinds) if kind == "plot"]
    )


def test_graph_first_materializes_story_and_theme_views_on_query() -> None:
    mem = build_memory(seed=29, architecture_mode="graph_first")

    for text in [
        "我们反复讨论本地优先的长期记忆。",
        "你又提到结构化记忆和检索一致性。",
        "我们继续围绕图记忆和主题整理展开。",
    ]:
        plot = mem.ingest(text, actors=("user", "agent"))
        assert plot.story_id is None

    trace = mem.query("我们之前围绕什么主题讨论过", k=6)

    assert trace.ranked
    assert mem.stories
    assert mem.themes
    assert any(plot.story_id is not None for plot in mem.plots.values())
    assert any(plot.theme_id is not None for plot in mem.plots.values())


def test_graph_first_evolve_generates_repair_or_dream_with_provenance() -> None:
    mem = build_memory(seed=31, architecture_mode="graph_first")

    plot = mem.ingest("你总是否定我，让我感觉自己不值得被接住。", actors=("user", "agent"))
    anchor_id = mem.core_anchor_ids[0]
    mem.graph.ensure_edge(
        anchor_id,
        plot.id,
        "contradicts_self",
        sign=-1,
        weight=0.9,
        confidence=0.9,
        provenance="test",
    )
    mem.graph.ensure_edge(
        plot.id,
        anchor_id,
        "contradicts_self",
        sign=-1,
        weight=0.9,
        confidence=0.9,
        provenance="test",
    )

    evolved = mem.evolve(dreams=1)

    assert evolved
    assert {item.source for item in evolved} <= {"dream", "repair"}
    assert any(
        mem.graph.edge_belief(src, dst).edge_type in {"dreams_about", "resolves"}
        for src, dst in mem.graph.g.edges()
        if src in {item.id for item in evolved} or dst in {item.id for item in evolved}
    )
