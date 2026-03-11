from __future__ import annotations

import numpy as np

from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.soul.engine import AuroraSoul, SoulConfig
from aurora.soul.extractors import CombinatorialNarrativeProvider, HeuristicMeaningProvider
from aurora.soul.models import Message, TextPart
from tests.helpers.query_router import build_test_query_analyzer


def build_memory(seed: int) -> AuroraSoul:
    embedder = HashEmbedding(dim=64, seed=seed)
    return AuroraSoul(
        cfg=SoulConfig(
            dim=64,
            metric_rank=16,
            max_plots=128,
        ),
        seed=seed,
        event_embedder=embedder,
        axis_embedder=embedder,
        meaning_provider=HeuristicMeaningProvider(),
        narrator=CombinatorialNarrativeProvider(),
        query_analyzer=build_test_query_analyzer(),
    )


def _messages(text: str) -> list[Message]:
    return [Message(role="user", parts=(TextPart(text=text),))]


def test_soul_memory_round_trip_preserves_identity_and_plots() -> None:
    mem = build_memory(seed=7)

    first = None
    for text in ["对方冷淡地拒绝我。", "你又一次忽视我。", "我有点害怕被丢下。"]:
        first = mem.ingest(_messages(text))
    mem.query(_messages("我们围绕什么关系张力反复出现过"), k=6)
    assert first is not None
    assert any(plot.story_id is not None for plot in mem.plots.values())
    assert mem.identity.current_mode_label

    state = mem.to_state_dict()
    restored = AuroraSoul.from_state_dict(
        state,
        event_embedder=HashEmbedding(dim=64, seed=7),
        axis_embedder=HashEmbedding(dim=64, seed=7),
        meaning_provider=HeuristicMeaningProvider(),
        narrator=CombinatorialNarrativeProvider(),
        query_analyzer=build_test_query_analyzer(),
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

    mem.ingest(_messages("你总是不理我，让我很难过。"))
    trace = mem.query(_messages("她现在是什么状态？"), k=4)
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
        query_analyzer=build_test_query_analyzer(),
    )
    mem.ingest(_messages("你突然安静下来，让我有点不确定。"))

    restored = AuroraSoul.from_state_dict(
        mem.to_state_dict(),
        event_embedder=embedder,
        axis_embedder=embedder,
        meaning_provider=BootstrapOnlyMeaning(),
        narrator=BootstrapOnlyNarrator(),
        query_analyzer=build_test_query_analyzer(),
    )

    assert restored.identity.current_mode_label == mem.identity.current_mode_label
    assert np.linalg.norm(restored.identity.self_vector) >= 0.0


def test_restore_rejects_removed_architecture_modes() -> None:
    mem = build_memory(seed=21)
    mem.ingest(_messages("你看起来比昨天更放松了一点。"))

    state = mem.to_state_dict()
    state["cfg"]["architecture_mode"] = "shadow"

    try:
        AuroraSoul.from_state_dict(
            state,
            event_embedder=HashEmbedding(dim=64, seed=21),
            axis_embedder=HashEmbedding(dim=64, seed=21),
            meaning_provider=HeuristicMeaningProvider(),
            narrator=CombinatorialNarrativeProvider(),
            query_analyzer=build_test_query_analyzer(),
        )
    except ValueError as exc:
        assert "Unsupported snapshot architecture_mode" in str(exc)
    else:
        raise AssertionError("restore should reject removed architecture modes")


def test_soul_config_rejects_removed_architecture_modes() -> None:
    try:
        SoulConfig(architecture_mode="shadow")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "graph_first" in str(exc)
    else:
        raise AssertionError("SoulConfig should reject removed architecture modes")


def test_graph_first_materializes_story_and_theme_views_on_query() -> None:
    mem = build_memory(seed=29)

    for text in [
        "我们反复讨论本地优先的长期记忆。",
        "你又提到结构化记忆和检索一致性。",
        "我们继续围绕图记忆和主题整理展开。",
    ]:
        plot = mem.ingest(_messages(text))
        assert plot.story_id is None

    trace = mem.query(_messages("我们之前围绕什么主题讨论过"), k=6)

    assert trace.ranked
    assert mem.stories
    assert mem.themes
    assert any(plot.story_id is not None for plot in mem.plots.values())
    assert any(plot.theme_id is not None for plot in mem.plots.values())


def test_query_does_not_refresh_views_on_hot_path() -> None:
    mem = build_memory(seed=30)
    mem.ingest(_messages("我们反复围绕图检索和主题聚类展开。"))

    def _fail_build(*args, **kwargs):
        raise AssertionError("query should not rebuild materialized views")

    mem.view_builder.build = _fail_build  # type: ignore[assignment]
    trace = mem.query(_messages("图检索"), k=4)

    assert trace.ranked is not None


def test_graph_first_evolve_generates_repair_or_dream_with_provenance() -> None:
    mem = build_memory(seed=31)

    plot = mem.ingest(_messages("你总是否定我，让我感觉自己不值得被接住。"))
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
        for src, dst in mem.graph.iter_edges()
        if src in {item.id for item in evolved} or dst in {item.id for item in evolved}
    )
