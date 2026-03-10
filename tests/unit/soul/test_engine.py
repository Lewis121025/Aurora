from __future__ import annotations

import numpy as np

from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.soul.engine import AuroraSoul, SoulConfig
from aurora.soul.extractors import CombinatorialNarrativeProvider, HeuristicMeaningProvider


def build_memory(seed: int) -> AuroraSoul:
    embedder = HashEmbedding(dim=64, seed=seed)
    return AuroraSoul(
        cfg=SoulConfig(dim=64, metric_rank=16, max_plots=128),
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
