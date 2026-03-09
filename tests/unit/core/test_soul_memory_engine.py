from __future__ import annotations

from aurora.core.soul_memory import AuroraSoulMemory, SoulMemoryConfig
from aurora.integrations.embeddings.hash import HashEmbedding


def test_soul_memory_round_trip_preserves_identity_and_plots() -> None:
    mem = AuroraSoulMemory(
        cfg=SoulMemoryConfig(dim=64, metric_rank=16, max_plots=128),
        seed=7,
        embedder=HashEmbedding(dim=64),
    )

    first = None
    for text in ["对方冷淡地拒绝我。", "你又一次忽视我。", "我有点害怕被丢下。"]:
        first = mem.ingest(text, actors=("user", "agent"))
        if first.story_id is not None:
            break
    assert first is not None
    assert first.story_id is not None
    assert mem.identity.phase == "dependent_child"

    state = mem.to_state_dict()
    restored = AuroraSoulMemory.from_state_dict(
        state,
        embedder=HashEmbedding(dim=64),
    )

    assert restored.identity.phase == mem.identity.phase
    assert restored.snapshot_identity().to_state_dict()["phase"] == "dependent_child"
    assert len(restored.plots) >= 1
    assert len(restored.stories) >= 1
    assert restored.narrative_summary().phase == "dependent_child"


def test_soul_memory_feedback_and_evolve_keep_new_observables_available() -> None:
    mem = AuroraSoulMemory(
        cfg=SoulMemoryConfig(dim=64, metric_rank=16, max_plots=128),
        seed=11,
        embedder=HashEmbedding(dim=64),
    )

    mem.ingest("你总是不理我，让我很难过。", actors=("user", "agent"))
    trace = mem.query("她现在是什么状态？", k=4)
    assert trace.ranked

    top_id, _, _ = trace.ranked[0]
    mem.feedback_retrieval("她现在是什么状态？", top_id, True)
    dreams = mem.evolve(dreams=1)

    snapshot = mem.snapshot_identity()
    assert snapshot.phase in {"dependent_child", "guarded_teen", "exploratory_youth", "integrated_self"}
    assert snapshot.dream_count >= len(dreams)
