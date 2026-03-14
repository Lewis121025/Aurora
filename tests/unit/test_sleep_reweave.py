from __future__ import annotations

from pathlib import Path

from aurora.memory.recall import rank_fragment_ids
from aurora.runtime.engine import AuroraEngine


def test_sleep_reweave_is_conservative_and_records_delta(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s", text="I love sketching with you")
    engine.handle_turn(session_id="s", text="Today was ordinary")

    before_salience = dict(engine.memory_store.fragment_salience)
    before_narrative = dict(engine.memory_store.fragment_narrative_weight)
    before_weights = {item.association_id: item.weight for item in engine.memory_store.associations}

    engine.sleep()

    after_salience = engine.memory_store.fragment_salience
    after_narrative = engine.memory_store.fragment_narrative_weight
    after_weights = {item.association_id: item.weight for item in engine.memory_store.associations}

    salience_delta = max(
        abs(after_salience[fragment_id] - value) for fragment_id, value in before_salience.items()
    )
    weight_delta = max(
        abs(after_weights[association_id] - value)
        for association_id, value in before_weights.items()
    )
    narrative_delta = max(
        abs(after_narrative[fragment_id] - value) for fragment_id, value in before_narrative.items()
    )

    assert engine.memory_store.sleep_cycles == 1
    assert engine.memory_store.last_reweave_delta > 0.0
    assert salience_delta <= 0.051
    assert narrative_delta <= 0.051
    assert weight_delta <= 0.051


def test_recall_prefers_relation_context_over_recency(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    first = engine.handle_turn(session_id="s", text="I love this memory")
    second = engine.handle_turn(session_id="s", text="You hurt me deeply")

    warm_ranked = rank_fragment_ids(
        store=engine.memory_store,
        relation_tone="warm",
        relation_strength=1.0,
        limit=2,
    )
    cold_ranked = rank_fragment_ids(
        store=engine.memory_store,
        relation_tone="cold",
        relation_strength=1.0,
        limit=2,
    )

    first_fragment_id = engine.memory_store.fragments[0].fragment_id
    second_fragment_id = engine.memory_store.fragments[1].fragment_id

    assert len(warm_ranked) == 2
    assert len(cold_ranked) == 2
    assert first.turn_id
    assert second.turn_id
    assert warm_ranked[0] == first_fragment_id
    assert cold_ranked[0] == second_fragment_id


def test_runtime_recovers_from_persisted_state(tmp_path: Path) -> None:
    first_engine = AuroraEngine.create(data_dir=str(tmp_path))
    first_engine.handle_turn(session_id="s", text="I love this memory")
    first_engine.sleep()

    second_engine = AuroraEngine.create(data_dir=str(tmp_path))

    assert len(second_engine.memory_store.fragments) == 1
    assert second_engine.memory_store.sleep_cycles == 1
    assert second_engine.persistence.turn_count() == 1
    assert second_engine.memory_store.fragment_narrative_weight
