from __future__ import annotations

from pathlib import Path

from aurora.runtime.engine import AuroraEngine


def test_sleep_reweave_softens_fragments_and_records_delta(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s", text="I love sketching with you")
    engine.handle_turn(session_id="s", text="You hurt me deeply")

    before_salience = {
        fragment_id: fragment.salience
        for fragment_id, fragment in engine.memory_store.fragments.items()
    }

    engine.sleep()

    after_salience = {
        fragment_id: fragment.salience
        for fragment_id, fragment in engine.memory_store.fragments.items()
    }
    salience_delta = max(
        abs(after_salience[fragment_id] - value) for fragment_id, value in before_salience.items()
    )

    assert engine.memory_store.sleep_cycles == 1
    assert engine.memory_store.last_reweave_delta > 0.0
    assert salience_delta <= 0.25


def test_recall_prefers_relation_context_over_global_recency(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    first = engine.handle_turn(session_id="a", text="I love this memory")
    second = engine.handle_turn(session_id="b", text="ordinary update")

    recalled = engine.memory_store.recall(relation_id="rel:a", limit=1)

    assert first.turn_id
    assert second.turn_id
    assert len(recalled) == 1
    assert recalled[0].relation_id == "rel:a"


def test_runtime_recovers_from_persisted_state(tmp_path: Path) -> None:
    first_engine = AuroraEngine.create(data_dir=str(tmp_path))
    first_engine.handle_turn(session_id="s", text="I love this memory")
    first_engine.sleep()

    second_engine = AuroraEngine.create(data_dir=str(tmp_path))

    assert len(second_engine.memory_store.fragments) == 1
    assert second_engine.memory_store.sleep_cycles == 1
    assert second_engine.persistence.turn_count() == 1
    assert second_engine.state.transitions
