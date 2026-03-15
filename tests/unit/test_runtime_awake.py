from __future__ import annotations

from pathlib import Path

from aurora.memory.recall import recent_recall
from aurora.runtime.contracts import Phase
from aurora.runtime.engine import AuroraEngine


def test_awake_writes_canonical_graph_objects(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))

    output = engine.handle_turn(session_id="s1", text="谢谢你理解我，我有点害怕")

    assert output.turn_id
    assert output.response_text
    assert output.aurora_move in {
        "approach",
        "withhold",
        "boundary",
        "repair",
        "silence",
        "witness",
    }
    assert engine.state.metabolic.phase is Phase.AWAKE
    assert len(engine.memory_store.fragments) == 2
    assert len(engine.memory_store.traces) >= 2
    assert len(engine.memory_store.associations) >= 1
    assert engine.relation_store.moment_count() == 1
    assert engine.relation_store.relation_count() == 1
    assert engine.state.metabolic.pending_sleep_relation_ids == ("rel:s1",)


def test_boundary_input_pushes_boundary_move(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))

    output = engine.handle_turn(session_id="s2", text="不要再继续，停，边界在这里")

    assert output.aurora_move == "boundary"
    assert "boundary" in output.response_text.lower()


def test_phase_transitions_are_lifecycle_events(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s3", text="我想知道我们之后会怎样")

    doze_output = engine.doze()
    sleep_output = engine.sleep()

    assert doze_output.phase is Phase.DOZE
    assert sleep_output.phase is Phase.SLEEP
    assert len(engine.state.transitions) >= 2
    assert engine.state.metabolic.pending_sleep_relation_ids == ()


def test_doze_hover_keeps_recent_relation_material_lightly_active(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s4", text="谢谢你理解我")

    engine.doze()

    recalled = recent_recall(engine.memory_store, "rel:s4", limit=4)
    assert recalled
    assert any(fragment.activation_count >= 1 for fragment in recalled)
