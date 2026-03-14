from __future__ import annotations

from pathlib import Path

from aurora.runtime.engine import AuroraEngine
from aurora.runtime.models import Phase


def test_awake_creates_relation_objects_instead_of_profile_state(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))

    output = engine.handle_turn(session_id="s1", text="谢谢你理解我，我有点害怕")

    assert output.turn_id
    assert output.response_text
    assert output.aurora_move in {"approach", "withhold", "boundary", "repair", "silence"}
    assert engine.state.being.phase is Phase.AWAKE
    assert len(engine.memory_store.fragments) == 2
    assert len(engine.memory_store.traces) >= 2
    assert len(engine.relation_store.states) == 1


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
