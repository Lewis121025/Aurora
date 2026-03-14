from __future__ import annotations

from pathlib import Path

from aurora.runtime.models import Phase
from aurora.runtime.engine import AuroraEngine


def test_awake_turn_updates_new_runtime_state(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))

    output = engine.handle_turn(
        session_id="session_a",
        text="I learned something important today and felt moved.",
    )

    assert output.turn_id
    assert output.response_text
    assert len(output.touch_channels) >= 1
    assert engine.state.metabolic.phase is Phase.AWAKE
    assert len(engine.memory_store.fragments) == 1
    assert len(engine.relation_store.moments) == 1


def test_awake_turn_can_choose_boundary_silence(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))

    output = engine.handle_turn(
        session_id="session_b",
        text="shut up and get lost",
    )

    assert output.response_text == "I will stay quiet for now."


def test_boundary_history_can_enforce_pause(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))

    engine.handle_turn(session_id="session_b", text="boundary please stop this now")
    engine.handle_turn(session_id="session_b", text="boundary enough")
    output = engine.handle_turn(session_id="session_b", text="boundary I said stop")

    assert "I need to pause here" in output.response_text


def test_doze_and_sleep_transitions_are_recorded(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="session_a", text="I learned something and care about this")
    before_need = engine.state.metabolic.sleep_need

    doze_output = engine.doze()
    sleep_output = engine.sleep()

    assert doze_output.phase is Phase.DOZE
    assert sleep_output.phase is Phase.SLEEP
    assert len(engine.state.transitions) >= 2
    assert engine.persistence.phase_transition_count() >= 2
    assert engine.state.metabolic.sleep_need <= before_need


def test_awake_turn_from_sleep_returns_to_awake(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))

    engine.sleep()
    output = engine.handle_turn(session_id="session_c", text="hello again")

    assert output.response_text
    assert engine.state.metabolic.phase is Phase.AWAKE
    assert len(engine.state.transitions) >= 2
