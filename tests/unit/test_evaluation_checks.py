from __future__ import annotations

from pathlib import Path

from aurora.evaluation.continuity import evaluate_continuity
from aurora.evaluation.relation_dynamics import evaluate_relation_dynamics
from aurora.evaluation.sleep_effects import evaluate_sleep_effects, snapshot_sleep_state
from aurora.runtime.engine import AuroraEngine


def test_continuity_check_accepts_live_runtime_state(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s1", text="谢谢你理解我")

    report = evaluate_continuity(
        state=engine.state,
        memory_store=engine.memory_store,
        relation_store=engine.relation_store,
    )

    assert report.ok


def test_relation_dynamics_check_tracks_boundary_and_repair_history(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s1", text="不要继续，边界在这里")
    engine.handle_turn(session_id="s1", text="对不起，我想修复")
    engine.sleep()

    report = evaluate_relation_dynamics(
        relation_store=engine.relation_store,
        memory_store=engine.memory_store,
        relation_id="rel:s1",
    )

    assert report.ok
    assert report.boundary_events >= 1
    assert report.repair_events >= 1


def test_sleep_effects_check_detects_reweave_output(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s1", text="谢谢你理解我")
    engine.handle_turn(session_id="s1", text="我还是有点受伤，也有边界冲突")

    before = snapshot_sleep_state(engine.state, engine.memory_store)
    engine.sleep()
    after = snapshot_sleep_state(engine.state, engine.memory_store)
    report = evaluate_sleep_effects(before, after)

    assert report.ok
    assert report.thread_delta >= 1
