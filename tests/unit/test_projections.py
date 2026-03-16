from __future__ import annotations

from pathlib import Path

from aurora.runtime.engine import AuroraEngine

from tests.conftest import ContextAwareLLM, StubLLM


def test_formation_records_boundary_and_repair_events(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())
    engine.handle_turn(session_id="s1", text="不要继续，边界在这里")
    engine.handle_turn(session_id="s1", text="对不起，我想修复")

    relation_id = engine.identity.relation_for("s1")
    assert relation_id is not None
    formation = engine.relation_store.formation_for(relation_id)

    assert formation.boundary_events >= 1
    assert formation.repair_events >= 1


def test_runtime_state_summary(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="s1", text="谢谢你理解我")
    summary = engine.state_summary()

    assert summary["phase"] == "awake"
    assert summary["turns"] == 1
    assert summary["memory_fragments"] >= 2
    assert "sleep_need" in summary
