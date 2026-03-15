from __future__ import annotations

from pathlib import Path

from aurora.relation.projectors import project_relation
from aurora.runtime.engine import AuroraEngine

from tests.conftest import ContextAwareLLM, StubLLM


def test_relation_projection_stays_derived_from_formation(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())
    engine.handle_turn(session_id="s1", text="不要继续，边界在这里")
    engine.handle_turn(session_id="s1", text="对不起，我想修复")

    formation = engine.relation_store.formation_for("rel:s1")
    projection = project_relation(formation)

    assert projection["boundary_events"] >= 1
    assert projection["repair_events"] >= 1
    assert 0.0 <= projection["trust"] <= 1.0
    assert 0.0 <= projection["distance"] <= 1.0


def test_runtime_state_summary_is_projection_only(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="s1", text="谢谢你理解我")
    summary = engine.state_summary()

    assert summary["phase"] == "awake"
    assert summary["turns"] == 1
    assert summary["memory_fragments"] >= 2
    assert "sleep_need" in summary
