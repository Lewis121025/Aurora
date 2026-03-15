from __future__ import annotations

from pathlib import Path

from aurora.memory.recall import recent_recall
from aurora.runtime.engine import AuroraEngine

from tests.conftest import ContextAwareLLM, StubLLM


def test_sleep_reweave_creates_thread_and_knot_from_clustered_fragments(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="s", text="谢谢你理解我")
    engine.handle_turn(session_id="s", text="我还是有点受伤，也有边界冲突")

    before_threads = len(engine.memory_store.threads)
    before_knots = len(engine.memory_store.knots)
    sleep_output = engine.sleep()

    assert sleep_output.phase.value == "sleep"
    assert len(engine.memory_store.threads) >= before_threads + 1
    assert len(engine.memory_store.knots) >= before_knots
    assert engine.memory_store.sleep_cycles >= 1


def test_recall_stays_relation_scoped(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="a", text="谢谢你，我记得这件事")
    engine.handle_turn(session_id="b", text="普通更新")

    recalled = recent_recall(engine.memory_store, relation_id="rel:a", limit=4)

    assert recalled
    assert all(fragment.relation_id == "rel:a" for fragment in recalled)


def test_runtime_recovers_orientation_and_threads_from_persistence(tmp_path: Path) -> None:
    first = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())
    first.handle_turn(session_id="s", text="谢谢你理解我")
    first.handle_turn(session_id="s", text="我还是有点受伤，但想修复")
    first.sleep()

    assert any(
        source.startswith("moment_")
        for source in first.state.orientation.self_evidence["recognition"]
    )

    second = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())

    assert second.persistence.turn_count() == 2
    assert second.memory_store.sleep_cycles >= 1
    assert second.memory_store.threads or second.state.orientation.anchor_thread_ids
    assert any(
        source.startswith("moment_")
        for source in second.state.orientation.self_evidence["recognition"]
    )


def test_sleep_pushes_thread_or_knot_sources_into_orientation(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="s", text="谢谢你理解我")
    engine.handle_turn(session_id="s", text="我还是有点受伤，也有边界冲突")

    engine.sleep()

    risk_sources = engine.state.orientation.world_evidence["risk"]
    assert engine.state.orientation.anchor_thread_ids
    assert any(
        source.startswith("thread_") or source.startswith("knot_")
        for source in risk_sources + engine.state.orientation.anchor_thread_ids
    )
