from __future__ import annotations

from pathlib import Path

from aurora.runtime.engine import AuroraEngine


def test_sleep_reweave_creates_chapter_from_clustered_fragments(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s", text="谢谢你理解我")
    engine.handle_turn(session_id="s", text="我还是有点受伤，但想修复")

    before_chapters = len(engine.memory_store.chapters)
    sleep_output = engine.sleep()

    assert sleep_output.phase.value == "sleep"
    assert len(engine.memory_store.chapters) >= before_chapters + 1
    assert engine.memory_store.last_reweave_delta >= 0.0


def test_recall_stays_relation_scoped(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="a", text="谢谢你，我记得这件事")
    engine.handle_turn(session_id="b", text="普通更新")

    recalled = engine.memory_store.recent_recall(relation_id="rel:a", limit=4)

    assert recalled
    assert all(fragment.relation_id == "rel:a" for fragment in recalled)


def test_runtime_recovers_being_and_chapters_from_persistence(tmp_path: Path) -> None:
    first = AuroraEngine.create(data_dir=str(tmp_path))
    first.handle_turn(session_id="s", text="谢谢你理解我")
    first.handle_turn(session_id="s", text="我还是有点受伤，但想修复")
    first.sleep()

    second = AuroraEngine.create(data_dir=str(tmp_path))

    assert second.persistence.turn_count() == 2
    assert second.memory_store.sleep_cycles == 1
    assert second.state.being.recent_chapter_bias or second.memory_store.chapters
