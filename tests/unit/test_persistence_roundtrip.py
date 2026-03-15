from __future__ import annotations

from pathlib import Path

from aurora.memory.recall import recent_recall
from aurora.runtime.engine import AuroraEngine


def test_persistence_roundtrip_preserves_all_graph_objects(tmp_path: Path) -> None:
    first = AuroraEngine.create(data_dir=str(tmp_path))
    first.handle_turn(session_id="s", text="谢谢你理解我")
    first.handle_turn(session_id="s", text="我还是有点受伤，也有边界冲突")
    first.sleep()

    frag_count = len(first.memory_store.fragments)
    trace_count = len(first.memory_store.traces)
    assoc_count = len(first.memory_store.associations)
    thread_count = len(first.memory_store.threads)
    knot_count = len(first.memory_store.knots)
    moment_count = first.relation_store.moment_count()

    second = AuroraEngine.create(data_dir=str(tmp_path))

    assert len(second.memory_store.fragments) == frag_count
    assert len(second.memory_store.traces) == trace_count
    assert len(second.memory_store.associations) == assoc_count
    assert len(second.memory_store.threads) == thread_count
    assert len(second.memory_store.knots) == knot_count
    assert second.relation_store.moment_count() == moment_count


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    engine.handle_turn(session_id="s", text="first turn")
    engine.handle_turn(session_id="s", text="second turn")

    frag_count = len(engine.memory_store.fragments)
    reloaded = AuroraEngine.create(data_dir=str(tmp_path))
    assert len(reloaded.memory_store.fragments) == frag_count


def test_recall_works_after_persistence_reload(tmp_path: Path) -> None:
    first = AuroraEngine.create(data_dir=str(tmp_path))
    first.handle_turn(session_id="s", text="谢谢你理解我")

    second = AuroraEngine.create(data_dir=str(tmp_path))
    recalled = recent_recall(second.memory_store, "rel:s", limit=4)

    assert len(recalled) >= 2
    assert all(f.relation_id == "rel:s" for f in recalled)
