from __future__ import annotations

from pathlib import Path

from aurora.field_engine import MemoryKernel
from aurora.store import SQLiteSnapshotStore


def test_store_round_trips_kernel_snapshot(tmp_path: Path) -> None:
    store = SQLiteSnapshotStore(tmp_path / "aurora.sqlite")
    kernel = MemoryKernel(seed=13)

    kernel.ingest("I live in Hangzhou.", metadata={"speaker": "user"})
    kernel.ingest("I like tea.", metadata={"speaker": "user"})
    snapshot_id = store.save_snapshot(kernel, reason="ingest", operation_summary={"count": 2})

    loaded = store.load_latest_kernel()

    assert snapshot_id == 1
    assert loaded is not None
    assert loaded.step == kernel.step
    assert set(loaded.atoms) == set(kernel.atoms)
    assert store.list_operations(limit=1)[0]["op_kind"] == "ingest"
    store.close()


def test_store_persists_session_turns_in_order(tmp_path: Path) -> None:
    store = SQLiteSnapshotStore(tmp_path / "aurora.sqlite")

    store.append_session_turn("session-a", "user", "Hello", event_id="evt-1", created_at=1.0)
    store.append_session_turn("session-a", "assistant", "Hi", event_id="evt-2", created_at=2.0)
    turns = store.list_session_turns("session-a")

    assert [(turn.ordinal, turn.role, turn.text) for turn in turns] == [
        (1, "user", "Hello"),
        (2, "assistant", "Hi"),
    ]
    assert store.session_count() == 1
    store.close()


def test_store_prunes_old_snapshots(tmp_path: Path) -> None:
    store = SQLiteSnapshotStore(tmp_path / "aurora.sqlite", max_snapshots=2)
    kernel = MemoryKernel(seed=13)

    for index in range(3):
        kernel.ingest(f"I saw event {index}.", metadata={"speaker": "user"})
        store.save_snapshot(kernel, reason=f"step-{index}")

    assert store.snapshot_count() == 2
    assert store.latest_snapshot_meta()["reason"] == "step-2"
    store.close()
