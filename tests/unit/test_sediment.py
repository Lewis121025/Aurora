from __future__ import annotations

import time

from aurora.memory.sediment import (
    ASSOCIATION_WEIGHT_FLOOR,
    SALIENCE_FLOOR,
    STALENESS_HOURS,
    UNRESOLVED_FLOOR,
    sediment,
)
from aurora.memory.store import MemoryStore
from aurora.runtime.contracts import AssocKind, TraceChannel


def test_sediment_removes_low_salience_fragments() -> None:
    store = MemoryStore()
    now = time.time()
    faded = store.create_fragment("rel:a", "t1", "faded memory", 0.5, SALIENCE_FLOOR - 0.01, UNRESOLVED_FLOOR - 0.01, now)
    alive = store.create_fragment("rel:a", "t2", "alive memory", 0.5, 0.5, 0.3, now)

    result = sediment(store, now)

    assert faded.fragment_id in result.removed_fragment_ids
    assert alive.fragment_id not in result.removed_fragment_ids
    assert faded.fragment_id not in store.fragments
    assert alive.fragment_id in store.fragments


def test_sediment_preserves_structurally_active_fragments() -> None:
    store = MemoryStore()
    now = time.time()
    f1 = store.create_fragment("rel:a", "t1", "in thread memory", 0.5, SALIENCE_FLOOR - 0.01, UNRESOLVED_FLOOR - 0.01, now)
    f2 = store.create_fragment("rel:a", "t2", "another fragment here", 0.5, 0.5, 0.3, now)
    store.create_trace("rel:a", f1.fragment_id, TraceChannel.WARMTH, 0.5, now)
    store.create_trace("rel:a", f2.fragment_id, TraceChannel.WARMTH, 0.5, now)

    from aurora.memory.thread import Thread
    store.add_thread(Thread(
        thread_id="thread_test",
        relation_id="rel:a",
        fragment_ids=(f1.fragment_id, f2.fragment_id),
        dominant_channels=(TraceChannel.WARMTH,),
        tension=0.3,
        coherence=0.5,
        created_at=now,
        last_rewoven_at=now,
    ))

    result = sediment(store, now)

    assert f1.fragment_id not in result.removed_fragment_ids
    assert f1.fragment_id in store.fragments


def test_sediment_removes_stale_fragments() -> None:
    store = MemoryStore()
    now = time.time()
    old_ts = now - STALENESS_HOURS * 3600 - 1
    stale = store.create_fragment("rel:a", "t1", "very old memory here", 0.5, 0.8, 0.5, old_ts)
    fresh = store.create_fragment("rel:a", "t2", "fresh memory text", 0.5, 0.8, 0.5, now)

    result = sediment(store, now)

    assert stale.fragment_id in result.removed_fragment_ids
    assert fresh.fragment_id not in result.removed_fragment_ids


def test_sediment_cleans_up_traces_and_associations() -> None:
    store = MemoryStore()
    now = time.time()
    f1 = store.create_fragment("rel:a", "t1", "faded one text", 0.5, SALIENCE_FLOOR - 0.01, UNRESOLVED_FLOOR - 0.01, now)
    f2 = store.create_fragment("rel:a", "t2", "faded two text", 0.5, SALIENCE_FLOOR - 0.01, UNRESOLVED_FLOOR - 0.01, now)
    trace = store.create_trace("rel:a", f1.fragment_id, TraceChannel.WARMTH, 0.5, now)
    edge = store.link_fragments(f1.fragment_id, f2.fragment_id, AssocKind.RELATION, 0.5, ("ev",), now)

    result = sediment(store, now)

    assert trace.trace_id in result.removed_trace_ids
    assert edge.edge_id in result.removed_association_ids
    assert trace.trace_id not in store.traces
    assert edge.edge_id not in store.associations


def test_sediment_removes_orphaned_threads() -> None:
    store = MemoryStore()
    now = time.time()
    old_ts = now - STALENESS_HOURS * 3600 - 1
    f1 = store.create_fragment("rel:a", "t1", "stale fragment one", 0.5, 0.8, 0.5, old_ts)
    f2 = store.create_fragment("rel:a", "t2", "stale fragment two", 0.5, 0.8, 0.5, old_ts)
    store.create_trace("rel:a", f1.fragment_id, TraceChannel.WARMTH, 0.5, old_ts)
    store.create_trace("rel:a", f2.fragment_id, TraceChannel.WARMTH, 0.5, old_ts)

    from aurora.memory.thread import Thread
    store.add_thread(Thread(
        thread_id="thread_orphan",
        relation_id="rel:a",
        fragment_ids=(f1.fragment_id, f2.fragment_id),
        dominant_channels=(TraceChannel.WARMTH,),
        tension=0.3,
        coherence=0.5,
        created_at=old_ts,
        last_rewoven_at=old_ts,
    ))

    fresh = store.create_fragment("rel:a", "t3", "fresh fragment text", 0.5, 0.8, 0.5, now)

    result = sediment(store, now)

    assert "thread_orphan" in result.removed_thread_ids
    assert "thread_orphan" not in store.threads
    assert fresh.fragment_id in store.fragments


def test_deletion_tracking_populated_after_sediment() -> None:
    store = MemoryStore()
    now = time.time()
    f = store.create_fragment("rel:a", "t1", "to be sedimented", 0.5, SALIENCE_FLOOR - 0.01, UNRESOLVED_FLOOR - 0.01, now)
    store.clear_dirty()

    sediment(store, now)

    assert f.fragment_id in store._deleted_fragments
    assert f.fragment_id not in store._dirty_fragments
