from __future__ import annotations

import time

from aurora.memory.store import MemoryStore
from aurora.runtime.contracts import AssocKind, TraceChannel


def test_add_fragment_indexes_by_relation() -> None:
    store = MemoryStore()
    now = time.time()
    frag = store.create_fragment("rel:a", "t1", "hello", 0.5, 0.5, 0.3, now)

    assert frag.fragment_id in store.fragments
    assert store.fragments_for_relation("rel:a") == (frag,)
    assert store.fragments_for_relation("rel:b") == ()


def test_add_trace_indexes_by_fragment() -> None:
    store = MemoryStore()
    now = time.time()
    frag = store.create_fragment("rel:a", "t1", "hello", 0.5, 0.5, 0.3, now)
    trace = store.create_trace("rel:a", frag.fragment_id, TraceChannel.WARMTH, 0.6, now)

    assert store.traces_for_fragment(frag.fragment_id) == (trace,)
    assert store.traces_for_fragment("nonexistent") == ()


def test_add_thread_backlinks_fragments() -> None:
    store = MemoryStore()
    now = time.time()
    f1 = store.create_fragment("rel:a", "t1", "one", 0.5, 0.5, 0.3, now)
    f2 = store.create_fragment("rel:a", "t2", "two", 0.5, 0.5, 0.3, now)

    from aurora.memory.thread import Thread

    thread = Thread(
        thread_id="thread_test",
        relation_id="rel:a",
        fragment_ids=(f1.fragment_id, f2.fragment_id),
        dominant_channels=(TraceChannel.WARMTH,),
        tension=0.4,
        coherence=0.6,
        created_at=now,
        last_rewoven_at=now,
    )
    store.add_thread(thread)

    assert "thread_test" in store.fragments[f1.fragment_id].thread_ids
    assert "thread_test" in store.fragments[f2.fragment_id].thread_ids


def test_strengthen_association_merges_evidence() -> None:
    store = MemoryStore()
    now = time.time()
    f1 = store.create_fragment("rel:a", "t1", "one", 0.5, 0.5, 0.3, now)
    f2 = store.create_fragment("rel:a", "t2", "two", 0.5, 0.5, 0.3, now)

    edge1 = store.link_fragments(
        f1.fragment_id, f2.fragment_id, AssocKind.RELATION, 0.5, ("ev1",), now
    )
    edge2 = store.strengthen_association(
        f1.fragment_id, f2.fragment_id, AssocKind.RELATION, 0.8, ("ev2",), now + 1
    )

    assert edge1.edge_id == edge2.edge_id
    assert edge2.weight >= 0.8
    assert "ev1" in edge2.evidence
    assert "ev2" in edge2.evidence
