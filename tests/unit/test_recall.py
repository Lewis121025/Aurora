from __future__ import annotations

import time

from aurora.memory.recall import SALIENCE_FLOOR, build_activation_channels, recent_recall
from aurora.memory.store import MemoryStore
from aurora.runtime.contracts import TraceChannel


def test_recent_recall_returns_empty_for_unknown_relation() -> None:
    store = MemoryStore()
    assert recent_recall(store, "rel:unknown") == ()


def test_recent_recall_ranks_by_salience() -> None:
    store = MemoryStore()
    now = time.time()
    low = store.create_fragment("rel:a", "t1", "low salience", 0.5, 0.1, 0.1, now)
    high = store.create_fragment("rel:a", "t2", "high salience", 0.5, 0.9, 0.1, now)

    recalled = recent_recall(store, "rel:a", limit=2, now_ts=now)
    assert recalled[0].fragment_id == high.fragment_id


def test_salience_floor_excludes_faded_fragments() -> None:
    store = MemoryStore()
    now = time.time()
    store.create_fragment("rel:a", "t1", "faded memory", 0.5, SALIENCE_FLOOR - 0.01, 0.1, now)
    visible = store.create_fragment("rel:a", "t2", "visible memory", 0.5, 0.3, 0.1, now)

    recalled = recent_recall(store, "rel:a", limit=8, now_ts=now)
    assert len(recalled) == 1
    assert recalled[0].fragment_id == visible.fragment_id


def test_recency_favors_recently_touched_fragments() -> None:
    store = MemoryStore()
    old_ts = time.time() - 72 * 3600
    new_ts = time.time()
    old = store.create_fragment("rel:a", "t1", "old memory text here", 0.5, 0.5, 0.1, old_ts)
    fresh = store.create_fragment("rel:a", "t2", "fresh memory text here", 0.5, 0.5, 0.1, new_ts)

    recalled = recent_recall(store, "rel:a", limit=2, now_ts=new_ts)
    assert recalled[0].fragment_id == fresh.fragment_id


def test_build_activation_channels_empty_on_no_traces() -> None:
    store = MemoryStore()
    now = time.time()
    frag = store.create_fragment("rel:a", "t1", "hello", 0.5, 0.5, 0.3, now)

    channels = build_activation_channels(store, (frag,))
    assert channels == ()


def test_build_activation_channels_ranks_by_intensity() -> None:
    store = MemoryStore()
    now = time.time()
    frag = store.create_fragment("rel:a", "t1", "hello", 0.5, 0.5, 0.3, now)
    store.create_trace("rel:a", frag.fragment_id, TraceChannel.WARMTH, 0.9, now)
    store.create_trace("rel:a", frag.fragment_id, TraceChannel.COHERENCE, 0.2, now)

    channels = build_activation_channels(store, (frag,))
    assert channels[0] == TraceChannel.WARMTH
