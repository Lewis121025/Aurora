from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from aurora.memory.affinity import neighbor_fragment_ids
from aurora.memory.recall import recent_recall
from aurora.runtime.contracts import clamp

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore


HOVER_SALIENCE_DELTA = 0.02
HOVER_UNRESOLVED_DELTA = -0.01
HOVER_NEIGHBOR_SALIENCE_DELTA = 0.01
HOVER_NEIGHBOR_UNRESOLVED_DELTA = -0.005
HOVER_THREAD_SALIENCE_DELTA = 0.015
HOVER_THREAD_UNRESOLVED_DELTA = -0.01

DECAY_SALIENCE_RATE = 0.012
DECAY_SALIENCE_CAP = 0.12
DECAY_UNRESOLVED_RATE = 0.01
DECAY_UNRESOLVED_CAP = 0.09
TRACE_DECAY_HOUR_DIVISOR = 48.0
TRACE_DECAY_CAP = 0.2
VIVIDNESS_RESISTANCE = 0.5


def hover_for_doze(store: MemoryStore, relation_ids: tuple[str, ...], now_ts: float) -> None:
    for relation_id in relation_ids:
        recalled = recent_recall(store, relation_id=relation_id, limit=4, now_ts=now_ts)
        for fragment in recalled[:2]:
            store.set_fragment(fragment.fragment_id, store.fragments[fragment.fragment_id].touched(
                at=now_ts,
                delta_salience=HOVER_SALIENCE_DELTA,
                delta_unresolved=HOVER_UNRESOLVED_DELTA,
            ))
            for nid in neighbor_fragment_ids(store, fragment.fragment_id)[:2]:
                store.set_fragment(nid, store.fragments[nid].touched(
                    at=now_ts,
                    delta_salience=HOVER_NEIGHBOR_SALIENCE_DELTA,
                    delta_unresolved=HOVER_NEIGHBOR_UNRESOLVED_DELTA,
                ))

        threads = store.threads_for_relation(relation_id)
        if threads:
            recent_thread = max(threads, key=lambda item: item.last_rewoven_at)
            for fragment_id in recent_thread.fragment_ids[:2]:
                if fragment_id in store.fragments:
                    store.set_fragment(fragment_id, store.fragments[fragment_id].touched(
                        at=now_ts,
                        delta_salience=HOVER_THREAD_SALIENCE_DELTA,
                        delta_unresolved=HOVER_THREAD_UNRESOLVED_DELTA,
                    ))


def decay_for_doze(
    store: MemoryStore, now_ts: float, relation_ids: tuple[str, ...] = (),
) -> None:
    target_fids: set[str] = set()
    for rel_id in relation_ids:
        target_fids.update(store.relation_fragments.get(rel_id, ()))
    if not target_fids:
        return

    for fragment_id in target_fids:
        fragment = store.fragments.get(fragment_id)
        if fragment is None:
            continue
        hours = max(0.0, (now_ts - fragment.last_touched_at) / 3600.0)
        resistance = 1.0 - fragment.vividness * VIVIDNESS_RESISTANCE
        salience_drop = min(DECAY_SALIENCE_CAP, hours * DECAY_SALIENCE_RATE * resistance)
        unresolved_drop = min(DECAY_UNRESOLVED_CAP, hours * DECAY_UNRESOLVED_RATE * resistance)
        store.set_fragment(fragment_id, replace(
            fragment,
            salience=clamp(fragment.salience - salience_drop),
            unresolvedness=clamp(fragment.unresolvedness - unresolved_drop),
            last_touched_at=now_ts,
        ))

    target_tids: set[str] = set()
    for fid in target_fids:
        target_tids.update(store.fragment_traces.get(fid, ()))

    for trace_id in target_tids:
        trace = store.traces.get(trace_id)
        if trace is None:
            continue
        hours = max(0.0, (now_ts - trace.last_touched_at) / 3600.0)
        next_intensity = clamp(
            trace.intensity - trace.carry * min(TRACE_DECAY_CAP, hours / TRACE_DECAY_HOUR_DIVISOR)
        )
        store.set_trace(trace_id, replace(trace, intensity=next_intensity, last_touched_at=now_ts))
