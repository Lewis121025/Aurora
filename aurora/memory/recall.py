from __future__ import annotations

import math
from typing import TYPE_CHECKING

from aurora.memory.affinity import structural_pressure
from aurora.memory.fragment import Fragment
from aurora.runtime.contracts import TraceChannel

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore


SALIENCE_WEIGHT = 0.30
UNRESOLVEDNESS_WEIGHT = 0.22
ACTIVATION_WEIGHT = 0.12
STRUCTURAL_WEIGHT = 0.10
THREAD_KNOT_WEIGHT = 0.10
RECENCY_WEIGHT = 0.16
ACTIVATION_CAP = 4.0
RECENCY_HALF_LIFE_HOURS = 24.0
SALIENCE_FLOOR = 0.06


def _recall_score(
    store: MemoryStore, item: Fragment, now_ts: float,
) -> float:
    hours_since_touch = max(0.0, (now_ts - item.last_touched_at) / 3600.0)
    recency = math.exp(-hours_since_touch / RECENCY_HALF_LIFE_HOURS)
    activation = min(math.log1p(item.activation_count) / math.log1p(ACTIVATION_CAP), 1.0)
    thread_knot = min(
        1.0,
        math.log1p(len(item.thread_ids)) * 0.5 + math.log1p(len(item.knot_ids)) * 0.6,
    )
    return (
        SALIENCE_WEIGHT * item.salience
        + UNRESOLVEDNESS_WEIGHT * item.unresolvedness
        + ACTIVATION_WEIGHT * activation
        + STRUCTURAL_WEIGHT * structural_pressure(store, item)
        + THREAD_KNOT_WEIGHT * thread_knot
        + RECENCY_WEIGHT * recency
    )


def recent_recall(
    store: MemoryStore, relation_id: str, limit: int = 8, now_ts: float = 0.0,
) -> tuple[Fragment, ...]:
    candidates = (
        f for f in store.fragments_for_relation(relation_id)
        if f.salience >= SALIENCE_FLOOR
    )
    ranked = sorted(
        candidates,
        key=lambda item: _recall_score(store, item, now_ts),
        reverse=True,
    )
    return tuple(ranked[:limit])


def build_activation_channels(
    store: MemoryStore, fragments: tuple[Fragment, ...]
) -> tuple[TraceChannel, ...]:
    scores: dict[TraceChannel, float] = {}
    for fragment in fragments:
        for trace in store.traces_for_fragment(fragment.fragment_id):
            scores[trace.channel] = (
                scores.get(trace.channel, 0.0) + trace.intensity * trace.carry
            )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return tuple(channel for channel, _ in ranked[:4])
