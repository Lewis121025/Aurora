from __future__ import annotations

from typing import TYPE_CHECKING

from aurora.memory.affinity import structural_pressure
from aurora.memory.fragment import Fragment
from aurora.runtime.contracts import TraceChannel

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore


SALIENCE_WEIGHT = 0.36
UNRESOLVEDNESS_WEIGHT = 0.26
ACTIVATION_WEIGHT = 0.14
STRUCTURAL_WEIGHT = 0.12
THREAD_KNOT_WEIGHT = 0.12
ACTIVATION_CAP = 4.0


def recent_recall(
    store: MemoryStore, relation_id: str, limit: int = 8
) -> tuple[Fragment, ...]:
    ranked = sorted(
        store.fragments_for_relation(relation_id),
        key=lambda item: (
            SALIENCE_WEIGHT * item.salience
            + UNRESOLVEDNESS_WEIGHT * item.unresolvedness
            + ACTIVATION_WEIGHT * min(item.activation_count / ACTIVATION_CAP, 1.0)
            + STRUCTURAL_WEIGHT * structural_pressure(store, item)
            + THREAD_KNOT_WEIGHT * min(1.0, len(item.thread_ids) * 0.4 + len(item.knot_ids) * 0.6)
        ),
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
