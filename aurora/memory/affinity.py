from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING, Iterable

from aurora.memory.fragment import Fragment
from aurora.runtime.contracts import TraceChannel

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore


AFFINITY_RELATION_WEIGHT = 0.30
AFFINITY_KEYWORD_WEIGHT = 0.24
AFFINITY_TRACE_WEIGHT = 0.26
AFFINITY_TEMPORAL_WEIGHT = 0.20
TEMPORAL_HALF_LIFE_HOURS = 16.0


def fragment_affinity(store: MemoryStore, left: Fragment, right: Fragment) -> float:
    temporal_distance = abs(left.created_at - right.created_at) / 3600.0
    temporal = math.exp(-temporal_distance / TEMPORAL_HALF_LIFE_HOURS)
    return (
        AFFINITY_RELATION_WEIGHT * float(left.relation_id == right.relation_id)
        + AFFINITY_KEYWORD_WEIGHT * keyword_overlap(left.tags, right.tags)
        + AFFINITY_TRACE_WEIGHT * trace_overlap(store, left.fragment_id, right.fragment_id)
        + AFFINITY_TEMPORAL_WEIGHT * temporal
    )


def keyword_overlap(left_tags: Iterable[str], right_tags: Iterable[str]) -> float:
    left = set(left_tags)
    right = set(right_tags)
    return len(left & right) / max(1, len(left | right)) if (left or right) else 0.0


def trace_overlap(store: MemoryStore, left_fragment_id: str, right_fragment_id: str) -> float:
    left = {trace.channel for trace in store.traces_for_fragment(left_fragment_id)}
    right = {trace.channel for trace in store.traces_for_fragment(right_fragment_id)}
    return len(left & right) / max(1, len(left | right)) if (left or right) else 0.0


def structural_pressure(store: MemoryStore, fragment: Fragment) -> float:
    return min(1.0, len(store.fragment_edges.get(fragment.fragment_id, ())) / 6.0)


def cluster_keyword_overlap(cluster: list[Fragment]) -> float:
    if len(cluster) < 2:
        return 0.0
    overlaps = [
        keyword_overlap(left.tags, right.tags) for left, right in combinations(cluster, 2)
    ]
    return sum(overlaps) / len(overlaps)


def cluster_trace_overlap(store: MemoryStore, cluster: list[Fragment]) -> float:
    if len(cluster) < 2:
        return 0.0
    overlaps: list[float] = []
    for left, right in combinations(cluster, 2):
        left_channels = {trace.channel for trace in store.traces_for_fragment(left.fragment_id)}
        right_channels = {trace.channel for trace in store.traces_for_fragment(right.fragment_id)}
        overlaps.append(
            len(left_channels & right_channels) / max(1, len(left_channels | right_channels))
        )
    return sum(overlaps) / len(overlaps)


def cluster_dominant_channels(
    store: MemoryStore, cluster: list[Fragment]
) -> tuple[TraceChannel, ...]:
    totals: dict[TraceChannel, float] = {}
    for fragment in cluster:
        for trace in store.traces_for_fragment(fragment.fragment_id):
            totals[trace.channel] = totals.get(trace.channel, 0.0) + trace.intensity
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return tuple(channel for channel, _ in ranked[:2])


def neighbor_fragment_ids(store: MemoryStore, fragment_id: str) -> tuple[str, ...]:
    neighbors: list[str] = []
    for edge_id in store.fragment_edges.get(fragment_id, ()):
        edge = store.associations[edge_id]
        neighbor_id = (
            edge.dst_fragment_id if edge.src_fragment_id == fragment_id else edge.src_fragment_id
        )
        if neighbor_id not in neighbors:
            neighbors.append(neighbor_id)
    return tuple(neighbors)


def region_edge_density(store: MemoryStore, fragment_ids: tuple[str, ...]) -> float:
    if len(fragment_ids) < 2:
        return 0.0
    linked = 0
    total = 0
    for left_id, right_id in combinations(fragment_ids, 2):
        total += 1
        if store._existing_edge_id(left_id, right_id) is not None:
            linked += 1
    return linked / total if total else 0.0
