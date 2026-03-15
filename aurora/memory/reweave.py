from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class NarrativeRegion:
    relation_id: str
    anchor_fragment_id: str
    fragment_ids: tuple[str, ...]
    dominant_channels: tuple[TraceChannel, ...]
    tension: float
    coherence: float
    support: float


@dataclass(frozen=True, slots=True)
class SleepMutation:
    created_thread_ids: tuple[str, ...]
    created_knot_ids: tuple[str, ...]
    strengthened_edge_ids: tuple[str, ...]
    softened_fragment_ids: tuple[str, ...]
    affected_relation_ids: tuple[str, ...]
    recall_bias: dict[str, tuple[str, ...]]
