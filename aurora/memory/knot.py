from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class Knot:
    knot_id: str
    relation_id: str
    fragment_ids: tuple[str, ...]
    dominant_channels: tuple[TraceChannel, ...]
    intensity: float
    resolved: bool
    created_at: float
    last_rewoven_at: float
