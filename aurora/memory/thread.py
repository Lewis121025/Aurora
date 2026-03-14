from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class Thread:
    thread_id: str
    relation_id: str
    fragment_ids: tuple[str, ...]
    dominant_channels: tuple[TraceChannel, ...]
    tension: float
    coherence: float
    created_at: float
    last_rewoven_at: float
