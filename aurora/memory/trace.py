from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class Trace:
    trace_id: str
    relation_id: str
    fragment_id: str
    channel: TraceChannel
    intensity: float
    carry: float
    created_at: float
    last_touched_at: float
