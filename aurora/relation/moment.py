from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import AuroraMove, TraceChannel


@dataclass(frozen=True, slots=True)
class RelationMoment:
    moment_id: str
    relation_id: str
    user_turn_id: str
    aurora_turn_id: str | None
    user_channels: tuple[TraceChannel, ...]
    aurora_move: AuroraMove
    boundary_event: bool
    repair_event: bool
    summary: str
    created_at: float
