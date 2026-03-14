from __future__ import annotations

from dataclasses import dataclass, field

from aurora.runtime.contracts import AuroraMove, TraceChannel


def _default_self_evidence() -> dict[str, int]:
    return {"recognition": 0, "fragility": 0, "openness": 0, "agency": 0}


def _default_world_evidence() -> dict[str, int]:
    return {"welcome": 0, "risk": 0, "mystery": 0, "stability": 0}


def _default_relation_evidence() -> dict[str, int]:
    return {"closeness": 0, "distance": 0, "boundary": 0, "repair": 0}


@dataclass(slots=True)
class Orientation:
    self_evidence: dict[str, int] = field(default_factory=_default_self_evidence)
    world_evidence: dict[str, int] = field(default_factory=_default_world_evidence)
    relation_evidence: dict[str, int] = field(default_factory=_default_relation_evidence)
    anchor_thread_ids: tuple[str, ...] = ()
    active_knot_ids: tuple[str, ...] = ()
    last_updated_at: float = 0.0

    def register_exchange(
        self,
        user_channels: tuple[TraceChannel, ...],
        aurora_move: AuroraMove,
        now_ts: float,
    ) -> None:
        channels = set(user_channels)
        if TraceChannel.RECOGNITION in channels:
            self.self_evidence["recognition"] += 1
            self.world_evidence["welcome"] += 1
            self.relation_evidence["closeness"] += 1
        if TraceChannel.HURT in channels:
            self.self_evidence["fragility"] += 1
            self.world_evidence["risk"] += 1
        if TraceChannel.CURIOSITY in channels or TraceChannel.WONDER in channels:
            self.world_evidence["mystery"] += 1
        if TraceChannel.BOUNDARY in channels:
            self.relation_evidence["boundary"] += 1
            self.world_evidence["risk"] += 1
        if TraceChannel.REPAIR in channels:
            self.relation_evidence["repair"] += 1
            self.world_evidence["stability"] += 1
        if TraceChannel.DISTANCE in channels:
            self.relation_evidence["distance"] += 1

        if aurora_move in {"approach", "repair"}:
            self.self_evidence["openness"] += 1
            self.self_evidence["agency"] += 1
            self.relation_evidence["closeness"] += 1
        elif aurora_move in {"withhold", "silence"}:
            self.relation_evidence["distance"] += 1
            self.self_evidence["agency"] += 1
        elif aurora_move == "boundary":
            self.relation_evidence["boundary"] += 1
            self.self_evidence["agency"] += 1

        self.last_updated_at = now_ts

    def absorb_sleep(
        self,
        thread_ids: tuple[str, ...],
        knot_ids: tuple[str, ...],
        dominant_channels: tuple[TraceChannel, ...],
        now_ts: float,
    ) -> None:
        self.anchor_thread_ids = tuple([*self.anchor_thread_ids, *thread_ids][-12:])
        self.active_knot_ids = tuple([*self.active_knot_ids, *knot_ids][-12:])
        channels = set(dominant_channels)
        if TraceChannel.WARMTH in channels or TraceChannel.RECOGNITION in channels:
            self.world_evidence["welcome"] += 1
            self.world_evidence["stability"] += 1
        if TraceChannel.HURT in channels or TraceChannel.BOUNDARY in channels:
            self.world_evidence["risk"] += 1
            self.self_evidence["fragility"] += 1
        self.last_updated_at = now_ts

    def snapshot(self) -> dict[str, dict[str, int] | tuple[str, ...]]:
        return {
            "self": dict(self.self_evidence),
            "world": dict(self.world_evidence),
            "relation": dict(self.relation_evidence),
            "anchor_threads": self.anchor_thread_ids,
            "active_knots": self.active_knot_ids,
        }
