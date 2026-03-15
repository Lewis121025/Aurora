from __future__ import annotations

from dataclasses import dataclass, field

from aurora.runtime.contracts import AuroraMove, TraceChannel


EvidenceMap = dict[str, tuple[str, ...]]


def _default_self_evidence() -> EvidenceMap:
    return {"recognition": (), "fragility": (), "openness": (), "agency": ()}


def _default_world_evidence() -> EvidenceMap:
    return {"welcome": (), "risk": (), "mystery": (), "stability": ()}


def _default_relation_evidence() -> EvidenceMap:
    return {"closeness": (), "distance": (), "boundary": (), "repair": ()}


def _merge_sources(
    existing: tuple[str, ...], *source_groups: tuple[str, ...], limit: int = 16
) -> tuple[str, ...]:
    merged = list(existing)
    for group in source_groups:
        for source in group:
            if not source:
                continue
            if source in merged:
                merged.remove(source)
            merged.append(source)
    return tuple(merged[-limit:])


def _append_sources(target: EvidenceMap, key: str, *source_groups: tuple[str, ...]) -> None:
    target[key] = _merge_sources(target[key], *source_groups)


def _snapshot_group(group: EvidenceMap) -> dict[str, dict[str, int | tuple[str, ...]]]:
    return {key: {"count": len(sources), "sources": sources} for key, sources in group.items()}


@dataclass(slots=True)
class Orientation:
    self_evidence: EvidenceMap = field(default_factory=_default_self_evidence)
    world_evidence: EvidenceMap = field(default_factory=_default_world_evidence)
    relation_evidence: EvidenceMap = field(default_factory=_default_relation_evidence)
    anchor_thread_ids: tuple[str, ...] = ()
    active_knot_ids: tuple[str, ...] = ()
    last_updated_at: float = 0.0

    def register_exchange(
        self,
        user_channels: tuple[TraceChannel, ...],
        aurora_move: AuroraMove,
        relation_moment_id: str,
        user_fragment_id: str,
        aurora_fragment_id: str,
        now_ts: float,
    ) -> None:
        channels = set(user_channels)
        moment_sources = (relation_moment_id,)
        exchange_sources = (relation_moment_id, user_fragment_id, aurora_fragment_id)
        if TraceChannel.RECOGNITION in channels:
            _append_sources(self.self_evidence, "recognition", exchange_sources)
            _append_sources(self.world_evidence, "welcome", exchange_sources)
            _append_sources(self.relation_evidence, "closeness", moment_sources)
        if TraceChannel.HURT in channels:
            _append_sources(self.self_evidence, "fragility", exchange_sources)
            _append_sources(self.world_evidence, "risk", exchange_sources)
        if TraceChannel.CURIOSITY in channels or TraceChannel.WONDER in channels:
            _append_sources(self.world_evidence, "mystery", exchange_sources)
        if TraceChannel.BOUNDARY in channels:
            _append_sources(self.relation_evidence, "boundary", moment_sources)
            _append_sources(self.world_evidence, "risk", exchange_sources)
        if TraceChannel.REPAIR in channels:
            _append_sources(self.relation_evidence, "repair", moment_sources)
            _append_sources(self.world_evidence, "stability", exchange_sources)
        if TraceChannel.DISTANCE in channels:
            _append_sources(self.relation_evidence, "distance", moment_sources)

        if aurora_move in {"approach", "repair"}:
            _append_sources(self.self_evidence, "openness", moment_sources)
            _append_sources(self.self_evidence, "agency", moment_sources)
            _append_sources(self.relation_evidence, "closeness", moment_sources)
        elif aurora_move in {"withhold", "silence"}:
            _append_sources(self.relation_evidence, "distance", moment_sources)
            _append_sources(self.self_evidence, "agency", moment_sources)
        elif aurora_move == "boundary":
            _append_sources(self.relation_evidence, "boundary", moment_sources)
            _append_sources(self.self_evidence, "agency", moment_sources)

        self.last_updated_at = now_ts

    def absorb_sleep(
        self,
        thread_ids: tuple[str, ...],
        knot_ids: tuple[str, ...],
        dominant_channels: tuple[TraceChannel, ...],
        now_ts: float,
    ) -> None:
        self.anchor_thread_ids = _merge_sources(self.anchor_thread_ids, thread_ids, limit=12)
        self.active_knot_ids = _merge_sources(self.active_knot_ids, knot_ids, limit=12)
        channels = set(dominant_channels)
        dominant_sources = knot_ids or thread_ids
        if TraceChannel.WARMTH in channels or TraceChannel.RECOGNITION in channels:
            _append_sources(self.world_evidence, "welcome", dominant_sources)
            _append_sources(self.world_evidence, "stability", dominant_sources)
        if TraceChannel.HURT in channels or TraceChannel.BOUNDARY in channels:
            _append_sources(self.world_evidence, "risk", dominant_sources)
            _append_sources(self.self_evidence, "fragility", dominant_sources)
        if TraceChannel.REPAIR in channels:
            _append_sources(self.relation_evidence, "repair", dominant_sources)
        if TraceChannel.BOUNDARY in channels:
            _append_sources(self.relation_evidence, "boundary", dominant_sources)
        self.last_updated_at = now_ts

    def snapshot(
        self,
    ) -> dict[str, dict[str, dict[str, int | tuple[str, ...]]] | tuple[str, ...]]:
        return {
            "self": _snapshot_group(self.self_evidence),
            "world": _snapshot_group(self.world_evidence),
            "relation": _snapshot_group(self.relation_evidence),
            "anchor_threads": self.anchor_thread_ids,
            "active_knots": self.active_knot_ids,
        }
