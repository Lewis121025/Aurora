from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from aurora.runtime.models import AuroraMove, RelationMoment, RelationState, TraceChannel, clamp


class RelationStore:
    def __init__(self) -> None:
        self.states: dict[str, RelationState] = {}
        self.moments: dict[str, list[RelationMoment]] = defaultdict(list)

    def state_for(self, relation_id: str) -> RelationState:
        if relation_id not in self.states:
            self.states[relation_id] = RelationState(relation_id=relation_id)
        return self.states[relation_id]

    def record_exchange(
        self,
        relation_id: str,
        user_turn_id: str,
        aurora_turn_id: str | None,
        user_channels: tuple[TraceChannel, ...],
        aurora_move: AuroraMove,
        summary: str,
        now_ts: float,
    ) -> RelationMoment:
        state = self.state_for(relation_id)
        channels = tuple(sorted(set(user_channels), key=lambda item: item.value))
        moment = RelationMoment(
            moment_id=f"moment_{uuid4().hex[:12]}",
            relation_id=relation_id,
            user_turn_id=user_turn_id,
            aurora_turn_id=aurora_turn_id,
            user_channels=channels,
            aurora_move=aurora_move,
            boundary_crossed=TraceChannel.BOUNDARY in channels or aurora_move == "boundary",
            repair_attempted=TraceChannel.REPAIR in channels or aurora_move == "repair",
            summary=summary,
            created_at=now_ts,
        )
        self.moments[relation_id].append(moment)
        self._update_state(state=state, moment=moment, now_ts=now_ts)
        return moment

    def summarize_relation(self, relation_id: str) -> dict[str, float | tuple[str, ...]]:
        return self.state_for(relation_id).snapshot()

    def _update_state(self, state: RelationState, moment: RelationMoment, now_ts: float) -> None:
        channels = set(moment.user_channels)
        trust_delta = 0.0
        reciprocity_delta = 0.0
        tension_delta = 0.0
        distance_delta = 0.0
        repairability_delta = 0.0

        if TraceChannel.WARMTH in channels:
            trust_delta += 0.08
            reciprocity_delta += 0.04
        if TraceChannel.RECOGNITION in channels:
            trust_delta += 0.07
            reciprocity_delta += 0.05
        if TraceChannel.CURIOSITY in channels:
            reciprocity_delta += 0.04
        if TraceChannel.HURT in channels:
            tension_delta += 0.10
            repairability_delta -= 0.04
        if TraceChannel.BOUNDARY in channels:
            tension_delta += 0.12
            distance_delta += 0.06
        if TraceChannel.DISTANCE in channels:
            distance_delta += 0.08
            reciprocity_delta -= 0.03
        if TraceChannel.REPAIR in channels:
            repairability_delta += 0.08
            trust_delta += 0.03

        if moment.aurora_move == "approach":
            reciprocity_delta += 0.06
            distance_delta -= 0.03
        elif moment.aurora_move == "withhold":
            distance_delta += 0.05
        elif moment.aurora_move == "boundary":
            tension_delta += 0.05
            distance_delta += 0.08
        elif moment.aurora_move == "repair":
            repairability_delta += 0.10
            trust_delta += 0.04
        elif moment.aurora_move == "silence":
            distance_delta += 0.03

        state.trust = clamp(state.trust + trust_delta)
        state.reciprocity = clamp(state.reciprocity + reciprocity_delta)
        state.boundary_tension = clamp(state.boundary_tension + tension_delta)
        state.distance = clamp(state.distance + distance_delta)
        state.repairability = clamp(state.repairability + repairability_delta)
        state.last_contact_at = now_ts
