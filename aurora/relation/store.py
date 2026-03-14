from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4

from aurora.runtime.models import (
    Orientation,
    RelationFormation,
    RelationMoment,
    RelationMove,
    TraceChannel,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class RelationStore:
    moments: dict[str, RelationMoment] = field(default_factory=dict)
    formations: dict[str, RelationFormation] = field(default_factory=dict)
    orientations: dict[str, Orientation] = field(default_factory=dict)

    def formation(self, relation_id: str) -> RelationFormation:
        existing = self.formations.get(relation_id)
        if existing is not None:
            return existing
        formation = RelationFormation(
            relation_id=relation_id,
            trust=0.0,
            familiarity=0.0,
            reciprocity=0.0,
            boundary_tension=0.1,
            repairability=0.5,
            active_thread_ids=(),
            active_knot_ids=(),
            last_contact_at=0.0,
        )
        self.formations[relation_id] = formation
        return formation

    def orientation(self, relation_id: str) -> Orientation:
        existing = self.orientations.get(relation_id)
        if existing is not None:
            return existing
        orientation = Orientation(
            relation_id=relation_id,
            self_orientation=0.0,
            world_orientation=0.0,
            relation_orientation=0.0,
            narrative_tilt=0.0,
            updated_at=0.0,
        )
        self.orientations[relation_id] = orientation
        return orientation

    def record_moment(
        self,
        relation_id: str,
        session_id: str,
        user_turn_id: str,
        aurora_turn_id: str,
        user_channels: tuple[TraceChannel, ...],
        user_move: RelationMove,
        aurora_move: RelationMove,
        created_at: float,
        note: str,
    ) -> RelationMoment:
        boundary_signal = 1.0 if TraceChannel.BOUNDARY in user_channels else 0.0
        resonance_score = (
            0.65
            if any(
                channel in {TraceChannel.WARMTH, TraceChannel.INSIGHT} for channel in user_channels
            )
            else 0.35
        )
        moment = RelationMoment(
            moment_id=f"moment_{uuid4().hex[:10]}",
            relation_id=relation_id,
            session_id=session_id,
            user_turn_id=user_turn_id,
            aurora_turn_id=aurora_turn_id,
            user_channels=user_channels,
            user_move=user_move,
            aurora_move=aurora_move,
            boundary_signal=boundary_signal,
            resonance_score=resonance_score,
            note=note,
            created_at=created_at,
        )
        self.moments[moment.moment_id] = moment
        self._evolve_from_moment(moment)
        return moment

    def _evolve_from_moment(self, moment: RelationMoment) -> None:
        formation = self.formation(moment.relation_id)
        trust = formation.trust
        familiarity = formation.familiarity
        reciprocity = formation.reciprocity
        boundary_tension = formation.boundary_tension
        repairability = formation.repairability

        if TraceChannel.WARMTH in moment.user_channels:
            trust += 0.10
            reciprocity += 0.06
        if TraceChannel.INSIGHT in moment.user_channels:
            trust += 0.05
        if TraceChannel.HURT in moment.user_channels:
            trust -= 0.12
            boundary_tension += 0.10
        if TraceChannel.BOUNDARY in moment.user_channels:
            boundary_tension += 0.18
        if moment.user_move is RelationMove.REPAIR or moment.aurora_move is RelationMove.REPAIR:
            repairability += 0.10
            trust += 0.04
        if moment.aurora_move is RelationMove.BOUNDARY:
            boundary_tension += 0.08

        updated = replace(
            formation,
            trust=_clamp(trust, -1.0, 1.0),
            familiarity=_clamp(familiarity + 0.08, 0.0, 1.0),
            reciprocity=_clamp(reciprocity, 0.0, 1.0),
            boundary_tension=_clamp(boundary_tension, 0.0, 1.0),
            repairability=_clamp(repairability, 0.0, 1.0),
            last_contact_at=moment.created_at,
        )
        self.formations[moment.relation_id] = updated

        orientation = self.orientation(moment.relation_id)
        orientation_updated = replace(
            orientation,
            self_orientation=_clamp(
                orientation.self_orientation
                + (0.05 if TraceChannel.INSIGHT in moment.user_channels else -0.01),
                -1.0,
                1.0,
            ),
            world_orientation=_clamp(
                orientation.world_orientation
                + 0.08 * updated.trust
                - 0.06 * updated.boundary_tension,
                -1.0,
                1.0,
            ),
            relation_orientation=_clamp(
                orientation.relation_orientation
                + 0.12 * updated.reciprocity
                - 0.10 * updated.boundary_tension,
                -1.0,
                1.0,
            ),
            narrative_tilt=_clamp(
                orientation.narrative_tilt
                + 0.08 * updated.repairability
                - 0.05 * updated.boundary_tension,
                -1.0,
                1.0,
            ),
            updated_at=moment.created_at,
        )
        self.orientations[moment.relation_id] = orientation_updated

    def absorb_reweave(
        self,
        relation_id: str,
        thread_ids: tuple[str, ...],
        knot_ids: tuple[str, ...],
        orientation_delta: tuple[float, float, float, float],
        now_ts: float,
    ) -> None:
        formation = self.formation(relation_id)
        merged_threads = tuple(dict.fromkeys((*formation.active_thread_ids, *thread_ids)))
        merged_knots = tuple(dict.fromkeys((*formation.active_knot_ids, *knot_ids)))
        updated = replace(
            formation,
            active_thread_ids=merged_threads,
            active_knot_ids=merged_knots,
            boundary_tension=_clamp(
                formation.boundary_tension + 0.05 * len(knot_ids) - 0.03 * len(thread_ids),
                0.0,
                1.0,
            ),
            last_contact_at=now_ts,
        )
        self.formations[relation_id] = updated

        orientation = self.orientation(relation_id)
        self.orientations[relation_id] = replace(
            orientation,
            self_orientation=_clamp(orientation.self_orientation + orientation_delta[0], -1.0, 1.0),
            world_orientation=_clamp(
                orientation.world_orientation + orientation_delta[1], -1.0, 1.0
            ),
            relation_orientation=_clamp(
                orientation.relation_orientation + orientation_delta[2], -1.0, 1.0
            ),
            narrative_tilt=_clamp(orientation.narrative_tilt + orientation_delta[3], -1.0, 1.0),
            updated_at=now_ts,
        )

    def dominant_channels(self, relation_id: str) -> tuple[TraceChannel, ...]:
        scores: dict[TraceChannel, float] = {}
        for moment in self.moments.values():
            if moment.relation_id != relation_id:
                continue
            for channel in moment.user_channels:
                scores.setdefault(channel, 0.0)
                scores[channel] += 1.0
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return tuple(channel for channel, _ in ranked[:3])
