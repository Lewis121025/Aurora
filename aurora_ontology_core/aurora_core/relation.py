from __future__ import annotations

from dataclasses import dataclass, field

from .events import TouchSignal, make_id
from .schema import AuroraMoveKind, OtherMoveKind


@dataclass(frozen=True, slots=True)
class RelationMoment:
    moment_id: str
    relation_id: str
    user_turn_id: str
    aurora_turn_id: str | None
    other_move: OtherMoveKind
    aurora_move: AuroraMoveKind
    effect_signature: TouchSignal
    boundary_event: bool
    note: str
    created_at: float


@dataclass(slots=True)
class RelationState:
    relation_id: str
    trust: float = 0.10
    familiarity: float = 0.00
    boundary_tension: float = 0.10
    repairability: float = 0.50
    shared_chapter_ids: set[str] = field(default_factory=set)
    moment_ids: list[str] = field(default_factory=list)

    def clamp(self) -> None:
        self.trust = min(max(self.trust, 0.0), 1.0)
        self.familiarity = min(max(self.familiarity, 0.0), 1.0)
        self.boundary_tension = min(max(self.boundary_tension, 0.0), 1.0)
        self.repairability = min(max(self.repairability, 0.0), 1.0)


class RelationSystem:
    def __init__(self) -> None:
        self.states: dict[str, RelationState] = {}
        self.moments: dict[str, RelationMoment] = {}

    def get(self, relation_id: str) -> RelationState:
        if relation_id not in self.states:
            self.states[relation_id] = RelationState(relation_id=relation_id)
        return self.states[relation_id]

    def record_exchange(
        self,
        *,
        relation_id: str,
        user_turn_id: str,
        aurora_turn_id: str | None,
        other_move: OtherMoveKind,
        aurora_move: AuroraMoveKind,
        effect_signature: TouchSignal,
        boundary_event: bool,
        created_at: float,
        note: str = "",
    ) -> RelationMoment:
        state = self.get(relation_id)
        moment = RelationMoment(
            moment_id=make_id("moment"),
            relation_id=relation_id,
            user_turn_id=user_turn_id,
            aurora_turn_id=aurora_turn_id,
            other_move=other_move,
            aurora_move=aurora_move,
            effect_signature=effect_signature,
            boundary_event=boundary_event,
            note=note,
            created_at=created_at,
        )
        self.moments[moment.moment_id] = moment
        state.moment_ids.append(moment.moment_id)

        state.familiarity += 0.08
        state.trust += 0.10 * effect_signature.weights.get("warmth", 0.0)
        state.trust += 0.12 * effect_signature.weights.get("recognition", 0.0)
        state.trust += 0.10 * effect_signature.weights.get("repair", 0.0)
        state.trust -= 0.14 * effect_signature.weights.get("hurt", 0.0)
        state.boundary_tension += 0.20 * effect_signature.weights.get("boundary", 0.0)
        state.boundary_tension += 0.12 if other_move == "pressure" else 0.0
        state.boundary_tension -= 0.06 if aurora_move == "repair" else 0.0
        state.boundary_tension += 0.08 if aurora_move == "boundary" else 0.0
        state.repairability += 0.10 if other_move == "repair" else 0.0
        state.repairability -= 0.08 if other_move == "rupture" else 0.0
        state.clamp()
        return moment

    def absorb_reweave(
        self,
        relation_id: str,
        *,
        chapter_ids: tuple[str, ...],
        relation_bias: float,
        tension_shift: float,
    ) -> RelationState:
        state = self.get(relation_id)
        state.shared_chapter_ids.update(chapter_ids)
        state.trust += relation_bias * 0.10
        state.boundary_tension += tension_shift * 0.20
        state.clamp()
        return state
