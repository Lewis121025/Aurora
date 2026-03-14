from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
from uuid import uuid4

from aurora.memory.store import MemoryStore
from aurora.persistence.store import SQLitePersistence
from aurora.phases.awake import run_awake
from aurora.phases.doze import run_doze
from aurora.phases.sleep import run_sleep
from aurora.relation.store import RelationStore
from aurora.runtime.bootstrap import initial_metabolic
from aurora.runtime.clock import SystemClock
from aurora.runtime.models import (
    MetabolicState,
    Phase,
    PhaseOutcome,
    PhaseTransition,
    RuntimeState,
    Speaker,
    Turn,
)
from aurora.runtime.policies import non_malice_floor


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True, slots=True)
class EngineOutput:
    turn_id: str
    response_text: str
    touch_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseOutput:
    phase: Phase
    transition_id: str


class StateSummary(TypedDict):
    phase: str
    sleep_need: float
    current_relation_id: str | None
    active_thread_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    last_transition_at: float
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    threads: int
    knots: int
    relation_moments: int
    trust: float
    boundary_tension: float
    sleep_cycles: int
    last_reweave_delta: float
    transitions: int


@dataclass(slots=True)
class AuroraEngine:
    memory_store: MemoryStore
    relation_store: RelationStore
    state: RuntimeState
    persistence: SQLitePersistence
    clock: SystemClock = field(default_factory=SystemClock)

    @classmethod
    def create(cls, data_dir: str | None = None) -> "AuroraEngine":
        clock = SystemClock()
        persistence = SQLitePersistence(data_dir=data_dir)
        memory_store, relation_store, state = persistence.load_runtime(
            initial=initial_metabolic(now_ts=clock.now())
        )
        return cls(
            memory_store=memory_store,
            relation_store=relation_store,
            state=state,
            persistence=persistence,
            clock=clock,
        )

    def handle_turn(self, session_id: str, text: str) -> EngineOutput:
        now_ts = self.clock.now()
        relation_id = f"rel:{session_id}"
        user_turn = Turn(
            turn_id=f"turn_{uuid4().hex[:10]}",
            relation_id=relation_id,
            session_id=session_id,
            speaker=Speaker.USER,
            text=text,
            created_at=now_ts,
        )

        outcome = run_awake(
            user_turn=user_turn,
            metabolic=self.state.metabolic,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=now_ts,
        )
        response_text = outcome.response_text
        if not non_malice_floor(response_text):
            response_text = "I cannot continue in that direction."

        self._apply_awake_effects(
            relation_id=relation_id, now_ts=now_ts, transition=outcome.transition
        )
        self.persistence.persist_awake(
            outcome=outcome,
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
        )
        return EngineOutput(
            turn_id=outcome.user_turn.turn_id,
            response_text=response_text,
            touch_channels=tuple(channel.value for channel in outcome.touch_channels),
        )

    def doze(self) -> PhaseOutput:
        now_ts = self.clock.now()
        outcome = run_doze(
            metabolic=self.state.metabolic,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=now_ts,
        )
        self._apply_phase_outcome(outcome)
        return PhaseOutput(
            phase=outcome.metabolic.phase, transition_id=outcome.transition.transition_id
        )

    def sleep(self) -> PhaseOutput:
        now_ts = self.clock.now()
        outcome = run_sleep(
            metabolic=self.state.metabolic,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=now_ts,
        )
        self._apply_phase_outcome(outcome)
        return PhaseOutput(
            phase=outcome.metabolic.phase, transition_id=outcome.transition.transition_id
        )

    def _apply_awake_effects(
        self, relation_id: str, now_ts: float, transition: PhaseTransition | None
    ) -> None:
        formation = self.relation_store.formation(relation_id)
        self.state.metabolic = MetabolicState(
            phase=Phase.AWAKE,
            sleep_need=_clamp(self.state.metabolic.sleep_need + 0.12, 0.0, 1.0),
            current_relation_id=relation_id,
            active_thread_ids=formation.active_thread_ids,
            active_knot_ids=formation.active_knot_ids,
            last_transition_at=now_ts,
        )
        if transition is not None:
            self.state.transitions.append(transition)

    def _apply_phase_outcome(self, outcome: PhaseOutcome) -> None:
        self.state.metabolic = outcome.metabolic
        self.state.transitions.append(outcome.transition)
        self.persistence.persist_phase(
            outcome=outcome,
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
        )

    def health_summary(self) -> dict[str, str | int]:
        return {
            "status": "ok",
            "phase": self.state.metabolic.phase.value,
            "turns": self.persistence.turn_count(),
            "transitions": self.persistence.phase_transition_count(),
        }

    def state_summary(self) -> StateSummary:
        metabolic = self.state.metabolic
        relation_id = metabolic.current_relation_id
        trust = 0.0
        boundary_tension = 0.0
        if relation_id is not None:
            formation = self.relation_store.formation(relation_id)
            trust = formation.trust
            boundary_tension = formation.boundary_tension

        return {
            "phase": metabolic.phase.value,
            "sleep_need": metabolic.sleep_need,
            "current_relation_id": metabolic.current_relation_id,
            "active_thread_ids": metabolic.active_thread_ids,
            "active_knot_ids": metabolic.active_knot_ids,
            "last_transition_at": metabolic.last_transition_at,
            "turns": self.persistence.turn_count(),
            "memory_fragments": len(self.memory_store.fragments),
            "memory_traces": len(self.memory_store.traces),
            "memory_associations": len(self.memory_store.associations),
            "threads": len(self.memory_store.threads),
            "knots": len(self.memory_store.knots),
            "relation_moments": len(self.relation_store.moments),
            "trust": trust,
            "boundary_tension": boundary_tension,
            "sleep_cycles": self.memory_store.sleep_cycles,
            "last_reweave_delta": self.memory_store.last_reweave_delta,
            "transitions": self.persistence.phase_transition_count(),
        }
