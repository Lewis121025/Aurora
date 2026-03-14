from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from aurora.memory.store import MemoryStore
from aurora.persistence.store import SQLitePersistence
from aurora.phases.awake import run_awake
from aurora.phases.doze import run_doze
from aurora.phases.sleep import run_sleep
from aurora.relation.store import RelationStore
from aurora.runtime.bootstrap import initial_being
from aurora.runtime.clock import SystemClock
from aurora.runtime.models import Phase, RuntimeState


@dataclass(frozen=True, slots=True)
class EngineOutput:
    turn_id: str
    response_text: str
    aurora_move: str
    dominant_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseOutput:
    phase: Phase
    transition_id: str


class StateSummary(TypedDict):
    phase: str
    continuity_pressure: float
    sleep_pressure: float
    coherence_pressure: float
    softness: float
    boundary_tension: float
    active_relation_id: str | None
    recent_chapter_bias: tuple[str, ...]
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    memory_chapters: int
    relation_count: int
    transitions: int
    sleep_cycles: int
    last_reweave_delta: float


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
        memory_store, relation_store, state = persistence.load_runtime(initial=initial_being())
        return cls(memory_store, relation_store, state, persistence, clock)

    def handle_turn(self, session_id: str, text: str) -> EngineOutput:
        now_ts = self.clock.now()
        relation_id = f"rel:{session_id}"
        outcome = run_awake(
            relation_id=relation_id,
            session_id=session_id,
            text=text,
            being=self.state.being,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=now_ts,
        )
        if outcome.transition is not None:
            self.state.transitions.append(outcome.transition)
        self.persistence.persist_awake(
            outcome=outcome,
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
        )
        return EngineOutput(
            turn_id=outcome.user_turn.turn_id,
            response_text=outcome.response_text,
            aurora_move=outcome.aurora_move,
            dominant_channels=tuple(
                channel.value for channel in outcome.activation.dominant_channels
            ),
        )

    def doze(self) -> PhaseOutput:
        outcome = run_doze(
            being=self.state.being, memory_store=self.memory_store, now_ts=self.clock.now()
        )
        self.state.transitions.append(outcome.transition)
        self.persistence.persist_phase(
            outcome=outcome,
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
        )
        return PhaseOutput(
            phase=outcome.being.phase, transition_id=outcome.transition.transition_id
        )

    def sleep(self) -> PhaseOutput:
        outcome = run_sleep(
            being=self.state.being,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=self.clock.now(),
        )
        self.state.transitions.append(outcome.transition)
        self.persistence.persist_phase(
            outcome=outcome,
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
        )
        return PhaseOutput(
            phase=outcome.being.phase, transition_id=outcome.transition.transition_id
        )

    def health_summary(self) -> dict[str, str | int]:
        return {
            "status": "ok",
            "phase": self.state.being.phase.value,
            "turns": self.persistence.turn_count(),
            "transitions": self.persistence.phase_transition_count(),
        }

    def state_summary(self) -> StateSummary:
        being = self.state.being
        return {
            "phase": being.phase.value,
            "continuity_pressure": being.continuity_pressure,
            "sleep_pressure": being.sleep_pressure,
            "coherence_pressure": being.coherence_pressure,
            "softness": being.softness,
            "boundary_tension": being.boundary_tension,
            "active_relation_id": being.active_relation_id,
            "recent_chapter_bias": being.recent_chapter_bias,
            "turns": self.persistence.turn_count(),
            "memory_fragments": len(self.memory_store.fragments),
            "memory_traces": len(self.memory_store.traces),
            "memory_associations": len(self.memory_store.associations),
            "memory_chapters": len(self.memory_store.chapters),
            "relation_count": len(self.relation_store.states),
            "transitions": self.persistence.phase_transition_count(),
            "sleep_cycles": self.memory_store.sleep_cycles,
            "last_reweave_delta": self.memory_store.last_reweave_delta,
        }
