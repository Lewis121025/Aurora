from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from aurora.memory.store import MemoryStore
from aurora.persistence.store import SQLitePersistence
from aurora.phases.awake import run_awake
from aurora.phases.doze import run_doze
from aurora.phases.sleep import run_sleep
from aurora.relation.store import RelationStore
from aurora.runtime.bootstrap import initial_runtime_state
from aurora.runtime.clock import SystemClock
from aurora.runtime.contracts import AuroraMove, Phase
from aurora.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class EngineOutput:
    turn_id: str
    response_text: str
    aurora_move: AuroraMove
    dominant_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseOutput:
    phase: Phase
    transition_id: str


class StateSummary(TypedDict):
    phase: str
    sleep_need: float
    active_relation_ids: tuple[str, ...]
    pending_sleep_relation_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    anchor_thread_ids: tuple[str, ...]
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    memory_threads: int
    memory_knots: int
    relation_formations: int
    relation_moments: int
    sleep_cycles: int
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
            initial=initial_runtime_state()
        )
        return cls(memory_store, relation_store, state, persistence, clock)

    def handle_turn(self, session_id: str, text: str) -> EngineOutput:
        now_ts = self.clock.now()
        relation_id = f"rel:{session_id}"
        outcome = run_awake(
            relation_id=relation_id,
            session_id=session_id,
            text=text,
            orientation=self.state.orientation,
            metabolic=self.state.metabolic,
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
            dominant_channels=tuple(channel.value for channel in outcome.dominant_channels),
        )

    def doze(self) -> PhaseOutput:
        outcome = run_doze(
            metabolic=self.state.metabolic,
            memory_store=self.memory_store,
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
            phase=outcome.phase,
            transition_id=outcome.transition.transition_id,
        )

    def sleep(self) -> PhaseOutput:
        outcome = run_sleep(
            metabolic=self.state.metabolic,
            orientation=self.state.orientation,
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
            phase=outcome.phase,
            transition_id=outcome.transition.transition_id,
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
        orientation = self.state.orientation
        return {
            "phase": metabolic.phase.value,
            "sleep_need": round(metabolic.sleep_need, 4),
            "active_relation_ids": metabolic.active_relation_ids,
            "pending_sleep_relation_ids": metabolic.pending_sleep_relation_ids,
            "active_knot_ids": metabolic.active_knot_ids,
            "anchor_thread_ids": orientation.anchor_thread_ids,
            "turns": self.persistence.turn_count(),
            "memory_fragments": len(self.memory_store.fragments),
            "memory_traces": len(self.memory_store.traces),
            "memory_associations": len(self.memory_store.associations),
            "memory_threads": len(self.memory_store.threads),
            "memory_knots": len(self.memory_store.knots),
            "relation_formations": len(self.relation_store.formations),
            "relation_moments": self.relation_store.moment_count(),
            "sleep_cycles": self.memory_store.sleep_cycles,
            "transitions": self.persistence.phase_transition_count(),
        }
