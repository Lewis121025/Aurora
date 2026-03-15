from __future__ import annotations

from dataclasses import dataclass, field

from aurora.llm.config import load_llm_config
from aurora.llm.openai_compat import create_provider
from aurora.llm.provider import LLMProvider
from aurora.memory.store import MemoryStore
from aurora.persistence.store import SQLitePersistence
from aurora.phases.awake import run_awake
from aurora.phases.doze import run_doze
from aurora.phases.sleep import run_sleep
from aurora.relation.store import RelationStore
from aurora.runtime.bootstrap import initial_runtime_state
from aurora.runtime.clock import SystemClock
from aurora.runtime.contracts import AuroraMove, Phase
from aurora.runtime.projections import (
    HealthSummary,
    StateSummary,
    project_health_summary,
    project_state_summary,
)
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


@dataclass(slots=True)
class AuroraEngine:
    memory_store: MemoryStore
    relation_store: RelationStore
    state: RuntimeState
    persistence: SQLitePersistence
    clock: SystemClock = field(default_factory=SystemClock)
    llm: LLMProvider | None = None

    @classmethod
    def create(cls, data_dir: str | None = None) -> "AuroraEngine":
        clock = SystemClock()
        persistence = SQLitePersistence(data_dir=data_dir)
        memory_store, relation_store, state = persistence.load_runtime(
            initial=initial_runtime_state()
        )
        llm_config = load_llm_config()
        llm: LLMProvider | None = create_provider(llm_config) if llm_config else None
        return cls(memory_store, relation_store, state, persistence, clock, llm)

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
            llm=self.llm,
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

    def health_summary(self) -> HealthSummary:
        return project_health_summary(
            state=self.state,
            turns=self.persistence.turn_count(),
            transitions=self.persistence.phase_transition_count(),
        )

    def state_summary(self) -> StateSummary:
        return project_state_summary(
            state=self.state,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            turns=self.persistence.turn_count(),
            transitions=self.persistence.phase_transition_count(),
        )
