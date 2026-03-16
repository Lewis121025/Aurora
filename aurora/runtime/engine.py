from __future__ import annotations

import time
from dataclasses import dataclass

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.llm.config import load_llm_config
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.memory.store import MemoryStore
from aurora.persistence.store import SQLitePersistence
from aurora.phases.awake import run_awake
from aurora.phases.doze import run_doze
from aurora.phases.sleep import run_sleep
from aurora.relation.store import RelationStore
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


class AuroraEngine:
    __slots__ = ("memory_store", "relation_store", "state", "persistence", "llm")

    def __init__(
        self,
        memory_store: MemoryStore,
        relation_store: RelationStore,
        state: RuntimeState,
        persistence: SQLitePersistence,
        llm: LLMProvider,
    ) -> None:
        self.memory_store = memory_store
        self.relation_store = relation_store
        self.state = state
        self.persistence = persistence
        self.llm = llm

    @classmethod
    def create(cls, data_dir: str | None = None, llm: LLMProvider | None = None) -> "AuroraEngine":
        persistence = SQLitePersistence(data_dir=data_dir)
        initial = RuntimeState(orientation=Orientation(), metabolic=MetabolicState())
        memory_store, relation_store, state = persistence.load_runtime(initial=initial)
        if llm is None:
            llm_config = load_llm_config()
            if llm_config is None:
                raise RuntimeError(
                    "Aurora requires an LLM provider. "
                    "Set AURORA_LLM_BASE_URL, AURORA_LLM_API_KEY, and AURORA_LLM_MODEL."
                )
            llm = OpenAICompatProvider(llm_config)
        return cls(memory_store, relation_store, state, persistence, llm)

    def handle_turn(self, session_id: str, text: str) -> EngineOutput:
        now_ts = time.time()
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
            self.state.append_transition(outcome.transition)
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
            now_ts=time.time(),
        )
        self.state.append_transition(outcome.transition)
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
            now_ts=time.time(),
        )
        self.state.append_transition(outcome.transition)
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
