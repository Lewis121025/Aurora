from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from aurora.memory.store import MemoryStore
from aurora.persistence.store import SQLitePersistence
from aurora.phases.awake import run_awake
from aurora.phases.doze import run_doze
from aurora.phases.sleep import run_sleep
from aurora.phases.transitions import phase_transition
from aurora.phases.phase_types import Phase
from aurora.relation.store import RelationStore
from aurora.runtime.bootstrap import initial_snapshot
from aurora.runtime.clock import SystemClock
from aurora.runtime.models import AwakeOutcome, InteractionTurn, PhaseOutcome, RuntimeState
from aurora.runtime.policies import non_malice_floor


@dataclass(frozen=True, slots=True)
class EngineOutput:
    turn_id: str
    response_text: str
    touch_modes: tuple[str, ...]


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

    @classmethod
    def create(cls, data_dir: str | None = None) -> "AuroraEngine":
        clock = SystemClock()
        persistence = SQLitePersistence(data_dir=data_dir)
        memory_store, relation_store, state = persistence.load_runtime(
            initial=initial_snapshot(now_ts=clock.now())
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
        turn = InteractionTurn(
            turn_id=f"turn_{uuid4().hex[:10]}",
            session_id=session_id,
            speaker="user",
            text=text,
            created_at=now_ts,
        )

        outcome = self._run_awake(turn=turn, now_ts=now_ts)
        response_text = outcome.response_text
        if not non_malice_floor(response_text):
            response_text = "I cannot continue in that direction."

        return EngineOutput(
            turn_id=turn.turn_id,
            response_text=response_text,
            touch_modes=outcome.touch_modes,
        )

    def _run_awake(self, turn: InteractionTurn, now_ts: float) -> AwakeOutcome:
        outcome = run_awake(
            turn=turn,
            snapshot=self.state.snapshot,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=now_ts,
        )
        self.state.snapshot = outcome.snapshot
        if outcome.transition is not None:
            self.state.transitions.append(outcome.transition)
        self.persistence.persist_awake(turn=turn, outcome=outcome, memory_store=self.memory_store)
        return outcome

    def doze(self, reason: str = "manual_doze") -> PhaseOutput:
        now_ts = self.clock.now()
        from_phase = self.state.snapshot.phase
        snapshot = run_doze(
            snapshot=self.state.snapshot,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=now_ts,
        )
        transition = phase_transition(
            from_phase=from_phase,
            to_phase=Phase.DOZE,
            reason=reason,
            created_at=now_ts,
        )
        outcome = PhaseOutcome(snapshot=snapshot, transition=transition)
        self._apply_phase_outcome(outcome)
        return PhaseOutput(phase=outcome.snapshot.phase, transition_id=transition.transition_id)

    def sleep(self, reason: str = "manual_sleep") -> PhaseOutput:
        now_ts = self.clock.now()
        from_phase = self.state.snapshot.phase
        snapshot = run_sleep(
            snapshot=self.state.snapshot,
            memory_store=self.memory_store,
            relation_store=self.relation_store,
            now_ts=now_ts,
        )
        transition = phase_transition(
            from_phase=from_phase,
            to_phase=Phase.SLEEP,
            reason=reason,
            created_at=now_ts,
        )
        outcome = PhaseOutcome(snapshot=snapshot, transition=transition)
        self._apply_phase_outcome(outcome)
        return PhaseOutput(phase=outcome.snapshot.phase, transition_id=transition.transition_id)

    def _apply_phase_outcome(self, outcome: PhaseOutcome) -> None:
        self.state.snapshot = outcome.snapshot
        self.state.transitions.append(outcome.transition)
        self.persistence.persist_phase(outcome=outcome, memory_store=self.memory_store)

    def health_summary(self) -> dict[str, str | int]:
        return {
            "status": "ok",
            "phase": self.state.snapshot.phase.value,
            "turns": self.persistence.turn_count(),
            "transitions": self.persistence.phase_transition_count(),
        }

    def state_summary(self) -> dict[str, str | float | int]:
        snapshot = self.state.snapshot
        relation_tone = self.relation_store.current_tone()
        relation_strength = self.relation_store.tone_strength(relation_tone)
        avg_salience = 0.0
        avg_narrative_weight = 0.0
        narrative_pressure = self.memory_store.narrative_pressure()
        if self.memory_store.fragment_salience:
            avg_salience = sum(self.memory_store.fragment_salience.values()) / len(
                self.memory_store.fragment_salience
            )
        if self.memory_store.fragment_narrative_weight:
            avg_narrative_weight = sum(self.memory_store.fragment_narrative_weight.values()) / len(
                self.memory_store.fragment_narrative_weight
            )
        return {
            "phase": snapshot.phase.value,
            "updated_at": snapshot.updated_at,
            "self_view": snapshot.self_view,
            "world_view": snapshot.world_view,
            "openness": snapshot.openness,
            "turns": self.persistence.turn_count(),
            "memory_fragments": len(self.memory_store.fragments),
            "memory_traces": len(self.memory_store.traces),
            "memory_associations": len(self.memory_store.associations),
            "avg_salience": avg_salience,
            "avg_narrative_weight": avg_narrative_weight,
            "narrative_pressure": narrative_pressure,
            "sleep_cycles": self.memory_store.sleep_cycles,
            "last_reweave_delta": self.memory_store.last_reweave_delta,
            "relation_moments": len(self.relation_store.moments),
            "relation_tone": relation_tone,
            "relation_strength": relation_strength,
            "transitions": self.persistence.phase_transition_count(),
        }
