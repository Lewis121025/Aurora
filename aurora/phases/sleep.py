from __future__ import annotations

from aurora.memory.store import MemoryStore
from aurora.phases.transitions import phase_transition
from aurora.relation.store import RelationStore
from aurora.runtime.models import BeingState, Phase, PhaseOutcome


def run_sleep(
    being: BeingState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> PhaseOutcome:
    previous = being.phase
    being.phase = Phase.SLEEP
    mutation = memory_store.reweave(relation_states=relation_store.states, now_ts=now_ts)
    being.drift(mutation.self_drift, mutation.world_drift)
    being.coherence_pressure = max(0.0, being.coherence_pressure - mutation.coherence_shift)
    being.boundary_tension = min(
        1.0,
        max(0.0, being.boundary_tension + mutation.tension_shift - 0.05),
    )
    being.sleep_pressure = max(0.0, being.sleep_pressure - 0.45)
    being.recent_chapter_bias = mutation.chapter_ids[-4:]
    return PhaseOutcome(
        being=being,
        transition=phase_transition(previous, Phase.SLEEP, "manual_sleep", now_ts),
        mutation=mutation,
    )
