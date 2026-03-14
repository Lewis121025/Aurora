from __future__ import annotations

from aurora.memory.store import MemoryStore
from aurora.phases.transitions import phase_transition
from aurora.runtime.models import BeingState, Phase, PhaseOutcome


def run_doze(being: BeingState, memory_store: MemoryStore, now_ts: float) -> PhaseOutcome:
    previous = being.phase
    being.phase = Phase.DOZE
    memory_store.decay_for_doze(now_ts)
    being.sleep_pressure = min(1.0, being.sleep_pressure + 0.10)
    being.coherence_pressure = min(1.0, being.coherence_pressure + 0.08)
    return PhaseOutcome(
        being=being,
        transition=phase_transition(previous, Phase.DOZE, "manual_doze", now_ts),
    )
