from __future__ import annotations

from aurora.being.metabolic_state import MetabolicState
from aurora.memory.store import MemoryStore
from aurora.phases.outcomes import PhaseOutcome
from aurora.phases.transitions import phase_transition
from aurora.runtime.contracts import Phase


def run_doze(metabolic: MetabolicState, memory_store: MemoryStore, now_ts: float) -> PhaseOutcome:
    previous = metabolic.phase
    metabolic.enter_phase(Phase.DOZE, now_ts)
    memory_store.decay_for_doze(now_ts)
    metabolic.bump_sleep_need(0.10)
    return PhaseOutcome(
        phase=Phase.DOZE,
        transition=phase_transition(previous, Phase.DOZE, "manual_doze", now_ts),
    )
