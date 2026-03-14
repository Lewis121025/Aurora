from __future__ import annotations

from aurora.memory.store import MemoryStore
from aurora.phases.transitions import phase_transition
from aurora.relation.store import RelationStore
from aurora.runtime.models import MetabolicState, Phase, PhaseOutcome


def run_doze(
    metabolic: MetabolicState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> PhaseOutcome:
    relation_id = metabolic.current_relation_id
    if relation_id is not None:
        memory_store.doze_consolidate(relation_id=relation_id, now_ts=now_ts)
        formation = relation_store.formation(relation_id)
        thread_ids = formation.active_thread_ids
        knot_ids = formation.active_knot_ids
    else:
        thread_ids = ()
        knot_ids = ()

    transition = phase_transition(
        from_phase=metabolic.phase,
        to_phase=Phase.DOZE,
        reason="manual_doze",
        created_at=now_ts,
    )
    next_metabolic = MetabolicState(
        phase=Phase.DOZE,
        sleep_need=max(0.0, metabolic.sleep_need - 0.20),
        current_relation_id=relation_id,
        active_thread_ids=thread_ids,
        active_knot_ids=knot_ids,
        last_transition_at=now_ts,
    )
    return PhaseOutcome(metabolic=next_metabolic, transition=transition)
