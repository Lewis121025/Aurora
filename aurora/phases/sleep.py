from __future__ import annotations

from aurora.memory.store import MemoryStore
from aurora.phases.transitions import phase_transition
from aurora.relation.store import RelationStore
from aurora.runtime.models import MetabolicState, Phase, PhaseOutcome, ReweaveMutation


def run_sleep(
    metabolic: MetabolicState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> PhaseOutcome:
    relation_id = metabolic.current_relation_id
    mutation: ReweaveMutation | None = None
    thread_ids: tuple[str, ...] = ()
    knot_ids: tuple[str, ...] = ()

    if relation_id is not None:
        softened_ids, edge_ids, created_thread_ids, created_knot_ids, delta = memory_store.reweave(
            relation_id=relation_id,
            now_ts=now_ts,
        )
        orientation_delta = (
            -0.01 * float(len(created_knot_ids)),
            0.02 * float(len(created_thread_ids)),
            0.03 * float(len(created_thread_ids)) - 0.02 * float(len(created_knot_ids)),
            0.02 * delta,
        )
        relation_store.absorb_reweave(
            relation_id=relation_id,
            thread_ids=created_thread_ids,
            knot_ids=created_knot_ids,
            orientation_delta=orientation_delta,
            now_ts=now_ts,
        )
        formation = relation_store.formation(relation_id)
        thread_ids = formation.active_thread_ids
        knot_ids = formation.active_knot_ids

        mutation = ReweaveMutation(
            mutation_id=f"mut_{int(now_ts * 1000)}",
            relation_id=relation_id,
            softened_fragment_ids=softened_ids,
            strengthened_edge_ids=edge_ids,
            created_thread_ids=created_thread_ids,
            updated_thread_ids=(),
            created_knot_ids=created_knot_ids,
            updated_knot_ids=(),
            orientation_delta=orientation_delta,
            created_at=now_ts,
        )

    transition = phase_transition(
        from_phase=metabolic.phase,
        to_phase=Phase.SLEEP,
        reason="manual_sleep",
        created_at=now_ts,
    )
    next_metabolic = MetabolicState(
        phase=Phase.SLEEP,
        sleep_need=max(0.0, metabolic.sleep_need - 0.55),
        current_relation_id=relation_id,
        active_thread_ids=thread_ids,
        active_knot_ids=knot_ids,
        last_transition_at=now_ts,
    )
    return PhaseOutcome(metabolic=next_metabolic, transition=transition, mutation=mutation)
