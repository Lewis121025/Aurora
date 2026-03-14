from __future__ import annotations

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.memory.store import MemoryStore
from aurora.phases.outcomes import PhaseOutcome
from aurora.phases.transitions import phase_transition
from aurora.relation.store import RelationStore
from aurora.runtime.contracts import Phase, TraceChannel


def run_sleep(
    metabolic: MetabolicState,
    orientation: Orientation,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> PhaseOutcome:
    previous = metabolic.phase
    metabolic.enter_phase(Phase.SLEEP, now_ts)

    mutation = memory_store.reweave(
        relation_formations=relation_store.formations,
        now_ts=now_ts,
        pending_relations=metabolic.pending_sleep_relation_ids or None,
    )

    dominant: set[TraceChannel] = set()
    for thread_id in mutation.created_thread_ids:
        thread = memory_store.threads[thread_id]
        dominant.update(thread.dominant_channels)
    for knot_id in mutation.created_knot_ids:
        knot = memory_store.knots[knot_id]
        dominant.update(knot.dominant_channels)

    orientation.absorb_sleep(
        thread_ids=mutation.created_thread_ids,
        knot_ids=mutation.created_knot_ids,
        dominant_channels=tuple(sorted(dominant, key=lambda item: item.value)),
        now_ts=now_ts,
    )
    metabolic.set_active_knots(mutation.created_knot_ids)
    metabolic.settle_after_sleep()

    return PhaseOutcome(
        phase=Phase.SLEEP,
        transition=phase_transition(previous, Phase.SLEEP, "manual_sleep", now_ts),
        mutation=mutation,
    )
