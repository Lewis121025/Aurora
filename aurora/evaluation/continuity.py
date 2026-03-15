from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.store import MemoryStore
from aurora.relation.store import RelationStore
from aurora.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class ContinuityCheck:
    active_relations_known: bool
    active_knots_known: bool
    anchor_threads_known: bool
    transitions_monotonic: bool

    @property
    def ok(self) -> bool:
        return (
            self.active_relations_known
            and self.active_knots_known
            and self.anchor_threads_known
            and self.transitions_monotonic
        )


def evaluate_continuity(
    state: RuntimeState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
) -> ContinuityCheck:
    known_relations = set(relation_store.formations) | set(memory_store.relation_fragments)
    active_relations_known = all(
        relation_id in known_relations for relation_id in state.metabolic.active_relation_ids
    )
    active_knots_known = all(
        knot_id in memory_store.knots for knot_id in state.metabolic.active_knot_ids
    )
    anchor_threads_known = all(
        thread_id in memory_store.threads for thread_id in state.orientation.anchor_thread_ids
    )
    transition_times = [item.created_at for item in state.transitions]
    transitions_monotonic = transition_times == sorted(transition_times)
    return ContinuityCheck(
        active_relations_known=active_relations_known,
        active_knots_known=active_knots_known,
        anchor_threads_known=anchor_threads_known,
        transitions_monotonic=transitions_monotonic,
    )
