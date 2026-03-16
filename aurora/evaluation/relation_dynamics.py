from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.store import MemoryStore
from aurora.relation.store import RelationStore


@dataclass(frozen=True, slots=True)
class RelationDynamicsCheck:
    moment_count: int
    boundary_events: int
    repair_events: int
    linked_threads: int
    linked_knots: int
    relation_has_memory: bool

    @property
    def ok(self) -> bool:
        return self.moment_count >= 1 and self.relation_has_memory


def evaluate_relation_dynamics(
    relation_store: RelationStore,
    memory_store: MemoryStore,
    relation_id: str,
) -> RelationDynamicsCheck:
    formation = relation_store.formation_for(relation_id)
    return RelationDynamicsCheck(
        moment_count=len(relation_store.moments.get(relation_id, ())),
        boundary_events=formation.boundary_events,
        repair_events=formation.repair_events,
        linked_threads=len(formation.thread_ids),
        linked_knots=len(formation.knot_ids),
        relation_has_memory=bool(memory_store.fragments_for_relation(relation_id)),
    )
