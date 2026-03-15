from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.store import MemoryStore
from aurora.relation.projectors import project_relation
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
    projection = project_relation(formation)
    return RelationDynamicsCheck(
        moment_count=len(relation_store.moments.get(relation_id, ())),
        boundary_events=projection["boundary_events"],
        repair_events=projection["repair_events"],
        linked_threads=len(projection["thread_ids"]),
        linked_knots=len(projection["knot_ids"]),
        relation_has_memory=bool(memory_store.fragments_for_relation(relation_id)),
    )
