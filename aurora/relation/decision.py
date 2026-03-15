from __future__ import annotations

from dataclasses import dataclass

from aurora.relation.formation import RelationFormation


@dataclass(frozen=True, slots=True)
class RelationDecisionContext:
    boundary_events: int
    repair_events: int
    resonance_events: int
    thread_count: int
    knot_count: int


def build_relation_decision_context(formation: RelationFormation) -> RelationDecisionContext:
    return RelationDecisionContext(
        boundary_events=formation.boundary_events,
        repair_events=formation.repair_events,
        resonance_events=formation.resonance_events,
        thread_count=len(formation.thread_ids),
        knot_count=len(formation.knot_ids),
    )
