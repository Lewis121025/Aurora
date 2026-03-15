from __future__ import annotations

from typing import TypedDict

from aurora.relation.formation import RelationFormation


class RelationProjection(TypedDict):
    trust: float
    distance: float
    repairability: float
    boundary_events: int
    repair_events: int
    resonance_events: int
    thread_ids: tuple[str, ...]
    knot_ids: tuple[str, ...]


def project_relation(formation: RelationFormation) -> RelationProjection:
    total = max(1, formation.boundary_events + formation.repair_events + formation.resonance_events)
    trust = (formation.resonance_events + formation.repair_events) / (total + 1)
    distance = formation.boundary_events / (total + 1)
    repairability = formation.repair_events / max(
        1, formation.repair_events + formation.boundary_events
    )
    return {
        "trust": round(trust, 4),
        "distance": round(distance, 4),
        "repairability": round(repairability, 4),
        "boundary_events": formation.boundary_events,
        "repair_events": formation.repair_events,
        "resonance_events": formation.resonance_events,
        "thread_ids": tuple(sorted(formation.thread_ids)),
        "knot_ids": tuple(sorted(formation.knot_ids)),
    }
