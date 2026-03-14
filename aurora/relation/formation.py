from __future__ import annotations

from dataclasses import dataclass, field

from aurora.relation.moment import RelationMoment


@dataclass(slots=True)
class RelationFormation:
    relation_id: str
    thread_ids: set[str] = field(default_factory=set)
    knot_ids: set[str] = field(default_factory=set)
    boundary_events: int = 0
    repair_events: int = 0
    resonance_events: int = 0
    last_contact_at: float = 0.0

    def register_moment(self, moment: RelationMoment) -> None:
        if moment.boundary_event:
            self.boundary_events += 1
        if moment.repair_event:
            self.repair_events += 1
        if moment.aurora_move in {"approach", "repair", "witness"}:
            self.resonance_events += 1
        self.last_contact_at = max(self.last_contact_at, moment.created_at)

    def absorb_sleep(
        self,
        thread_ids: tuple[str, ...],
        knot_ids: tuple[str, ...],
        now_ts: float,
    ) -> None:
        self.thread_ids.update(thread_ids)
        self.knot_ids.update(knot_ids)
        self.last_contact_at = max(self.last_contact_at, now_ts)

    def snapshot(self) -> dict[str, float | tuple[str, ...] | int]:
        total = max(1, self.boundary_events + self.repair_events + self.resonance_events)
        trust = (self.resonance_events + self.repair_events) / (total + 1)
        distance = self.boundary_events / (total + 1)
        repairability = self.repair_events / max(1, self.repair_events + self.boundary_events)
        return {
            "trust": round(trust, 4),
            "distance": round(distance, 4),
            "repairability": round(repairability, 4),
            "boundary_events": self.boundary_events,
            "repair_events": self.repair_events,
            "resonance_events": self.resonance_events,
            "thread_ids": tuple(sorted(self.thread_ids)),
            "knot_ids": tuple(sorted(self.knot_ids)),
        }
