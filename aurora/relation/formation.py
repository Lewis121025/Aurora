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
