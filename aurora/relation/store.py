from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from aurora.relation.formation import RelationFormation
from aurora.relation.moment import RelationMoment
from aurora.runtime.contracts import AuroraMove, TraceChannel


MOMENT_CAP = 128


class RelationStore:
    def __init__(self) -> None:
        self.formations: dict[str, RelationFormation] = {}
        self.moments: dict[str, list[RelationMoment]] = defaultdict(list)
        self._dirty_formations: set[str] = set()
        self._dirty_moment_relations: set[str] = set()

    def formation_for(self, relation_id: str) -> RelationFormation:
        if relation_id not in self.formations:
            self.formations[relation_id] = RelationFormation(relation_id=relation_id)
            self._dirty_formations.add(relation_id)
        return self.formations[relation_id]

    def record_exchange(
        self,
        relation_id: str,
        user_turn_id: str,
        aurora_turn_id: str | None,
        user_channels: tuple[TraceChannel, ...],
        aurora_move: AuroraMove,
        summary: str,
        now_ts: float,
    ) -> RelationMoment:
        normalized_channels = tuple(sorted(set(user_channels), key=lambda c: c.value))
        boundary_event = TraceChannel.BOUNDARY in normalized_channels or aurora_move == "boundary"
        repair_event = TraceChannel.REPAIR in normalized_channels or aurora_move == "repair"
        moment = RelationMoment(
            moment_id=f"moment_{uuid4().hex[:12]}",
            relation_id=relation_id,
            user_turn_id=user_turn_id,
            aurora_turn_id=aurora_turn_id,
            user_channels=normalized_channels,
            aurora_move=aurora_move,
            boundary_event=boundary_event,
            repair_event=repair_event,
            summary=summary,
            created_at=now_ts,
        )
        bucket = self.moments[relation_id]
        bucket.append(moment)
        if len(bucket) > MOMENT_CAP:
            del bucket[: len(bucket) - MOMENT_CAP]
        self.formation_for(relation_id).register_moment(moment)
        self._dirty_formations.add(relation_id)
        self._dirty_moment_relations.add(relation_id)
        return moment

    def absorb_sleep(
        self,
        relation_id: str,
        thread_ids: tuple[str, ...],
        knot_ids: tuple[str, ...],
        now_ts: float,
    ) -> None:
        self.formation_for(relation_id).absorb_sleep(
            thread_ids=thread_ids,
            knot_ids=knot_ids,
            now_ts=now_ts,
        )
        self._dirty_formations.add(relation_id)

    def relation_count(self) -> int:
        return len(self.formations)

    def moment_count(self) -> int:
        return sum(len(items) for items in self.moments.values())

    def clear_dirty(self) -> None:
        self._dirty_formations.clear()
        self._dirty_moment_relations.clear()
