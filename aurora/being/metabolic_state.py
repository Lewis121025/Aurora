from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import Phase, clamp


@dataclass(slots=True)
class MetabolicState:
    phase: Phase = Phase.AWAKE
    sleep_need: float = 0.0
    active_relation_ids: tuple[str, ...] = ()
    active_knot_ids: tuple[str, ...] = ()
    pending_sleep_relation_ids: tuple[str, ...] = ()
    last_transition_at: float = 0.0

    def enter_phase(self, next_phase: Phase, now_ts: float) -> None:
        if self.phase != next_phase:
            self.phase = next_phase
            self.last_transition_at = now_ts

    def queue_relation_for_sleep(self, relation_id: str) -> None:
        if relation_id not in self.pending_sleep_relation_ids:
            self.pending_sleep_relation_ids = self.pending_sleep_relation_ids + (relation_id,)

    def set_active_relation(self, relation_id: str) -> None:
        if relation_id in self.active_relation_ids:
            self.active_relation_ids = tuple(
                [item for item in self.active_relation_ids if item != relation_id] + [relation_id]
            )
            return
        self.active_relation_ids = tuple([*self.active_relation_ids, relation_id][-4:])

    def set_active_knots(self, knot_ids: tuple[str, ...]) -> None:
        self.active_knot_ids = tuple(knot_ids[-8:])

    def settle_after_sleep(self) -> None:
        self.pending_sleep_relation_ids = ()
        self.sleep_need = clamp(self.sleep_need - 0.45)

    def bump_sleep_need(self, amount: float) -> None:
        self.sleep_need = clamp(self.sleep_need + amount)
