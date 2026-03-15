from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.store import MemoryStore
from aurora.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class SleepSnapshot:
    sleep_cycles: int
    thread_count: int
    knot_count: int
    anchor_thread_count: int
    active_knot_count: int
    pending_sleep_relations: int


@dataclass(frozen=True, slots=True)
class SleepEffectsCheck:
    sleep_cycles_delta: int
    thread_delta: int
    knot_delta: int
    anchor_thread_delta: int
    active_knot_delta: int
    pending_sleep_cleared: bool

    @property
    def ok(self) -> bool:
        return self.sleep_cycles_delta >= 1 and self.pending_sleep_cleared


def snapshot_sleep_state(state: RuntimeState, memory_store: MemoryStore) -> SleepSnapshot:
    return SleepSnapshot(
        sleep_cycles=memory_store.sleep_cycles,
        thread_count=len(memory_store.threads),
        knot_count=len(memory_store.knots),
        anchor_thread_count=len(state.orientation.anchor_thread_ids),
        active_knot_count=len(state.metabolic.active_knot_ids),
        pending_sleep_relations=len(state.metabolic.pending_sleep_relation_ids),
    )


def evaluate_sleep_effects(before: SleepSnapshot, after: SleepSnapshot) -> SleepEffectsCheck:
    return SleepEffectsCheck(
        sleep_cycles_delta=after.sleep_cycles - before.sleep_cycles,
        thread_delta=after.thread_count - before.thread_count,
        knot_delta=after.knot_count - before.knot_count,
        anchor_thread_delta=after.anchor_thread_count - before.anchor_thread_count,
        active_knot_delta=after.active_knot_count - before.active_knot_count,
        pending_sleep_cleared=before.pending_sleep_relations >= after.pending_sleep_relations
        and after.pending_sleep_relations == 0,
    )
