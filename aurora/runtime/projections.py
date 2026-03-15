from __future__ import annotations

from typing import TypedDict

from aurora.memory.store import MemoryStore
from aurora.relation.store import RelationStore
from aurora.runtime.state import RuntimeState


class HealthSummary(TypedDict):
    status: str
    phase: str
    turns: int
    transitions: int


class StateSummary(TypedDict):
    phase: str
    sleep_need: float
    active_relation_ids: tuple[str, ...]
    pending_sleep_relation_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    anchor_thread_ids: tuple[str, ...]
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    memory_threads: int
    memory_knots: int
    relation_formations: int
    relation_moments: int
    sleep_cycles: int
    transitions: int


def project_health_summary(state: RuntimeState, turns: int, transitions: int) -> HealthSummary:
    return {
        "status": "ok",
        "phase": state.metabolic.phase.value,
        "turns": turns,
        "transitions": transitions,
    }


def project_state_summary(
    state: RuntimeState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    turns: int,
    transitions: int,
) -> StateSummary:
    metabolic = state.metabolic
    orientation = state.orientation
    return {
        "phase": metabolic.phase.value,
        "sleep_need": round(metabolic.sleep_need, 4),
        "active_relation_ids": metabolic.active_relation_ids,
        "pending_sleep_relation_ids": metabolic.pending_sleep_relation_ids,
        "active_knot_ids": metabolic.active_knot_ids,
        "anchor_thread_ids": orientation.anchor_thread_ids,
        "turns": turns,
        "memory_fragments": len(memory_store.fragments),
        "memory_traces": len(memory_store.traces),
        "memory_associations": len(memory_store.associations),
        "memory_threads": len(memory_store.threads),
        "memory_knots": len(memory_store.knots),
        "relation_formations": len(relation_store.formations),
        "relation_moments": relation_store.moment_count(),
        "sleep_cycles": memory_store.sleep_cycles,
        "transitions": transitions,
    }
