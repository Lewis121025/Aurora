from __future__ import annotations

from aurora.being.drift import apply_doze_drift
from aurora.memory.store import MemoryStore
from aurora.phases.phase_types import Phase
from aurora.relation.store import RelationStore
from aurora.runtime.models import ExistentialSnapshot


def run_doze(
    snapshot: ExistentialSnapshot,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> ExistentialSnapshot:
    drifted = apply_doze_drift(
        snapshot=snapshot,
        relation_tone=relation_store.current_tone(),
        memory_activation=memory_store.recent_activation(),
        narrative_pressure=memory_store.narrative_pressure(),
        updated_at=now_ts,
    )
    return ExistentialSnapshot(
        phase=Phase.DOZE,
        self_view=drifted.self_view,
        world_view=drifted.world_view,
        openness=drifted.openness,
        updated_at=drifted.updated_at,
    )
