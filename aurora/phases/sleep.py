from __future__ import annotations

from aurora.being.profile import load_dynamics_profile
from aurora.memory.reweave import run_sleep_reweave
from aurora.memory.store import MemoryStore
from aurora.phases.phase_types import Phase
from aurora.relation.store import RelationStore
from aurora.runtime.models import ExistentialSnapshot


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def run_sleep(
    snapshot: ExistentialSnapshot,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> ExistentialSnapshot:
    profile = load_dynamics_profile().sleep
    relation_tone = relation_store.current_tone()
    relation_strength = relation_store.tone_strength(relation_tone)
    delta = run_sleep_reweave(
        store=memory_store,
        relation_tone=relation_tone,
        relation_strength=relation_strength,
    )

    world_target = profile.world_targets[relation_tone]
    open_target = profile.open_targets[relation_tone]

    world_view = _clamp(
        snapshot.world_view + profile.world_scale * delta * (world_target - snapshot.world_view),
        -1.0,
        1.0,
    )
    openness = _clamp(
        snapshot.openness + profile.open_scale * delta * (open_target - snapshot.openness),
        0.0,
        1.0,
    )
    self_view = _clamp(
        snapshot.self_view
        + profile.self_scale * delta * (profile.self_target - snapshot.self_view),
        -1.0,
        1.0,
    )
    return ExistentialSnapshot(
        phase=Phase.SLEEP,
        self_view=self_view,
        world_view=world_view,
        openness=openness,
        updated_at=now_ts,
    )
