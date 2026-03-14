from __future__ import annotations

from aurora.being.profile import load_dynamics_profile
from aurora.runtime.models import Tone
from aurora.runtime.models import ExistentialSnapshot


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_doze_drift(
    snapshot: ExistentialSnapshot,
    relation_tone: Tone,
    memory_activation: float,
    narrative_pressure: float,
    updated_at: float,
) -> ExistentialSnapshot:
    profile = load_dynamics_profile().doze
    equilibrium_world = profile.world_equilibrium[relation_tone]
    equilibrium_open = profile.open_equilibrium[relation_tone]

    drift_factor = (
        profile.base_drift
        + profile.memory_scale * memory_activation
        + profile.narrative_scale * narrative_pressure
    )
    world_view = snapshot.world_view + drift_factor * (equilibrium_world - snapshot.world_view)
    openness = snapshot.openness + drift_factor * (equilibrium_open - snapshot.openness)
    self_view = snapshot.self_view + profile.self_scale * memory_activation * (
        profile.self_target - snapshot.self_view
    )

    return ExistentialSnapshot(
        phase=snapshot.phase,
        self_view=_clamp(self_view, -1.0, 1.0),
        world_view=_clamp(world_view, -1.0, 1.0),
        openness=_clamp(openness, 0.0, 1.0),
        updated_at=updated_at,
    )
