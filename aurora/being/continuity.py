from __future__ import annotations

from aurora.being.profile import load_dynamics_profile
from aurora.phases.phase_types import Phase
from aurora.runtime.models import Tone
from aurora.runtime.models import ExistentialSnapshot


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def evolve_snapshot(
    snapshot: ExistentialSnapshot,
    touch_forces: dict[str, float],
    relation_tone: Tone,
    updated_at: float,
) -> ExistentialSnapshot:
    profile = load_dynamics_profile()
    self_view = snapshot.self_view
    world_view = snapshot.world_view
    openness = snapshot.openness
    touch_strength = max(touch_forces.values(), default=0.0)

    for mode, force in touch_forces.items():
        delta = profile.mode_deltas.get(mode)
        if delta is None:
            continue
        self_view += delta.self_delta * force
        world_view += delta.world_delta * force
        openness += delta.open_delta * force

    coupling = 0.06 * openness
    self_world_gap = world_view - self_view
    self_view += coupling * self_world_gap
    world_view -= coupling * self_world_gap * 0.5
    openness += 0.03 * (0.5 - abs(self_world_gap))

    relation_delta = profile.relation_bias[relation_tone]
    world_view += relation_delta.world_delta * (0.5 + 0.5 * touch_strength)
    openness += relation_delta.open_delta * 0.5

    return ExistentialSnapshot(
        phase=Phase.AWAKE,
        self_view=_clamp(self_view, -1.0, 1.0),
        world_view=_clamp(world_view, -1.0, 1.0),
        openness=_clamp(openness, 0.0, 1.0),
        updated_at=updated_at,
    )
