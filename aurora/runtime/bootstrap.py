from __future__ import annotations

from aurora.phases.phase_types import Phase
from aurora.runtime.models import ExistentialSnapshot


def initial_snapshot(now_ts: float) -> ExistentialSnapshot:
    return ExistentialSnapshot(
        phase=Phase.AWAKE,
        self_view=0.05,
        world_view=0.10,
        openness=0.70,
        updated_at=now_ts,
    )
