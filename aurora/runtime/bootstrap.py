from __future__ import annotations

from aurora.runtime.models import MetabolicState, Phase


def initial_metabolic(now_ts: float) -> MetabolicState:
    return MetabolicState(
        phase=Phase.AWAKE,
        sleep_need=0.0,
        current_relation_id=None,
        active_thread_ids=(),
        active_knot_ids=(),
        last_transition_at=now_ts,
    )
