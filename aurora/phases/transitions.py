from __future__ import annotations

from uuid import uuid4

from aurora.runtime.models import Phase, PhaseTransition


def phase_transition(
    from_phase: Phase, to_phase: Phase, reason: str, created_at: float
) -> PhaseTransition:
    return PhaseTransition(
        transition_id=f"pt_{uuid4().hex[:10]}",
        from_phase=from_phase,
        to_phase=to_phase,
        reason=reason,
        created_at=created_at,
    )
