from __future__ import annotations

from aurora.being.continuity import evolve_snapshot
from aurora.being.touch import infer_touch_modes, touch_intensity
from aurora.expression.context import build_expression_context
from aurora.expression.response import render_response
from aurora.memory.formation import form_memory_effects
from aurora.phases.transitions import phase_transition
from aurora.phases.phase_types import Phase
from aurora.relation.store import RelationStore
from aurora.memory.store import MemoryStore
from aurora.runtime.models import (
    AwakeOutcome,
    ExistentialSnapshot,
    InteractionTurn,
    PhaseTransition,
)


def run_awake(
    turn: InteractionTurn,
    snapshot: ExistentialSnapshot,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
) -> AwakeOutcome:
    touch_modes = infer_touch_modes(turn.text)
    fragment, traces, associations = form_memory_effects(
        store=memory_store,
        turn=turn,
        touch_modes=touch_modes,
        created_at=now_ts,
    )
    relation_moment = relation_store.record_turn(
        turn=turn,
        touch_modes=touch_modes,
        created_at=now_ts,
    )

    updated_snapshot = evolve_snapshot(
        snapshot=snapshot,
        touch_modes=touch_modes,
        touch_strength=touch_intensity(touch_modes),
        relation_tone=relation_moment.tone,
        updated_at=now_ts,
    )
    context = build_expression_context(
        turn_text=turn.text,
        touch_modes=touch_modes,
        memory_store=memory_store,
        relation_store=relation_store,
    )
    response_text = render_response(context=context)

    transition: PhaseTransition | None = None
    if snapshot.phase is not Phase.AWAKE:
        transition = phase_transition(
            from_phase=snapshot.phase,
            to_phase=Phase.AWAKE,
            reason="incoming_turn",
            created_at=now_ts,
        )

    return AwakeOutcome(
        response_text=response_text,
        snapshot=updated_snapshot,
        touch_modes=touch_modes,
        fragment=fragment,
        traces=traces,
        associations=associations,
        relation_moment=relation_moment,
        transition=transition,
    )
