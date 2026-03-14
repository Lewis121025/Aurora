from __future__ import annotations

from aurora.memory.store import MemoryStore
from aurora.runtime.models import AssociationDelta, Fragment, InteractionTurn, Trace


def form_memory_effects(
    store: MemoryStore,
    turn: InteractionTurn,
    touch_modes: tuple[str, ...],
    created_at: float,
) -> tuple[Fragment, tuple[Trace, ...], tuple[AssociationDelta, ...]]:
    return store.remember_turn(turn=turn, touch_modes=touch_modes, created_at=created_at)
