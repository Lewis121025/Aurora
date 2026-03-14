from __future__ import annotations

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.runtime.state import RuntimeState


def initial_runtime_state() -> RuntimeState:
    return RuntimeState(orientation=Orientation(), metabolic=MetabolicState())
