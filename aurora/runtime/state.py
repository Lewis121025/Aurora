from __future__ import annotations

from dataclasses import dataclass, field

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.runtime.contracts import PhaseTransition


@dataclass(slots=True)
class RuntimeState:
    orientation: Orientation
    metabolic: MetabolicState
    transitions: list[PhaseTransition] = field(default_factory=list)
