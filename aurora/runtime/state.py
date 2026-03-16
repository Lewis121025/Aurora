from __future__ import annotations

from dataclasses import dataclass, field

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.runtime.contracts import PhaseTransition

TRANSITION_CAP = 256


@dataclass(slots=True)
class RuntimeState:
    orientation: Orientation
    metabolic: MetabolicState
    transitions: list[PhaseTransition] = field(default_factory=list)

    def append_transition(self, transition: PhaseTransition) -> None:
        self.transitions.append(transition)
        if len(self.transitions) > TRANSITION_CAP:
            del self.transitions[: len(self.transitions) - TRANSITION_CAP]
