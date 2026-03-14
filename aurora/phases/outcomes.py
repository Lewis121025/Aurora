from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.reweave import SleepMutation
from aurora.runtime.contracts import Phase, PhaseTransition


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    phase: Phase
    transition: PhaseTransition
    mutation: SleepMutation | None = None
