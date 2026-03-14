from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Phase


@dataclass(slots=True)
class BeingState:
    phase: Phase = "awake"
    continuity_pressure: float = 0.20
    coherence_pressure: float = 0.25
    sleep_pressure: float = 0.15
    boundary_tension: float = 0.15
    approach_drive: float = 0.40
    withdrawal_drive: float = 0.20
    active_chapter_ids: tuple[str, ...] = ()
    last_transition_at: float = 0.0
    last_awake_at: float = 0.0
    history: list[str] = field(default_factory=list)

    def clamp(self) -> None:
        self.continuity_pressure = min(max(self.continuity_pressure, 0.0), 1.0)
        self.coherence_pressure = min(max(self.coherence_pressure, 0.0), 1.0)
        self.sleep_pressure = min(max(self.sleep_pressure, 0.0), 1.0)
        self.boundary_tension = min(max(self.boundary_tension, 0.0), 1.0)
        self.approach_drive = min(max(self.approach_drive, 0.0), 1.0)
        self.withdrawal_drive = min(max(self.withdrawal_drive, 0.0), 1.0)

    def note(self, message: str) -> None:
        self.history.append(message)
        self.history = self.history[-32:]
