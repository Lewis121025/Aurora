from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Phase(str, Enum):
    AWAKE = "awake"
    DOZE = "doze"
    SLEEP = "sleep"


class Speaker(str, Enum):
    USER = "user"
    AURORA = "aurora"


class TraceChannel(str, Enum):
    WARMTH = "warmth"
    HURT = "hurt"
    RECOGNITION = "recognition"
    DISTANCE = "distance"
    CURIOSITY = "curiosity"
    BOUNDARY = "boundary"
    REPAIR = "repair"
    COHERENCE = "coherence"
    WONDER = "wonder"


class AssocKind(str, Enum):
    RESONANCE = "resonance"
    CONTRAST = "contrast"
    REPAIR = "repair"
    BOUNDARY = "boundary"
    THREAD = "thread"
    RELATION = "relation"
    TEMPORAL = "temporal"
    KNOT = "knot"


AuroraMove = Literal["approach", "withhold", "boundary", "repair", "silence", "witness"]


@dataclass(frozen=True, slots=True)
class Turn:
    turn_id: str
    relation_id: str
    session_id: str
    speaker: Speaker
    text: str
    created_at: float


@dataclass(frozen=True, slots=True)
class PhaseTransition:
    transition_id: str
    from_phase: Phase
    to_phase: Phase
    reason: str
    created_at: float


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
