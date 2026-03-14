from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Literal


class Phase(str, Enum):
    AWAKE = "awake"
    DOZE = "doze"
    SLEEP = "sleep"


class Speaker(str, Enum):
    USER = "user"
    AURORA = "aurora"
    SYSTEM = "system"


class AssocKind(str, Enum):
    RESONANCE = "resonance"
    CONTRAST = "contrast"
    CAUSAL_GUESS = "causal_guess"
    REPAIR = "repair"
    BOUNDARY = "boundary"
    CHAPTER = "chapter"
    RELATION = "relation"


class TraceChannel(str, Enum):
    WARMTH = "warmth"
    HURT = "hurt"
    RECOGNITION = "recognition"
    DISTANCE = "distance"
    CURIOSITY = "curiosity"
    BOUNDARY = "boundary"
    REPAIR = "repair"
    COHERENCE = "coherence"


class ChapterRole(str, Enum):
    SEED = "seed"
    TURNING_POINT = "turning_point"
    RUPTURE = "rupture"
    REPAIR_ATTEMPT = "repair_attempt"
    MOTIF = "motif"
    UNRESOLVED_KNOT = "unresolved_knot"
    ANCHOR = "anchor"


AuroraMove = Literal[
    "approach",
    "withhold",
    "boundary",
    "repair",
    "silence",
]


@dataclass(frozen=True, slots=True)
class InteractionTurn:
    turn_id: str
    relation_id: str
    session_id: str
    speaker: Speaker
    text: str
    created_at: float


@dataclass(frozen=True, slots=True)
class Fragment:
    fragment_id: str
    relation_id: str
    turn_id: str | None
    surface: str
    tags: tuple[str, ...]
    vividness: float
    salience: float
    unresolvedness: float
    chapter_ids: tuple[str, ...]
    created_at: float
    last_touched_at: float
    activation_count: int = 0

    def touched(self, at: float, delta_salience: float = 0.08) -> "Fragment":
        return replace(
            self,
            salience=min(1.0, self.salience + delta_salience),
            last_touched_at=at,
            activation_count=self.activation_count + 1,
        )


@dataclass(frozen=True, slots=True)
class TraceResidue:
    trace_id: str
    relation_id: str
    fragment_id: str
    channel: TraceChannel
    intensity: float
    decay_rate: float
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class AssociationEdge:
    edge_id: str
    src_fragment_id: str
    dst_fragment_id: str
    kind: AssocKind
    weight: float
    evidence: tuple[str, ...]
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class Chapter:
    chapter_id: str
    relation_id: str
    title: str
    motif: str
    fragment_ids: tuple[str, ...]
    roles: dict[str, ChapterRole]
    tension: float
    coherence: float
    created_at: float
    last_rewoven_at: float


@dataclass(frozen=True, slots=True)
class RelationMoment:
    moment_id: str
    relation_id: str
    user_turn_id: str
    aurora_turn_id: str | None
    user_channels: tuple[TraceChannel, ...]
    aurora_move: AuroraMove
    boundary_crossed: bool
    repair_attempted: bool
    summary: str
    created_at: float


@dataclass(slots=True)
class RelationState:
    relation_id: str
    trust: float = 0.0
    reciprocity: float = 0.0
    boundary_tension: float = 0.0
    repairability: float = 0.5
    distance: float = 0.0
    shared_chapters: set[str] = field(default_factory=set)
    last_contact_at: float = 0.0

    def snapshot(self) -> dict[str, float | tuple[str, ...]]:
        return {
            "trust": round(self.trust, 4),
            "reciprocity": round(self.reciprocity, 4),
            "boundary_tension": round(self.boundary_tension, 4),
            "repairability": round(self.repairability, 4),
            "distance": round(self.distance, 4),
            "shared_chapters": tuple(sorted(self.shared_chapters)),
        }


@dataclass(slots=True)
class BeingState:
    phase: Phase = Phase.AWAKE
    continuity_pressure: float = 0.0
    sleep_pressure: float = 0.0
    coherence_pressure: float = 0.0
    softness: float = 0.5
    boundary_tension: float = 0.0
    self_vector: dict[str, float] = field(
        default_factory=lambda: {
            "recognition": 0.0,
            "fragility": 0.0,
            "openness": 0.0,
            "agency": 0.0,
        }
    )
    world_vector: dict[str, float] = field(
        default_factory=lambda: {
            "welcome": 0.0,
            "risk": 0.0,
            "mystery": 0.0,
            "stability": 0.0,
        }
    )
    recent_chapter_bias: tuple[str, ...] = ()

    def drift(self, updates: dict[str, float], world_updates: dict[str, float]) -> None:
        for key, value in updates.items():
            self.self_vector[key] = max(-1.0, min(1.0, self.self_vector.get(key, 0.0) + value))
        for key, value in world_updates.items():
            self.world_vector[key] = max(-1.0, min(1.0, self.world_vector.get(key, 0.0) + value))


@dataclass(frozen=True, slots=True)
class ActivationBundle:
    relation_state: dict[str, float | tuple[str, ...]]
    fragments: tuple[Fragment, ...]
    traces: tuple[TraceResidue, ...]
    chapters: tuple[Chapter, ...]
    dominant_channels: tuple[TraceChannel, ...]
    boundary_required: bool


@dataclass(frozen=True, slots=True)
class AwakeDraft:
    user_turn: InteractionTurn
    user_fragment: Fragment
    user_traces: tuple[TraceResidue, ...]
    activation: ActivationBundle
    response_hint: str


@dataclass(frozen=True, slots=True)
class ReweaveResult:
    chapter_ids: tuple[str, ...]
    updated_fragment_ids: tuple[str, ...]
    strengthened_edge_ids: tuple[str, ...]
    coherence_shift: float
    tension_shift: float
    self_drift: dict[str, float]
    world_drift: dict[str, float]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TurnCommit:
    user_turn: InteractionTurn
    aurora_turn: InteractionTurn
    user_fragment: Fragment
    aurora_fragment: Fragment
    user_traces: tuple[TraceResidue, ...]
    aurora_traces: tuple[TraceResidue, ...]
    response_text: str
    aurora_move: AuroraMove


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def iter_channels(traces: Iterable[TraceResidue]) -> tuple[TraceChannel, ...]:
    return tuple(sorted({trace.channel for trace in traces}, key=lambda item: item.value))
