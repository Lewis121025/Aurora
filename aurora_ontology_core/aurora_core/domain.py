from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class Phase(str, Enum):
    AWAKE = "awake"
    DOZE = "doze"
    SLEEP = "sleep"


class Speaker(str, Enum):
    USER = "user"
    AURORA = "aurora"
    SYSTEM = "system"


class TraceChannel(str, Enum):
    WARMTH = "warmth"
    HURT = "hurt"
    RECOGNITION = "recognition"
    DISTANCE = "distance"
    CURIOSITY = "curiosity"
    BOUNDARY = "boundary"
    WONDER = "wonder"
    REPAIR = "repair"


class AssocKind(str, Enum):
    RESONANCE = "resonance"
    CONTRAST = "contrast"
    CAUSAL_GUESS = "causal_guess"
    REPAIR = "repair"
    BOUNDARY = "boundary"
    CHAPTER = "chapter"
    TEMPORAL = "temporal"


class RelationMove(str, Enum):
    APPROACH = "approach"
    WITHHOLD = "withhold"
    BOUNDARY = "boundary"
    REPAIR = "repair"
    OBSERVE = "observe"


class ChapterStatus(str, Enum):
    OPEN = "open"
    SETTLING = "settling"
    DORMANT = "dormant"


@dataclass(frozen=True, slots=True)
class InteractionTurn:
    turn_id: str
    relation_id: str
    session_id: str
    speaker: Speaker
    text: str
    created_at: float
    reply_to_turn_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Fragment:
    fragment_id: str
    turn_id: str | None
    relation_id: str | None
    surface: str
    semantic_tags: tuple[str, ...]
    touch_channels: tuple[TraceChannel, ...]
    vividness: float
    salience: float
    unresolvedness: float
    activation: float
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class TraceResidue:
    trace_id: str
    fragment_id: str
    relation_id: str | None
    channel: TraceChannel
    intensity: float
    persistence: float
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class AssociationEdge:
    edge_id: str
    src_fragment_id: str
    dst_fragment_id: str
    kind: AssocKind
    weight: float
    evidence_count: int
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class Chapter:
    chapter_id: str
    relation_id: str
    title: str
    fragment_ids: tuple[str, ...]
    motif_channels: tuple[TraceChannel, ...]
    tension_level: float
    coherence: float
    open_questions: tuple[str, ...]
    status: ChapterStatus
    created_at: float
    last_rewoven_at: float


@dataclass(frozen=True, slots=True)
class RelationMoment:
    moment_id: str
    relation_id: str
    session_id: str
    user_turn_id: str
    aurora_turn_id: str
    user_channels: tuple[TraceChannel, ...]
    aurora_move: RelationMove
    user_move: RelationMove
    boundary_signal: float
    resonance_score: float
    note: str
    created_at: float


@dataclass(frozen=True, slots=True)
class RelationState:
    relation_id: str
    trust: float = 0.0
    familiarity: float = 0.0
    reciprocity: float = 0.0
    boundary_tension: float = 0.0
    repairability: float = 0.5
    active_chapter_ids: tuple[str, ...] = ()
    motif_channels: tuple[TraceChannel, ...] = ()
    last_contact_at: float = 0.0


@dataclass(frozen=True, slots=True)
class BeingState:
    phase: Phase
    self_continuity: float
    world_trust: float
    relation_readiness: float
    boundary_tension: float
    narrative_pressure: float
    sleep_pressure: float
    current_relation_id: str | None = None
    active_chapter_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReweaveResult:
    chapter_ids: tuple[str, ...]
    coherence_shift: float
    tension_shift: float
    relation_bias: float
    retired_edges: int
    created_edges: int
