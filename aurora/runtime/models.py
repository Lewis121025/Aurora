from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
    INSIGHT = "insight"
    BOUNDARY = "boundary"
    CURIOSITY = "curiosity"
    AMBIENT = "ambient"


class AssocKind(str, Enum):
    TEMPORAL = "temporal"
    RESONANCE = "resonance"
    CONTRAST = "contrast"
    BOUNDARY = "boundary"
    REPAIR = "repair"
    THREAD = "thread"


class RelationMove(str, Enum):
    APPROACH = "approach"
    OBSERVE = "observe"
    BOUNDARY = "boundary"
    WITHHOLD = "withhold"
    REPAIR = "repair"


@dataclass(frozen=True, slots=True)
class Turn:
    turn_id: str
    relation_id: str
    session_id: str
    speaker: Speaker
    text: str
    created_at: float
    reply_to_turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class Fragment:
    fragment_id: str
    turn_id: str
    relation_id: str
    surface: str
    touch_channels: tuple[TraceChannel, ...]
    salience: float
    vividness: float
    unresolvedness: float
    activation: float
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class Trace:
    trace_id: str
    fragment_id: str
    relation_id: str
    channel: TraceChannel
    intensity: float
    persistence: float
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class Association:
    edge_id: str
    src_fragment_id: str
    dst_fragment_id: str
    kind: AssocKind
    weight: float
    evidence_count: int
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class Thread:
    thread_id: str
    relation_id: str
    fragment_ids: tuple[str, ...]
    motif_channels: tuple[TraceChannel, ...]
    coherence: float
    tension: float
    synopsis: str
    created_at: float
    last_rewoven_at: float


@dataclass(frozen=True, slots=True)
class Knot:
    knot_id: str
    relation_id: str
    fragment_ids: tuple[str, ...]
    channel: TraceChannel
    density: float
    heat: float
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class RelationMoment:
    moment_id: str
    relation_id: str
    session_id: str
    user_turn_id: str
    aurora_turn_id: str
    user_channels: tuple[TraceChannel, ...]
    user_move: RelationMove
    aurora_move: RelationMove
    boundary_signal: float
    resonance_score: float
    note: str
    created_at: float


@dataclass(frozen=True, slots=True)
class RelationFormation:
    relation_id: str
    trust: float
    familiarity: float
    reciprocity: float
    boundary_tension: float
    repairability: float
    active_thread_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    last_contact_at: float


@dataclass(frozen=True, slots=True)
class Orientation:
    relation_id: str
    self_orientation: float
    world_orientation: float
    relation_orientation: float
    narrative_tilt: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class MetabolicState:
    phase: Phase
    sleep_need: float
    current_relation_id: str | None
    active_thread_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    last_transition_at: float


@dataclass(frozen=True, slots=True)
class ReweaveMutation:
    mutation_id: str
    relation_id: str
    softened_fragment_ids: tuple[str, ...]
    strengthened_edge_ids: tuple[str, ...]
    created_thread_ids: tuple[str, ...]
    updated_thread_ids: tuple[str, ...]
    created_knot_ids: tuple[str, ...]
    updated_knot_ids: tuple[str, ...]
    orientation_delta: tuple[float, float, float, float]
    created_at: float


@dataclass(frozen=True, slots=True)
class ActivationView:
    relation_id: str
    relation: RelationFormation
    orientation: Orientation
    fragments: tuple[Fragment, ...]
    threads: tuple[Thread, ...]
    knots: tuple[Knot, ...]
    channels: tuple[TraceChannel, ...]
    boundary_required: bool


@dataclass(frozen=True, slots=True)
class PhaseTransition:
    transition_id: str
    from_phase: Phase
    to_phase: Phase
    reason: str
    created_at: float


@dataclass(frozen=True, slots=True)
class AwakeOutcome:
    user_turn: Turn
    aurora_turn: Turn
    touch_channels: tuple[TraceChannel, ...]
    response_text: str
    activation: ActivationView
    transition: PhaseTransition | None


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    metabolic: MetabolicState
    transition: PhaseTransition
    mutation: ReweaveMutation | None = None


@dataclass(slots=True)
class RuntimeState:
    metabolic: MetabolicState
    transitions: list[PhaseTransition] = field(default_factory=list)
