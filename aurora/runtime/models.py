from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from aurora.phases.phase_types import Phase

Tone = Literal["warm", "neutral", "cold", "boundary"]


@dataclass(frozen=True, slots=True)
class InteractionTurn:
    turn_id: str
    session_id: str
    speaker: str
    text: str
    created_at: float


@dataclass(frozen=True, slots=True)
class Fragment:
    fragment_id: str
    turn_id: str
    text: str
    created_at: float


@dataclass(frozen=True, slots=True)
class Trace:
    trace_id: str
    turn_id: str
    mode: str
    intensity: float
    created_at: float


@dataclass(frozen=True, slots=True)
class AssociationDelta:
    association_id: str
    source_fragment_id: str
    target_fragment_id: str
    weight: float
    created_at: float


@dataclass(frozen=True, slots=True)
class RelationMoment:
    moment_id: str
    session_id: str
    turn_id: str
    tone: Tone
    summary: str
    created_at: float


@dataclass(frozen=True, slots=True)
class ExistentialSnapshot:
    phase: Phase
    self_view: float
    world_view: float
    openness: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class PhaseTransition:
    transition_id: str
    from_phase: Phase
    to_phase: Phase
    reason: str
    created_at: float


@dataclass(frozen=True, slots=True)
class ExpressionContext:
    input_text: str
    relation_tone: Tone
    relation_strength: float
    memory_snippets: tuple[str, ...]
    touch_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AwakeOutcome:
    response_text: str
    snapshot: ExistentialSnapshot
    touch_modes: tuple[str, ...]
    fragment: Fragment
    traces: tuple[Trace, ...]
    associations: tuple[AssociationDelta, ...]
    relation_moment: RelationMoment
    transition: PhaseTransition | None


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    snapshot: ExistentialSnapshot
    transition: PhaseTransition


@dataclass(slots=True)
class RuntimeState:
    snapshot: ExistentialSnapshot
    transitions: list[PhaseTransition] = field(default_factory=list)
