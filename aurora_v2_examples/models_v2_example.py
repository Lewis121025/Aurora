from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Tone = Literal["warm", "neutral", "cold", "boundary"]
MoveKind = Literal["share", "probe", "pressure", "repair", "approach", "withhold", "boundary", "silence"]
TraceChannel = Literal["warmth", "hurt", "recognition", "distance", "curiosity", "boundary"]
AssocKind = Literal["resonance", "contrast", "repair", "boundary", "chapter"]


class Phase(str, Enum):
    AWAKE = "awake"
    DOZE = "doze"
    SLEEP = "sleep"


@dataclass(frozen=True, slots=True)
class InteractionTurn:
    """
    与你当前版本相比，最关键的增量是 relation_id。
    session_id 是会话，relation_id 是持续关系对象。
    """
    turn_id: str
    relation_id: str
    session_id: str
    speaker: Literal["user", "aurora", "system"]
    text: str
    created_at: float


@dataclass(frozen=True, slots=True)
class Fragment:
    """
    记忆片段不再只是 text，而是带有显著性、未完成性、关系归属和章节归属。
    """
    fragment_id: str
    turn_id: str | None
    relation_id: str | None
    surface: str
    vividness: float
    salience: float
    unresolvedness: float
    chapter_ids: tuple[str, ...]
    created_at: float


@dataclass(frozen=True, slots=True)
class TraceResidue:
    """
    Trace 不再只是 mode+intensity，而是类型化残留。
    """
    trace_id: str
    fragment_id: str
    channel: TraceChannel
    intensity: float
    decay: float
    created_at: float


@dataclass(frozen=True, slots=True)
class AssociationEdge:
    """
    与当前 AssociationDelta 不同，这里是持久边对象，不是一次性的变化量。
    """
    edge_id: str
    source_fragment_id: str
    target_fragment_id: str
    kind: AssocKind
    weight: float
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class Chapter:
    """
    sleep 之后新增的叙事组织对象。
    """
    chapter_id: str
    relation_id: str | None
    title: str
    motif: str
    anchor_fragment_ids: tuple[str, ...]
    coherence: float
    tension: float
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class RelationMoment:
    """
    关系时刻记录双向动作，而不是只记录用户一句话。
    """
    moment_id: str
    relation_id: str
    session_id: str
    user_turn_id: str
    aurora_turn_id: str | None
    tone: Tone
    user_move: MoveKind
    aurora_move: MoveKind
    effect_channels: tuple[TraceChannel, ...]
    summary: str
    created_at: float


@dataclass(slots=True)
class RelationState:
    relation_id: str
    current_tone: Tone = "neutral"
    trust: float = 0.0
    boundary_tension: float = 0.0
    repairability: float = 0.5
    shared_chapter_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class ExistentialSnapshot:
    """
    兼容你当前三元组设计，但增加两个过程性压力值。
    下一阶段你可以再把它替换成更隐式的 latent bundle。
    """
    phase: Phase
    self_view: float
    world_view: float
    openness: float
    coherence_pressure: float
    sleep_pressure: float
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
    relation_id: str
    relation_tone: Tone
    relation_strength: float
    recalled_fragments: tuple[Fragment, ...]
    active_chapters: tuple[Chapter, ...]
    active_channels: tuple[TraceChannel, ...]
    touch_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AwakeOutcome:
    response_text: str
    snapshot: ExistentialSnapshot
    touch_modes: tuple[str, ...]
    user_fragment: Fragment
    aurora_fragment: Fragment | None
    traces: tuple[TraceResidue, ...]
    associations: tuple[AssociationEdge, ...]
    relation_moment: RelationMoment
    transition: PhaseTransition | None
