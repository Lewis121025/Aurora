"""Aurora v3 runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AtomType = Literal["fact", "rule", "lexicon", "loop", "revision", "forget"]
AtomStatus = Literal["active", "resolved", "superseded", "hidden"]
LoopType = Literal["commitment", "contradiction", "unfinished_thread", "unresolved_question"]
LoopStatus = Literal["active", "resolved"]
FactStatus = Literal["active", "superseded", "hidden"]
FactKind = Literal["profile", "preference", "current_state", "biographical"]
EventKind = Literal["user_turn", "assistant_turn", "compile_failure"]
RecallKind = Literal["atom"]
MemoryOpType = AtomType


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float into the target range."""
    return max(lo, min(hi, value))


@dataclass(slots=True)
class RelationField:
    """Hidden derived relation state."""

    relation_id: str
    trust: float = 0.35
    distance: float = 0.65
    warmth: float = 0.30
    tension: float = 0.0
    repair_debt: float = 0.0
    shared_lexicon: list[str] = field(default_factory=list)
    interaction_rules: list[str] = field(default_factory=list)
    last_compiled_at: float = 0.0


@dataclass(frozen=True, slots=True)
class MemoryAtom:
    """The only long-term semantic truth in Aurora."""

    atom_id: str
    relation_id: str
    atom_type: AtomType
    payload: dict[str, Any]
    status: AtomStatus
    confidence: float
    salience: float
    visibility: float
    evidence_event_ids: tuple[str, ...] = ()
    affects_atom_ids: tuple[str, ...] = ()
    supersedes_atom_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Append-only evidence event."""

    event_id: str
    relation_id: str
    kind: EventKind
    role: str
    text: str
    created_at: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryOp:
    """Compiler output before being applied as atoms."""

    op_type: MemoryOpType
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OpenLoop:
    """Derived open loop view."""

    loop_id: str
    relation_id: str
    loop_type: LoopType
    status: LoopStatus
    summary: str
    urgency: float
    opened_at: float
    updated_at: float
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FactRecord:
    """Derived fact view."""

    fact_id: str
    relation_id: str
    content: str
    fact_kind: FactKind
    document_date: float
    event_date: float
    status: FactStatus
    supersedes: str | None
    confidence: float
    visibility: float
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallHit:
    """Selected atom or evidence hit."""

    item_id: str
    kind: RecallKind
    content: str
    score: float
    why_recalled: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Recall result for one relation-scoped query."""

    relation_id: str
    query: str
    hits: tuple[RecallHit, ...]


@dataclass(frozen=True, slots=True)
class TurnOutput:
    """Hot-path turn output."""

    turn_id: str
    relation_id: str
    response_text: str
    recall_used: bool
    recalled_ids: tuple[str, ...]
    applied_atom_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RelationSnapshot:
    """Debug snapshot."""

    relation_id: str
    field: RelationField
    open_loops: tuple[OpenLoop, ...]
    facts: tuple[FactRecord, ...]
    atoms: tuple[MemoryAtom, ...]
    recent_events: tuple[EventRecord, ...]
