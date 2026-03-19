"""Aurora runtime contracts for the memory field kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar, cast

AtomKind = Literal[
    "evidence",
    "memory",
    "episode",
    "inhibition",
]
TranscriptRole = Literal["user", "assistant"]
EventKind = Literal["user_turn", "assistant_turn", "compile_failure"]
_ATOM_KINDS = frozenset({"evidence", "memory", "episode", "inhibition"})
_EVENT_KINDS = frozenset({"user_turn", "assistant_turn", "compile_failure"})
_TRANSCRIPT_ROLES = frozenset({"user", "assistant"})


@dataclass(frozen=True, slots=True)
class TimeSpan:
    """Time span of one autobiographical episode."""

    start: float
    end: float


@dataclass(frozen=True, slots=True)
class EvidenceContent:
    """Append-only evidence payload."""

    event_kind: EventKind
    role: str
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MemoryContent:
    """Text-first memory-field payload."""

    text: str
    referents: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeContent:
    """Episode payload."""

    text: str
    time_span: TimeSpan
    referents: tuple[str, ...] = ()


AtomContent = EvidenceContent | MemoryContent | EpisodeContent


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    """One ordered item in a session transcript."""

    role: TranscriptRole
    text: str
    created_at: float


@dataclass(frozen=True, slots=True)
class MemoryAtom:
    """Immutable persistent memory node."""

    atom_id: str
    subject_id: str
    atom_kind: AtomKind
    content: AtomContent
    confidence: float
    salience: float
    source_atom_ids: tuple[str, ...] = ()
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class MemoryEdge:
    """Immutable directed memory-field edge."""

    edge_id: str
    subject_id: str
    source_atom_id: str
    target_atom_id: str
    influence: float
    confidence: float
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class ActivatedAtom:
    """Activated memory node view."""

    atom_id: str
    atom_kind: AtomKind
    text: str
    activation: float
    confidence: float
    salience: float
    created_at: float


@dataclass(frozen=True, slots=True)
class ActivatedEdge:
    """Activated edge view."""

    source_atom_id: str
    target_atom_id: str
    influence: float
    confidence: float


@dataclass(frozen=True, slots=True)
class SubjectMemoryState:
    """Current memory-field view for one subject."""

    subject_id: str
    summary: str
    atoms: tuple[ActivatedAtom, ...]
    edges: tuple[ActivatedEdge, ...]


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Query-activated memory-field view."""

    subject_id: str
    query: str
    summary: str
    atoms: tuple[ActivatedAtom, ...]
    edges: tuple[ActivatedEdge, ...]


@dataclass(frozen=True, slots=True)
class IngestOutput:
    """Session-level durable ingest output."""

    subject_id: str
    session_id: str
    created_atom_ids: tuple[str, ...] = ()
    created_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnOutput:
    """Turn output."""

    turn_id: str
    subject_id: str
    session_id: str
    response_text: str
    recall_used: bool
    segment_committed: bool = False


def atom_text(atom: MemoryAtom) -> str:
    """Render one atom into plain text."""
    content = atom.content
    if isinstance(content, EvidenceContent):
        return content.text.strip()
    if isinstance(content, MemoryContent):
        return content.text.strip()
    if isinstance(content, EpisodeContent):
        return content.text.strip()
    return ""


def atom_content_to_dict(content: AtomContent) -> dict[str, Any]:
    """Convert one typed payload into its stable JSON form."""
    return asdict(content)


def atom_content_from_dict(atom_kind: AtomKind, raw: dict[str, Any]) -> AtomContent:
    """Decode one persisted payload into its typed form."""
    if atom_kind == "evidence":
        return EvidenceContent(
            event_kind=_event_kind(_required_string(raw, "event_kind")),
            role=_required_string(raw, "role"),
            text=_required_string(raw, "text"),
            payload=_required_dict(raw, "payload"),
        )
    if atom_kind == "episode":
        return EpisodeContent(
            text=_required_string(raw, "text"),
            time_span=_time_span(_required_dict(raw, "time_span")),
            referents=_required_strings(raw.get("referents")),
        )
    return MemoryContent(
        text=_required_string(raw, "text"),
        referents=_required_strings(raw.get("referents")),
    )


_ContentType = TypeVar(
    "_ContentType",
    EvidenceContent,
    MemoryContent,
    EpisodeContent,
)


def evidence_content(atom: MemoryAtom) -> EvidenceContent:
    return _expect_content(atom, "evidence", EvidenceContent)


def _expect_content(atom: MemoryAtom, atom_kind: AtomKind, content_type: type[_ContentType]) -> _ContentType:
    if atom.atom_kind != atom_kind or not isinstance(atom.content, content_type):
        raise TypeError(f"atom {atom.atom_id} has mismatched payload for kind {atom.atom_kind}")
    return atom.content


def atom_kind_from_value(value: object) -> AtomKind:
    text = _non_empty_string(value, "atom_kind")
    if text not in _ATOM_KINDS:
        raise ValueError(f"invalid atom_kind: {text}")
    return cast(AtomKind, text)


def transcript_role_from_value(value: object) -> TranscriptRole:
    text = _non_empty_string(value, "role")
    if text not in _TRANSCRIPT_ROLES:
        raise ValueError(f"invalid transcript role: {text}")
    return cast(TranscriptRole, text)


def _event_kind(value: str) -> EventKind:
    if value not in _EVENT_KINDS:
        raise ValueError(f"invalid event_kind: {value}")
    return cast(EventKind, value)


def _required_string(raw: dict[str, Any], key: str) -> str:
    if key not in raw:
        raise ValueError(f"missing required content field: {key}")
    return _non_empty_string(raw[key], key)


def _required_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("referents must be a JSON array")
    referents: list[str] = []
    for item in value:
        referent = _non_empty_string(item, "referent")
        referents.append(referent)
    return tuple(referents)


def _required_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw:
        raise ValueError(f"missing required content field: {key}")
    value = raw[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return {str(dict_key): item for dict_key, item in value.items()}


def _time_span(raw: dict[str, Any]) -> TimeSpan:
    start = _required_float(raw, "start")
    end = _required_float(raw, "end")
    if end < start:
        raise ValueError("episode time_span end must be >= start")
    return TimeSpan(start=start, end=end)


def _required_float(raw: dict[str, Any], key: str) -> float:
    if key not in raw:
        raise ValueError(f"missing required numeric field: {key}")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text
