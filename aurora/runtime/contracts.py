"""Aurora runtime contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar, cast

AtomKind = Literal["evidence", "episode", "semantic", "procedural", "cognitive", "affective", "narrative", "inhibition"]
AtomStatus = Literal["active", "superseded", "inhibited"]
EventKind = Literal["user_turn", "assistant_turn", "compile_failure"]
SemanticScope = Literal["self", "world"]
RecallMode = Literal["blended", "episodic", "semantic", "procedural", "cognitive", "affective", "narrative"]
RecallTemporalScope = Literal["current", "historical", "both"]
MemoryKind = Literal["episode", "semantic", "procedural", "cognitive", "affective", "narrative"]
NarrativeStatus = Literal["active", "resolved"]

SEMANTIC_ATTRIBUTE_LOCATION_CURRENT = "location.current"
SEMANTIC_ATTRIBUTE_WORK_LOCATION = "work.location"
SEMANTIC_ATTRIBUTE_PREFERENCE_LIKE = "preference.like"
PROCEDURAL_TRIGGER_PLAN = "plan"
PROCEDURAL_TRIGGER_ASSISTANT_COMMITMENT = "assistant_commitment"


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float into the target range."""
    return max(lo, min(hi, value))


@dataclass(frozen=True, slots=True)
class AffectMarker:
    """Emotion attached to an episode view or atom."""

    label: str
    intensity: float
    valence: float


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
class EpisodeContent:
    """Autobiographical episode payload."""

    scene_type: str
    title: str
    summary: str
    actors: tuple[str, ...]
    setting: str
    time_span: TimeSpan
    emotion_markers: tuple[AffectMarker, ...]
    text: str


@dataclass(frozen=True, slots=True)
class SemanticContent:
    """Semantic memory payload."""

    subject: str
    attribute: str
    value: str
    text: str


@dataclass(frozen=True, slots=True)
class ProceduralContent:
    """Procedural memory payload."""

    rule: str
    trigger: str
    steps: tuple[str, ...]
    text: str
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class CognitiveContent:
    """Structured cognitive memory payload."""

    beliefs: tuple[str, ...]
    goals: tuple[str, ...]
    conflicts: tuple[str, ...]
    intentions: tuple[str, ...]
    commitments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AffectiveContent:
    """Affective memory payload."""

    mood: str
    valence: float
    intensity: float
    feelings: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class NarrativeContent:
    """Narrative memory payload."""

    theme: str
    storyline: str
    status: NarrativeStatus
    episode_ids: tuple[str, ...]
    unresolved_threads: tuple[str, ...]
    role_changes: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class InhibitionContent:
    """Inhibition payload."""

    summary: str
    target_summary: str
    text: str


AtomContent = (
    EvidenceContent
    | EpisodeContent
    | SemanticContent
    | ProceduralContent
    | CognitiveContent
    | AffectiveContent
    | NarrativeContent
    | InhibitionContent
)


@dataclass(frozen=True, slots=True)
class MemoryAtom:
    """Unified persistent memory unit."""

    atom_id: str
    subject_id: str
    atom_kind: AtomKind
    content: AtomContent
    status: AtomStatus
    confidence: float
    salience: float
    accessibility: float
    scope: SemanticScope | None = None
    source_atom_ids: tuple[str, ...] = ()
    supersedes_atom_id: str | None = None
    inhibits_atom_ids: tuple[str, ...] = ()
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class SemanticMemoryItem:
    """Public semantic memory view item."""

    subject: str
    attribute: str
    value: str
    text: str


@dataclass(frozen=True, slots=True)
class ProceduralMemoryItem:
    """Public procedural memory view item."""

    rule: str
    trigger: str
    steps: tuple[str, ...]
    text: str
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class EpisodeMemoryItem:
    """Public episodic memory view item."""

    title: str
    summary: str
    actors: tuple[str, ...]
    setting: str
    time_span: TimeSpan
    emotion_markers: tuple[AffectMarker, ...]
    text: str


@dataclass(frozen=True, slots=True)
class ActiveCognition:
    """Public structured cognitive state."""

    beliefs: tuple[str, ...]
    goals: tuple[str, ...]
    conflicts: tuple[str, ...]
    intentions: tuple[str, ...]
    commitments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AffectiveState:
    """Public affective state."""

    mood: str
    valence: float
    intensity: float
    active_feelings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NarrativeArc:
    """Public narrative arc."""

    theme: str
    storyline: str
    status: NarrativeStatus
    episode_count: int
    unresolved_threads: tuple[str, ...] = ()
    role_changes: tuple[str, ...] = ()
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class NarrativeState:
    """Public narrative state."""

    arcs: tuple[NarrativeArc, ...]
    active_themes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecallHit:
    """Public recall hit."""

    memory_kind: MemoryKind
    content: str
    score: float
    why_recalled: str


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Recall result for one subject-scoped query."""

    subject_id: str
    query: str
    temporal_scope: RecallTemporalScope
    mode: RecallMode
    hits: tuple[RecallHit, ...]


@dataclass(frozen=True, slots=True)
class TurnOutput:
    """Hot-path turn output."""

    turn_id: str
    subject_id: str
    response_text: str
    recall_used: bool
    applied_atom_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SubjectMemoryState:
    """Current public memory state for one subject."""

    subject_id: str
    semantic_self_model: tuple[SemanticMemoryItem, ...]
    semantic_world_model: tuple[SemanticMemoryItem, ...]
    procedural_memory: tuple[ProceduralMemoryItem, ...]
    active_cognition: ActiveCognition
    affective_state: AffectiveState
    narrative_state: NarrativeState
    recent_episodes: tuple[EpisodeMemoryItem, ...]


def atom_content_to_dict(content: AtomContent) -> dict[str, Any]:
    """Convert one typed payload into its stable JSON form."""
    return asdict(content)


def atom_content_from_dict(atom_kind: AtomKind, raw: dict[str, Any]) -> AtomContent:
    """Decode one persisted payload into its typed form."""
    if atom_kind == "evidence":
        return EvidenceContent(
            event_kind=cast(EventKind, _string(raw, "event_kind")),
            role=_string(raw, "role"),
            text=_string(raw, "text"),
            payload=_dict(raw.get("payload")),
        )
    if atom_kind == "episode":
        return EpisodeContent(
            scene_type=_string(raw, "scene_type"),
            title=_string(raw, "title"),
            summary=_string(raw, "summary"),
            actors=_strings(raw.get("actors")),
            setting=_string(raw, "setting"),
            time_span=TimeSpan(
                start=float(_dict(raw.get("time_span")).get("start", 0.0)),
                end=float(_dict(raw.get("time_span")).get("end", 0.0)),
            ),
            emotion_markers=tuple(
                AffectMarker(
                    label=_string(item, "label"),
                    intensity=float(item.get("intensity", 0.0)),
                    valence=float(item.get("valence", 0.0)),
                )
                for item in _dicts(raw.get("emotion_markers"))
            ),
            text=_string(raw, "text"),
        )
    if atom_kind == "semantic":
        return SemanticContent(
            subject=_string(raw, "subject"),
            attribute=_string(raw, "attribute"),
            value=_string(raw, "value"),
            text=_string(raw, "text"),
        )
    if atom_kind == "procedural":
        return ProceduralContent(
            rule=_string(raw, "rule"),
            trigger=_string(raw, "trigger"),
            steps=_strings(raw.get("steps")),
            text=_string(raw, "text"),
            owner=_optional_string(raw.get("owner")),
        )
    if atom_kind == "cognitive":
        return CognitiveContent(
            beliefs=_strings(raw.get("beliefs")),
            goals=_strings(raw.get("goals")),
            conflicts=_strings(raw.get("conflicts")),
            intentions=_strings(raw.get("intentions")),
            commitments=_strings(raw.get("commitments")),
        )
    if atom_kind == "affective":
        return AffectiveContent(
            mood=_string(raw, "mood"),
            valence=float(raw.get("valence", 0.0)),
            intensity=float(raw.get("intensity", 0.0)),
            feelings=_strings(raw.get("feelings")),
            text=_string(raw, "text"),
        )
    if atom_kind == "narrative":
        return NarrativeContent(
            theme=_string(raw, "theme"),
            storyline=_string(raw, "storyline"),
            status=cast(NarrativeStatus, _string(raw, "status")),
            episode_ids=_strings(raw.get("episode_ids")),
            unresolved_threads=_strings(raw.get("unresolved_threads")),
            role_changes=_strings(raw.get("role_changes")),
            text=_string(raw, "text"),
        )
    return InhibitionContent(
        summary=_string(raw, "summary"),
        target_summary=_string(raw, "target_summary"),
        text=_string(raw, "text"),
    )


_ContentType = TypeVar(
    "_ContentType",
    EvidenceContent,
    EpisodeContent,
    SemanticContent,
    ProceduralContent,
    CognitiveContent,
    AffectiveContent,
    NarrativeContent,
    InhibitionContent,
)


def evidence_content(atom: MemoryAtom) -> EvidenceContent:
    return _expect_content(atom, "evidence", EvidenceContent)


def episode_content(atom: MemoryAtom) -> EpisodeContent:
    return _expect_content(atom, "episode", EpisodeContent)


def semantic_content(atom: MemoryAtom) -> SemanticContent:
    return _expect_content(atom, "semantic", SemanticContent)


def procedural_content(atom: MemoryAtom) -> ProceduralContent:
    return _expect_content(atom, "procedural", ProceduralContent)


def cognitive_content(atom: MemoryAtom) -> CognitiveContent:
    return _expect_content(atom, "cognitive", CognitiveContent)


def affective_content(atom: MemoryAtom) -> AffectiveContent:
    return _expect_content(atom, "affective", AffectiveContent)


def narrative_content(atom: MemoryAtom) -> NarrativeContent:
    return _expect_content(atom, "narrative", NarrativeContent)


def inhibition_content(atom: MemoryAtom) -> InhibitionContent:
    return _expect_content(atom, "inhibition", InhibitionContent)


def _expect_content(atom: MemoryAtom, atom_kind: AtomKind, content_type: type[_ContentType]) -> _ContentType:
    if atom.atom_kind != atom_kind or not isinstance(atom.content, content_type):
        raise TypeError(f"atom {atom.atom_id} has mismatched payload for kind {atom.atom_kind}")
    return atom.content


def _string(raw: dict[str, Any], key: str) -> str:
    return str(raw.get(key, "")).strip()


def _optional_string(value: object) -> str | None:
    text = str(value).strip()
    return text or None


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _dicts(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_dict(item) for item in value if isinstance(item, dict))
