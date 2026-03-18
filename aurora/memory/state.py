"""Subject-centric state projection over unified memory atoms."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from aurora.runtime.contracts import (
    ActiveCognition,
    AffectMarker,
    AffectiveState,
    EpisodeMemoryItem,
    MemoryAtom,
    NarrativeArc,
    NarrativeState,
    ProceduralMemoryItem,
    SemanticMemoryItem,
    SubjectMemoryState,
    affective_content,
    cognitive_content,
    episode_content,
    evidence_content,
    inhibition_content,
    narrative_content,
    procedural_content,
    semantic_content,
)


def atom_text(atom: MemoryAtom) -> str:
    """Render one atom into compact recallable text."""
    text = _payload_text(atom.content)
    if text:
        return text

    if atom.atom_kind == "episode":
        episode = episode_content(atom)
        fragments = [episode.title.strip(), episode.summary.strip(), episode.setting.strip()]
        if episode.emotion_markers:
            fragments.append("/".join(marker.label for marker in episode.emotion_markers))
        return " - ".join(fragment for fragment in fragments if fragment).strip()

    if atom.atom_kind == "semantic":
        semantic = semantic_content(atom)
        return " ".join(part for part in (semantic.subject, semantic.attribute, semantic.value) if part).strip()

    if atom.atom_kind == "procedural":
        procedural = procedural_content(atom)
        if procedural.trigger:
            return f"{procedural.rule} ({procedural.trigger})".strip()
        return procedural.rule

    if atom.atom_kind == "cognitive":
        cognition = cognitive_content(atom)
        parts: list[str] = []
        for label, values in (
            ("beliefs", cognition.beliefs),
            ("goals", cognition.goals),
            ("conflicts", cognition.conflicts),
            ("intentions", cognition.intentions),
            ("commitments", cognition.commitments),
        ):
            if values:
                parts.append(f"{label}: {'; '.join(values)}")
        return " | ".join(parts)

    if atom.atom_kind == "affective":
        affect = affective_content(atom)
        if affect.feelings:
            return f"{affect.mood}: {'/'.join(affect.feelings)}"
        return affect.mood

    if atom.atom_kind == "narrative":
        narrative = narrative_content(atom)
        return " - ".join(part for part in (narrative.theme, narrative.storyline) if part).strip()

    if atom.atom_kind == "inhibition":
        return inhibition_content(atom).summary.strip()

    if atom.atom_kind == "evidence":
        return evidence_content(atom).text.strip()
    return ""


def visible_recall_atoms(atoms: Iterable[MemoryAtom], *, include_superseded: bool = False) -> tuple[MemoryAtom, ...]:
    """Return atoms that can participate in recall."""
    visible: list[MemoryAtom] = []
    for atom in atoms:
        if atom.accessibility <= 0.0 or atom.status == "inhibited":
            continue
        if not include_superseded and atom.status == "superseded":
            continue
        if atom.atom_kind in {"inhibition", "evidence"}:
            continue
        visible.append(atom)
    return tuple(visible)


def suppressed_source_ids(atoms: Iterable[MemoryAtom]) -> tuple[str, ...]:
    """Return lineage ids hidden from current continuity."""
    blocked: list[str] = []
    for atom in atoms:
        if atom.atom_kind == "evidence":
            continue
        if atom.atom_kind == "inhibition":
            if atom.status == "active":
                blocked.extend(atom.source_atom_ids)
            continue
        if atom.status == "superseded":
            blocked.append(atom.atom_id)
            continue
        if atom.status != "inhibited":
            continue
        blocked.extend((atom.atom_id, *atom.source_atom_ids))
    return _dedupe(blocked)


def project_subject_state(*, subject_id: str, atoms: tuple[MemoryAtom, ...]) -> SubjectMemoryState:
    """Project current public state from unified memory atoms."""
    projected_atoms = _current_state_atoms(atoms)
    semantic_self_model = tuple(
        _semantic_view(atom)
        for atom in projected_atoms
        if atom.atom_kind == "semantic" and atom.scope in {None, "self"}
    )
    semantic_world_model = tuple(
        _semantic_view(atom)
        for atom in projected_atoms
        if atom.atom_kind == "semantic" and atom.scope == "world"
    )
    procedural_memory = tuple(
        _procedural_view(atom)
        for atom in projected_atoms
        if atom.atom_kind == "procedural"
    )
    recent_episodes = tuple(
        _episode_view(atom)
        for atom in projected_atoms
        if atom.atom_kind == "episode"
    )[-8:]

    return SubjectMemoryState(
        subject_id=subject_id,
        semantic_self_model=semantic_self_model,
        semantic_world_model=semantic_world_model,
        procedural_memory=procedural_memory,
        active_cognition=_project_active_cognition(projected_atoms),
        affective_state=_project_affective_state(projected_atoms, recent_episodes),
        narrative_state=_project_narrative_state(projected_atoms),
        recent_episodes=recent_episodes,
    )


def _project_active_cognition(atoms: tuple[MemoryAtom, ...]) -> ActiveCognition:
    cognitive_atoms = [
        atom
        for atom in atoms
        if atom.atom_kind == "cognitive" and atom.status == "active" and atom.accessibility > 0.0
    ]
    if not cognitive_atoms:
        return ActiveCognition(
            beliefs=(),
            goals=(),
            conflicts=(),
            intentions=(),
            commitments=(),
        )

    content = cognitive_content(cognitive_atoms[-1])
    return ActiveCognition(
        beliefs=_dedupe(content.beliefs),
        goals=_dedupe(content.goals),
        conflicts=_dedupe(content.conflicts),
        intentions=_dedupe(content.intentions),
        commitments=_dedupe(content.commitments),
    )


def _project_affective_state(
    atoms: tuple[MemoryAtom, ...],
    recent_episodes: tuple[EpisodeMemoryItem, ...],
) -> AffectiveState:
    affective_atoms = [
        atom
        for atom in atoms
        if atom.atom_kind == "affective" and atom.status == "active" and atom.accessibility > 0.0
    ]
    if affective_atoms:
        content = affective_content(affective_atoms[-1])
        return AffectiveState(
            mood=content.mood,
            valence=content.valence,
            intensity=content.intensity,
            active_feelings=_dedupe(content.feelings),
        )

    markers: list[AffectMarker] = [marker for episode in recent_episodes for marker in episode.emotion_markers]
    if markers:
        marker_group = tuple(markers)
        return AffectiveState(
            mood=_mood_from_markers(marker_group),
            valence=sum(marker.valence for marker in marker_group) / len(marker_group),
            intensity=sum(marker.intensity for marker in marker_group) / len(marker_group),
            active_feelings=_dedupe(marker.label for marker in marker_group),
        )

    return AffectiveState(mood="neutral", valence=0.0, intensity=0.0, active_feelings=())


def _project_narrative_state(atoms: tuple[MemoryAtom, ...]) -> NarrativeState:
    grouped: dict[str, list[MemoryAtom]] = defaultdict(list)
    for atom in atoms:
        if atom.atom_kind != "narrative" or atom.status != "active" or atom.accessibility <= 0.0:
            continue
        theme = narrative_content(atom).theme.strip()
        if theme:
            grouped[theme].append(atom)

    arcs: list[NarrativeArc] = []
    for theme, theme_atoms in grouped.items():
        ordered = sorted(theme_atoms, key=lambda item: (item.updated_at, item.created_at, item.atom_id))
        latest = narrative_content(ordered[-1])
        contents = tuple(narrative_content(atom) for atom in ordered)
        storyline = " / ".join(value for value in (content.storyline.strip() for content in contents) if value)
        episode_ids = _dedupe(episode_id for content in contents for episode_id in content.episode_ids)
        arcs.append(
            NarrativeArc(
                theme=theme,
                storyline=storyline,
                status=latest.status,
                episode_count=len(episode_ids),
                unresolved_threads=_dedupe(value for content in contents for value in content.unresolved_threads),
                role_changes=_dedupe(value for content in contents for value in content.role_changes),
                updated_at=ordered[-1].updated_at,
            )
        )

    ordered_arcs = tuple(sorted(arcs, key=lambda item: (item.updated_at, item.theme)))
    return NarrativeState(
        arcs=ordered_arcs,
        active_themes=tuple(arc.theme for arc in ordered_arcs if arc.status == "active"),
    )


def _semantic_view(atom: MemoryAtom) -> SemanticMemoryItem:
    content = semantic_content(atom)
    return SemanticMemoryItem(
        subject=content.subject,
        attribute=content.attribute,
        value=content.value,
        text=content.text,
    )


def _procedural_view(atom: MemoryAtom) -> ProceduralMemoryItem:
    content = procedural_content(atom)
    return ProceduralMemoryItem(
        rule=content.rule,
        trigger=content.trigger,
        steps=content.steps,
        text=content.text,
        owner=content.owner,
    )


def _episode_view(atom: MemoryAtom) -> EpisodeMemoryItem:
    content = episode_content(atom)
    return EpisodeMemoryItem(
        title=content.title,
        summary=content.summary,
        actors=content.actors,
        setting=content.setting,
        time_span=content.time_span,
        emotion_markers=content.emotion_markers,
        text=content.text,
    )


def _current_state_atoms(atoms: tuple[MemoryAtom, ...]) -> tuple[MemoryAtom, ...]:
    ordered_atoms = tuple(sorted(atoms, key=lambda item: (item.updated_at, item.created_at, item.atom_id)))
    blocked_sources = set(suppressed_source_ids(ordered_atoms))
    hidden_episode_ids = {
        atom.atom_id
        for atom in ordered_atoms
        if atom.atom_kind == "episode"
        and atom.status == "active"
        and any(source_id in blocked_sources for source_id in (atom.atom_id, *atom.source_atom_ids))
    }
    return tuple(
        atom
        for atom in ordered_atoms
        if atom.atom_kind not in {"evidence", "inhibition"}
        and atom.status == "active"
        and atom.accessibility > 0.0
        and not (
            atom.atom_kind == "episode"
            and any(source_id in blocked_sources for source_id in (atom.atom_id, *atom.source_atom_ids))
        )
        and not (
            atom.atom_kind == "narrative"
            and (
                any(source_id in blocked_sources for source_id in atom.source_atom_ids)
                or any(source_id in hidden_episode_ids for source_id in atom.source_atom_ids)
            )
        )
    )


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _payload_text(content: object) -> str:
    text = getattr(content, "text", "")
    return text.strip() if isinstance(text, str) else ""


def _mood_from_markers(markers: tuple[AffectMarker, ...]) -> str:
    has_positive = any(marker.valence > 0.0 for marker in markers)
    has_negative = any(marker.valence < 0.0 for marker in markers)
    if has_positive and has_negative:
        return "mixed"
    if has_positive:
        return "positive"
    if has_negative:
        return "negative"
    return "neutral"
