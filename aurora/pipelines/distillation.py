"""Aurora evidence-to-atom distillation."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from typing import cast
from uuid import uuid4

from aurora.llm.provider import LLMProvider
from aurora.runtime.contracts import (
    AffectiveContent,
    AffectMarker,
    AtomContent,
    AtomKind,
    CognitiveContent,
    EpisodeContent,
    EventKind,
    EvidenceContent,
    InhibitionContent,
    MemoryAtom,
    NarrativeContent,
    NarrativeStatus,
    PROCEDURAL_TRIGGER_ASSISTANT_COMMITMENT,
    PROCEDURAL_TRIGGER_PLAN,
    ProceduralContent,
    SEMANTIC_ATTRIBUTE_LOCATION_CURRENT,
    SEMANTIC_ATTRIBUTE_PREFERENCE_LIKE,
    SEMANTIC_ATTRIBUTE_WORK_LOCATION,
    SemanticContent,
    SemanticScope,
    TimeSpan,
    clamp,
    evidence_content,
    procedural_content,
    semantic_content,
)

_USER_COMPILER_PROMPT = (
    "[AURORA_USER_MEMORY_COMPILER]\n"
    "You compile one human subject's current memory update from a user message.\n"
    "Return JSON only with this exact top-level schema:\n"
    "{\n"
    '  "semantic": [{"subject":"user","attribute":"location.current|work.location|preference.like","scope":"self|world","value":"...","text":"..."}],\n'
    '  "procedural": [{"rule":"...","trigger":"plan","steps":["..."],"text":"...","owner":null}],\n'
    '  "cognitive": {"beliefs":["..."],"goals":["..."],"conflicts":["..."],"intentions":["..."],"commitments":["..."]} | null,\n'
    '  "affective": {"mood":"...","valence":0.0,"intensity":0.0,"feelings":["..."],"text":"..."} | null,\n'
    '  "inhibitions": [{"summary":"...","target_summary":"...","target_atom_ids":["existing-atom-id"]}]\n'
    "}\n"
    "Rules:\n"
    "- Only emit memories warranted by the user message.\n"
    "- Cognitive content must be structured summaries, never raw chain-of-thought.\n"
    "- When the user revises or suppresses current memory, emit replacement current-state snapshots when needed.\n"
    "- Inhibitions must reference existing atom ids from the provided memory context; never invent ids.\n"
    "- Use null or [] when nothing should be written."
)
_TURN_COMPILER_PROMPT = (
    "[AURORA_TURN_MEMORY_COMPILER]\n"
    "You compile a completed user/assistant turn into an autobiographical episode and follow-up memory.\n"
    "Return JSON only with this exact top-level schema:\n"
    "{\n"
    '  "episode": {\n'
    '    "title":"...",\n'
    '    "summary":"...",\n'
    '    "actors":["user","aurora"],\n'
    '    "setting":"...",\n'
    '    "emotion_markers":[{"label":"...","intensity":0.0,"valence":0.0}],\n'
    '    "text":"..."\n'
    "  },\n"
    '  "procedural": [{"rule":"...","trigger":"assistant_commitment","steps":["..."],"text":"...","owner":"aurora"}],\n'
    '  "narrative": [{"theme":"...","storyline":"...","status":"active|resolved","unresolved_threads":["..."],"role_changes":["..."],"text":"..."}]\n'
    "}\n"
    "Rules:\n"
    "- Episode must summarize both sides of the turn.\n"
    "- Assistant commitments only when the assistant explicitly committed to future action.\n"
    "- Narrative memory only when the turn advances or resolves an autobiographical arc.\n"
    "- Do not invent evidence ids, internal fields, or chain-of-thought."
)


class MemoryCompiler:
    """Structured compiler from evidence into memory atoms."""

    __slots__ = ("llm",)

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def compile_user_turn(
        self,
        *,
        subject_id: str,
        user_atom: MemoryAtom,
        existing_atoms: tuple[MemoryAtom, ...],
        now_ts: float,
    ) -> tuple[tuple[MemoryAtom, ...], tuple[MemoryAtom, ...]]:
        return distill_user_atoms(
            subject_id=subject_id,
            user_atom=user_atom,
            existing_atoms=existing_atoms,
            llm=self.llm,
            now_ts=now_ts,
        )

    def compile_completed_turn(
        self,
        *,
        subject_id: str,
        user_atom: MemoryAtom,
        assistant_atom: MemoryAtom,
        existing_atoms: tuple[MemoryAtom, ...],
        now_ts: float,
    ) -> tuple[MemoryAtom, tuple[MemoryAtom, ...], tuple[MemoryAtom, ...]]:
        return distill_completed_turn_atoms(
            subject_id=subject_id,
            user_atom=user_atom,
            assistant_atom=assistant_atom,
            existing_atoms=existing_atoms,
            llm=self.llm,
            now_ts=now_ts,
        )


def make_evidence_atom(
    *,
    subject_id: str,
    event_kind: EventKind,
    role: str,
    text: str,
    now_ts: float,
    payload: dict[str, object] | None = None,
) -> MemoryAtom:
    """Create one append-only evidence atom."""
    return MemoryAtom(
        atom_id=f"atom_evidence_{uuid4().hex[:12]}",
        subject_id=subject_id,
        atom_kind="evidence",
        content=EvidenceContent(
            event_kind=event_kind,
            role=role,
            text=text,
            payload=payload or {},
        ),
        status="active",
        confidence=1.0,
        salience=0.35,
        accessibility=1.0,
        created_at=now_ts,
        updated_at=now_ts,
    )


def distill_user_atoms(
    *,
    subject_id: str,
    user_atom: MemoryAtom,
    existing_atoms: tuple[MemoryAtom, ...],
    llm: LLMProvider,
    now_ts: float,
) -> tuple[tuple[MemoryAtom, ...], tuple[MemoryAtom, ...]]:
    """Compile user evidence into current-state memory atoms."""
    compiled = _compile_user_memory(
        llm=llm,
        user_text=evidence_content(user_atom).text,
        existing_atoms=existing_atoms,
    )
    created: list[MemoryAtom] = []
    created.extend(
        _semantic_atoms_from_payload(
            subject_id=subject_id,
            source_atom_ids=(user_atom.atom_id,),
            payload=_object_list(compiled.get("semantic")),
            now_ts=now_ts,
        )
    )
    created.extend(
        _procedural_atoms_from_payload(
            subject_id=subject_id,
            source_atom_ids=(user_atom.atom_id,),
            payload=_object_list(compiled.get("procedural")),
            allowed_triggers={PROCEDURAL_TRIGGER_PLAN},
            default_owner=None,
            now_ts=now_ts,
        )
    )

    cognitive_payload = _optional_object(compiled.get("cognitive"))
    if cognitive_payload is not None:
        created.append(
            _cognitive_atom_from_payload(
                subject_id=subject_id,
                source_atom_ids=(user_atom.atom_id,),
                payload=cognitive_payload,
                now_ts=now_ts,
            )
        )

    affective_payload = _optional_object(compiled.get("affective"))
    if affective_payload is not None:
        created.append(
            _affective_atom_from_payload(
                subject_id=subject_id,
                source_atom_ids=(user_atom.atom_id,),
                payload=affective_payload,
                now_ts=now_ts,
            )
        )

    created.extend(
        _inhibition_atoms_from_payload(
            subject_id=subject_id,
            source_atom_ids=(user_atom.atom_id,),
            payload=_object_list(compiled.get("inhibitions")),
            now_ts=now_ts,
        )
    )
    return _reconcile_atoms(existing_atoms=existing_atoms, created_atoms=tuple(created), now_ts=now_ts)


def distill_completed_turn_atoms(
    *,
    subject_id: str,
    user_atom: MemoryAtom,
    assistant_atom: MemoryAtom,
    existing_atoms: tuple[MemoryAtom, ...],
    llm: LLMProvider,
    now_ts: float,
) -> tuple[MemoryAtom, tuple[MemoryAtom, ...], tuple[MemoryAtom, ...]]:
    """Compile a completed turn into an episode and post-response memory atoms."""
    compiled = _compile_completed_turn(
        llm=llm,
        user_text=evidence_content(user_atom).text,
        assistant_text=evidence_content(assistant_atom).text,
        existing_atoms=existing_atoms,
    )
    episode_atom = _episode_atom_from_payload(
        subject_id=subject_id,
        user_atom=user_atom,
        assistant_atom=assistant_atom,
        payload=_required_object(compiled.get("episode"), "episode"),
        now_ts=now_ts,
    )
    source_atom_ids = (episode_atom.atom_id, user_atom.atom_id, assistant_atom.atom_id)
    created: list[MemoryAtom] = [
        *_procedural_atoms_from_payload(
            subject_id=subject_id,
            source_atom_ids=source_atom_ids,
            payload=_object_list(compiled.get("procedural")),
            allowed_triggers={PROCEDURAL_TRIGGER_ASSISTANT_COMMITMENT},
            default_owner="aurora",
            now_ts=now_ts,
        ),
        *_narrative_atoms_from_payload(
            subject_id=subject_id,
            source_atom_ids=source_atom_ids,
            payload=_object_list(compiled.get("narrative")),
            episode_atom_id=episode_atom.atom_id,
            now_ts=now_ts,
        ),
    ]
    finalized, updated = _reconcile_atoms(existing_atoms=existing_atoms, created_atoms=tuple(created), now_ts=now_ts)
    return episode_atom, finalized, updated


def _compile_user_memory(
    *,
    llm: LLMProvider,
    user_text: str,
    existing_atoms: tuple[MemoryAtom, ...],
) -> dict[str, object]:
    return _json_object(
        llm.complete(
            [
                {"role": "system", "content": _USER_COMPILER_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_text": user_text,
                            "current_memory": _compiler_context(existing_atoms),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
    )


def _compile_completed_turn(
    *,
    llm: LLMProvider,
    user_text: str,
    assistant_text: str,
    existing_atoms: tuple[MemoryAtom, ...],
) -> dict[str, object]:
    return _json_object(
        llm.complete(
            [
                {"role": "system", "content": _TURN_COMPILER_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_text": user_text,
                            "assistant_text": assistant_text,
                            "current_memory": _compiler_context(existing_atoms),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
    )


def _compiler_context(existing_atoms: tuple[MemoryAtom, ...]) -> list[dict[str, object]]:
    return [
        {
            "atom_id": atom.atom_id,
            "atom_kind": atom.atom_kind,
            "status": atom.status,
            "scope": atom.scope,
            "source_atom_ids": list(atom.source_atom_ids),
            "supersedes_atom_id": atom.supersedes_atom_id,
            "inhibits_atom_ids": list(atom.inhibits_atom_ids),
            "content": asdict(atom.content),
        }
        for atom in existing_atoms
        if atom.atom_kind != "evidence"
    ]


def _semantic_atoms_from_payload(
    *,
    subject_id: str,
    source_atom_ids: tuple[str, ...],
    payload: list[dict[str, object]],
    now_ts: float,
) -> tuple[MemoryAtom, ...]:
    atoms: list[MemoryAtom] = []
    for item in payload:
        attribute = _semantic_attribute(item.get("attribute"))
        if attribute is None:
            continue
        value = _string(item.get("value"))
        text = _string(item.get("text"))
        if not value or not text:
            continue
        atoms.append(
            _atom(
                subject_id=subject_id,
                atom_kind="semantic",
                content=SemanticContent(
                    subject=_string(item.get("subject")) or "user",
                    attribute=attribute,
                    value=value,
                    text=text,
                ),
                source_atom_ids=source_atom_ids,
                now_ts=now_ts,
                confidence=0.92,
                salience=0.86,
                scope=_semantic_scope(item.get("scope")),
            )
        )
    return tuple(atoms)


def _procedural_atoms_from_payload(
    *,
    subject_id: str,
    source_atom_ids: tuple[str, ...],
    payload: list[dict[str, object]],
    allowed_triggers: set[str],
    default_owner: str | None,
    now_ts: float,
) -> tuple[MemoryAtom, ...]:
    atoms: list[MemoryAtom] = []
    for item in payload:
        trigger = _procedural_trigger_value(item.get("trigger"))
        if trigger not in allowed_triggers:
            continue
        rule = _string(item.get("rule"))
        text = _string(item.get("text")) or rule
        if not rule or not text:
            continue
        owner = _string(item.get("owner")) or default_owner
        atoms.append(
            _atom(
                subject_id=subject_id,
                atom_kind="procedural",
                content=ProceduralContent(
                    rule=rule,
                    trigger=trigger,
                    steps=_strings(item.get("steps")),
                    text=text,
                    owner=owner or None,
                ),
                source_atom_ids=source_atom_ids,
                now_ts=now_ts,
                confidence=0.88,
                salience=0.84,
            )
        )
    return tuple(atoms)


def _cognitive_atom_from_payload(
    *,
    subject_id: str,
    source_atom_ids: tuple[str, ...],
    payload: dict[str, object],
    now_ts: float,
) -> MemoryAtom:
    return _atom(
        subject_id=subject_id,
        atom_kind="cognitive",
        content=CognitiveContent(
            beliefs=_strings(payload.get("beliefs")),
            goals=_strings(payload.get("goals")),
            conflicts=_strings(payload.get("conflicts")),
            intentions=_strings(payload.get("intentions")),
            commitments=_strings(payload.get("commitments")),
        ),
        source_atom_ids=source_atom_ids,
        now_ts=now_ts,
        confidence=0.84,
        salience=0.82,
    )


def _affective_atom_from_payload(
    *,
    subject_id: str,
    source_atom_ids: tuple[str, ...],
    payload: dict[str, object],
    now_ts: float,
) -> MemoryAtom:
    feelings = _strings(payload.get("feelings"))
    return _atom(
        subject_id=subject_id,
        atom_kind="affective",
        content=AffectiveContent(
            mood=_string(payload.get("mood")) or "neutral",
            valence=_float(payload.get("valence")),
            intensity=clamp(_float(payload.get("intensity"))),
            feelings=feelings,
            text=_string(payload.get("text")) or (_string(payload.get("mood")) or "neutral"),
        ),
        source_atom_ids=source_atom_ids,
        now_ts=now_ts,
        confidence=0.82,
        salience=0.76,
    )


def _inhibition_atoms_from_payload(
    *,
    subject_id: str,
    source_atom_ids: tuple[str, ...],
    payload: list[dict[str, object]],
    now_ts: float,
) -> tuple[MemoryAtom, ...]:
    atoms: list[MemoryAtom] = []
    for item in payload:
        target_atom_ids = _strings(item.get("target_atom_ids"))
        if not target_atom_ids:
            continue
        summary = _string(item.get("summary"))
        target_summary = _string(item.get("target_summary"))
        if not summary:
            continue
        atoms.append(
            replace(
                _atom(
                    subject_id=subject_id,
                    atom_kind="inhibition",
                    content=InhibitionContent(
                        summary=summary,
                        target_summary=target_summary,
                        text=summary,
                    ),
                    source_atom_ids=source_atom_ids,
                    now_ts=now_ts,
                    confidence=0.96,
                    salience=0.92,
                    scope="self",
                ),
                inhibits_atom_ids=target_atom_ids,
            )
        )
    return tuple(atoms)


def _episode_atom_from_payload(
    *,
    subject_id: str,
    user_atom: MemoryAtom,
    assistant_atom: MemoryAtom,
    payload: dict[str, object],
    now_ts: float,
) -> MemoryAtom:
    summary = _string(payload.get("summary"))
    text = _string(payload.get("text")) or summary
    if not summary or not text:
        raise ValueError("episode compiler output must include summary and text")
    markers = tuple(
        AffectMarker(
            label=_string(item.get("label")),
            intensity=clamp(_float(item.get("intensity"))),
            valence=_float(item.get("valence")),
        )
        for item in _object_list(payload.get("emotion_markers"))
        if _string(item.get("label"))
    )
    return MemoryAtom(
        atom_id=f"atom_episode_{uuid4().hex[:12]}",
        subject_id=subject_id,
        atom_kind="episode",
        content=EpisodeContent(
            scene_type="interaction",
            title=_string(payload.get("title")) or "interaction",
            summary=summary,
            actors=_strings(payload.get("actors")) or ("user", "aurora"),
            setting=_string(payload.get("setting")) or "未指明",
            time_span=TimeSpan(start=user_atom.created_at, end=assistant_atom.created_at),
            emotion_markers=markers,
            text=text,
        ),
        status="active",
        confidence=0.98,
        salience=0.74,
        accessibility=1.0,
        source_atom_ids=(user_atom.atom_id, assistant_atom.atom_id),
        created_at=now_ts,
        updated_at=now_ts,
    )


def _narrative_atoms_from_payload(
    *,
    subject_id: str,
    source_atom_ids: tuple[str, ...],
    payload: list[dict[str, object]],
    episode_atom_id: str,
    now_ts: float,
) -> tuple[MemoryAtom, ...]:
    atoms: list[MemoryAtom] = []
    for item in payload:
        theme = _string(item.get("theme"))
        storyline = _string(item.get("storyline"))
        if not theme or not storyline:
            continue
        status = _narrative_status(item.get("status"))
        if status is None:
            continue
        atoms.append(
            _atom(
                subject_id=subject_id,
                atom_kind="narrative",
                content=NarrativeContent(
                    theme=theme,
                    storyline=storyline,
                    status=status,
                    episode_ids=(episode_atom_id,),
                    unresolved_threads=_strings(item.get("unresolved_threads")),
                    role_changes=_strings(item.get("role_changes")),
                    text=_string(item.get("text")) or f"{theme} - {storyline}",
                ),
                source_atom_ids=source_atom_ids,
                now_ts=now_ts,
                confidence=0.80,
                salience=0.86,
            )
        )
    return tuple(atoms)


def _reconcile_atoms(
    *,
    existing_atoms: tuple[MemoryAtom, ...],
    created_atoms: tuple[MemoryAtom, ...],
    now_ts: float,
) -> tuple[tuple[MemoryAtom, ...], tuple[MemoryAtom, ...]]:
    finalized: list[MemoryAtom] = []
    updated: list[MemoryAtom] = []
    current_atoms = list(existing_atoms)

    for atom in created_atoms:
        if atom.atom_kind == "semantic":
            prior = _find_semantic_conflict(current_atoms, atom)
            if prior is not None:
                if prior.content == atom.content:
                    continue
                updated.append(replace(prior, status="superseded", updated_at=now_ts))
                atom = replace(atom, supersedes_atom_id=prior.atom_id)
            elif _has_active_duplicate(current_atoms, atom):
                continue
        elif atom.atom_kind in {"cognitive", "affective"}:
            prior_atoms = _active_atoms_by_kind(current_atoms, atom.atom_kind)
            if prior_atoms and prior_atoms[-1].content == atom.content:
                continue
            if prior_atoms:
                atom = replace(atom, supersedes_atom_id=prior_atoms[-1].atom_id)
            updated.extend(replace(prior, status="superseded", updated_at=now_ts) for prior in prior_atoms)
        elif atom.atom_kind == "procedural":
            if _has_active_duplicate(current_atoms, atom):
                continue
            if _procedural_trigger(atom) == PROCEDURAL_TRIGGER_PLAN:
                prior_atoms = _active_plan_procedural_atoms(current_atoms)
                if prior_atoms:
                    atom = replace(atom, supersedes_atom_id=prior_atoms[-1].atom_id)
                updated.extend(replace(prior, status="superseded", updated_at=now_ts) for prior in prior_atoms)
        elif atom.atom_kind == "inhibition":
            target_ids = tuple(
                target.atom_id
                for target in current_atoms
                if target.atom_id in atom.inhibits_atom_ids and target.status == "active"
            )
            if not target_ids:
                continue
            atom = replace(atom, inhibits_atom_ids=target_ids)
            if _has_active_duplicate(current_atoms, atom):
                continue
            for target in current_atoms:
                if target.atom_id not in target_ids or target.status != "active":
                    continue
                updated.append(replace(target, status="inhibited", accessibility=0.0, updated_at=now_ts))
        elif _has_active_duplicate(current_atoms, atom):
            continue

        finalized.append(atom)
        current_atoms.append(atom)

    return tuple(finalized), tuple(updated)


def _find_semantic_conflict(existing_atoms: list[MemoryAtom], candidate: MemoryAtom) -> MemoryAtom | None:
    candidate_subject, candidate_attribute, _ = _semantic_signature(candidate)
    if not candidate_subject or not candidate_attribute:
        return None
    for atom in reversed(existing_atoms):
        if atom.atom_kind != "semantic" or atom.status != "active":
            continue
        subject, attribute, _ = _semantic_signature(atom)
        if subject == candidate_subject and attribute == candidate_attribute:
            return atom
    return None


def _semantic_signature(atom: MemoryAtom) -> tuple[str, str, str]:
    content = semantic_content(atom)
    return (
        content.subject.strip(),
        content.attribute.strip(),
        content.value.strip(),
    )


def _active_atoms_by_kind(existing_atoms: list[MemoryAtom], atom_kind: AtomKind) -> tuple[MemoryAtom, ...]:
    return tuple(
        atom
        for atom in existing_atoms
        if atom.atom_kind == atom_kind and atom.status == "active"
    )


def _active_plan_procedural_atoms(existing_atoms: list[MemoryAtom]) -> tuple[MemoryAtom, ...]:
    return tuple(
        atom
        for atom in existing_atoms
        if atom.atom_kind == "procedural"
        and atom.status == "active"
        and _procedural_trigger(atom) == PROCEDURAL_TRIGGER_PLAN
    )


def _procedural_trigger(atom: MemoryAtom) -> str:
    return procedural_content(atom).trigger.strip()


def _has_active_duplicate(existing_atoms: list[MemoryAtom], candidate: MemoryAtom) -> bool:
    return any(
        atom.atom_kind == candidate.atom_kind
        and atom.status == "active"
        and atom.scope == candidate.scope
        and atom.content == candidate.content
        and atom.inhibits_atom_ids == candidate.inhibits_atom_ids
        for atom in existing_atoms
    )


def _atom(
    *,
    subject_id: str,
    atom_kind: AtomKind,
    content: AtomContent,
    source_atom_ids: tuple[str, ...],
    now_ts: float,
    confidence: float,
    salience: float,
    scope: SemanticScope | None = None,
) -> MemoryAtom:
    return MemoryAtom(
        atom_id=f"atom_{atom_kind}_{uuid4().hex[:12]}",
        subject_id=subject_id,
        atom_kind=atom_kind,
        content=content,
        status="active",
        confidence=clamp(confidence),
        salience=clamp(salience),
        accessibility=1.0,
        scope=scope,
        source_atom_ids=source_atom_ids,
        created_at=now_ts,
        updated_at=now_ts,
    )


def _json_object(raw: str) -> dict[str, object]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("compiler output must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _required_object(value: object, field_name: str) -> dict[str, object]:
    obj = _optional_object(value)
    if obj is None:
        raise ValueError(f"{field_name} must be an object")
    return obj


def _optional_object(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("compiler object fields must be JSON objects")
    return {str(key): item for key, item in value.items()}


def _object_list(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("compiler list fields must be JSON arrays")
    return [
        {str(key): item for key, item in item_value.items()}
        for item_value in value
        if isinstance(item_value, dict)
    ]


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(text for text in (_string(item) for item in value) if text)


def _float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _semantic_scope(value: object) -> SemanticScope | None:
    scope = _string(value)
    if scope in {"self", "world"}:
        return cast(SemanticScope, scope)
    return "self"


def _semantic_attribute(value: object) -> str | None:
    attribute = _string(value)
    if attribute in {
        SEMANTIC_ATTRIBUTE_LOCATION_CURRENT,
        SEMANTIC_ATTRIBUTE_WORK_LOCATION,
        SEMANTIC_ATTRIBUTE_PREFERENCE_LIKE,
    }:
        return attribute
    return None


def _narrative_status(value: object) -> NarrativeStatus | None:
    status = _string(value)
    if status in {"active", "resolved"}:
        return cast(NarrativeStatus, status)
    return None


def _procedural_trigger_value(value: object) -> str:
    trigger = _string(value)
    if trigger == PROCEDURAL_TRIGGER_ASSISTANT_COMMITMENT:
        return PROCEDURAL_TRIGGER_ASSISTANT_COMMITMENT
    return PROCEDURAL_TRIGGER_PLAN
