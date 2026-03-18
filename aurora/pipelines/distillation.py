"""Aurora text-to-memory-field distillation."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import cast
from uuid import uuid4

from aurora.llm.provider import LLMProvider
from aurora.runtime.contracts import (
    AtomKind,
    EpisodeContent,
    EventKind,
    EvidenceContent,
    MemoryContent,
    MemoryAtom,
    MemoryEdge,
    TimeSpan,
    evidence_content,
)

_USER_ALLOWED_KINDS: frozenset[AtomKind] = frozenset({"memory", "inhibition"})
_TURN_ALLOWED_KINDS: frozenset[AtomKind] = frozenset({"memory", "episode", "inhibition"})

_USER_FIELD_PROMPT = (
    "[AURORA_USER_FIELD_COMPILER]\n"
    "Convert one user message into memory field nodes and signed weighted edges.\n"
    "Return JSON only with this exact schema:\n"
    "{\n"
    '  "atoms": [\n'
    '    {"kind":"memory|inhibition","text":"...","confidence":0.0,"salience":0.0,"referents":["..."]}\n'
    "  ],\n"
    '  "edges": [\n'
    '    {"source":"new:0|atom-id","target":"new:0|atom-id","influence":-1.0,"confidence":0.0}\n'
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- Emit only durable memory traces or inhibition traces.\n"
    "- Use positive influence for reinforcement/association and negative influence for inhibition/tension.\n"
    "- Only reference existing atom ids that appear in current_field.\n"
    "- Never create edges that touch evidence atoms.\n"
    "- If nothing durable should be written, return empty arrays.\n"
)

_TURN_FIELD_PROMPT = (
    "[AURORA_TURN_FIELD_COMPILER]\n"
    "Convert a completed user/assistant turn into memory field nodes and signed weighted edges.\n"
    "Return JSON only with this exact schema:\n"
    "{\n"
    '  "atoms": [\n'
    '    {"kind":"episode|memory|inhibition","text":"...","confidence":0.0,"salience":0.0,"referents":["..."]}\n'
    "  ],\n"
    '  "edges": [\n'
    '    {"source":"new:0|atom-id","target":"new:0|atom-id","influence":-1.0,"confidence":0.0}\n'
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- Episode nodes should capture both sides of the turn in one text.\n"
    "- Non-episode durable traces must use kind=memory or kind=inhibition.\n"
    "- Only emit future-facing traces when the assistant clearly committed to something durable.\n"
    "- Never create edges that touch evidence atoms.\n"
    "- If nothing durable should be written, return empty arrays.\n"
)


class MemoryCompiler:
    """Structured compiler from evidence into memory-field atoms and edges.

    The compiler is a proposal layer. Invalid kinds, ranges, references, and
    evidence-touching edges are dropped instead of normalized into new truth.
    """

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
    ) -> tuple[tuple[MemoryAtom, ...], tuple[MemoryEdge, ...]]:
        compiled = _compile_user_field(
            llm=self.llm,
            user_text=evidence_content(user_atom).text,
            existing_atoms=existing_atoms,
        )
        created_atoms = _atoms_from_payload(
            subject_id=subject_id,
            source_atom_ids=(user_atom.atom_id,),
            payload=_object_list(compiled.get("atoms")),
            now_ts=now_ts,
            allowed_kinds=_USER_ALLOWED_KINDS,
        )
        created_edges = _edges_from_payload(
            subject_id=subject_id,
            payload=_object_list(compiled.get("edges")),
            created_atoms=created_atoms,
            existing_atoms=existing_atoms,
            now_ts=now_ts,
        )
        return created_atoms, created_edges

    def compile_completed_turn(
        self,
        *,
        subject_id: str,
        user_atom: MemoryAtom,
        assistant_atom: MemoryAtom,
        existing_atoms: tuple[MemoryAtom, ...],
        now_ts: float,
    ) -> tuple[tuple[MemoryAtom, ...], tuple[MemoryEdge, ...]]:
        compiled = _compile_turn_field(
            llm=self.llm,
            user_text=evidence_content(user_atom).text,
            assistant_text=evidence_content(assistant_atom).text,
            existing_atoms=existing_atoms,
        )
        created_atoms = _atoms_from_payload(
            subject_id=subject_id,
            source_atom_ids=(user_atom.atom_id, assistant_atom.atom_id),
            payload=_object_list(compiled.get("atoms")),
            now_ts=now_ts,
            episode_span=TimeSpan(start=user_atom.created_at, end=assistant_atom.created_at),
            allowed_kinds=_TURN_ALLOWED_KINDS,
        )
        created_edges = _edges_from_payload(
            subject_id=subject_id,
            payload=_object_list(compiled.get("edges")),
            created_atoms=created_atoms,
            existing_atoms=existing_atoms,
            now_ts=now_ts,
        )
        return created_atoms, created_edges


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
        confidence=1.0,
        salience=0.2,
        source_atom_ids=(),
        created_at=now_ts,
    )


def _compile_user_field(
    *,
    llm: LLMProvider,
    user_text: str,
    existing_atoms: tuple[MemoryAtom, ...],
) -> dict[str, object]:
    return _json_object(
        llm.complete(
            [
                {"role": "system", "content": _USER_FIELD_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_text": user_text,
                            "current_field": _compiler_context(existing_atoms),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
    )


def _compile_turn_field(
    *,
    llm: LLMProvider,
    user_text: str,
    assistant_text: str,
    existing_atoms: tuple[MemoryAtom, ...],
) -> dict[str, object]:
    return _json_object(
        llm.complete(
            [
                {"role": "system", "content": _TURN_FIELD_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_text": user_text,
                            "assistant_text": assistant_text,
                            "current_field": _compiler_context(existing_atoms),
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
            "kind": atom.atom_kind,
            "text": atom.content.text if hasattr(atom.content, "text") else "",
            "confidence": atom.confidence,
            "salience": atom.salience,
            "source_atom_ids": list(atom.source_atom_ids),
            "content": asdict(atom.content),
            "created_at": atom.created_at,
        }
        for atom in existing_atoms
    ]


def _atoms_from_payload(
    *,
    subject_id: str,
    source_atom_ids: tuple[str, ...],
    payload: list[dict[str, object]],
    now_ts: float,
    allowed_kinds: frozenset[AtomKind],
    episode_span: TimeSpan | None = None,
) -> tuple[MemoryAtom, ...]:
    atoms: list[MemoryAtom] = []
    for item in payload:
        atom_kind = _atom_kind(item.get("kind"))
        text = _string(item.get("text"))
        if atom_kind is None or atom_kind not in allowed_kinds or not text:
            continue
        confidence = _required_float(item.get("confidence"))
        salience = _required_float(item.get("salience"))
        if confidence is None or salience is None:
            continue
        referents = _strings(item.get("referents"))
        content: EpisodeContent | MemoryContent
        if atom_kind == "episode":
            if episode_span is None:
                continue
            content = EpisodeContent(text=text, time_span=episode_span, referents=referents)
        else:
            content = MemoryContent(text=text, referents=referents)
        if not _unit_interval(confidence) or not _unit_interval(salience):
            continue
        atoms.append(
            MemoryAtom(
                atom_id=f"atom_{atom_kind}_{uuid4().hex[:12]}",
                subject_id=subject_id,
                atom_kind=atom_kind,
                content=content,
                confidence=confidence,
                salience=salience,
                source_atom_ids=source_atom_ids,
                created_at=now_ts,
            )
        )
    return tuple(atoms)


def _edges_from_payload(
    *,
    subject_id: str,
    payload: list[dict[str, object]],
    created_atoms: tuple[MemoryAtom, ...],
    existing_atoms: tuple[MemoryAtom, ...],
    now_ts: float,
) -> tuple[MemoryEdge, ...]:
    atoms_by_reference = {
        **{f"new:{index}": atom for index, atom in enumerate(created_atoms)},
        **{atom.atom_id: atom for atom in existing_atoms},
        **{atom.atom_id: atom for atom in created_atoms},
    }
    edges: list[MemoryEdge] = []
    for item in payload:
        source_reference = _string(item.get("source"))
        target_reference = _string(item.get("target"))
        source_atom = atoms_by_reference.get(source_reference)
        target_atom = atoms_by_reference.get(target_reference)
        if source_atom is None or target_atom is None or source_atom.atom_id == target_atom.atom_id:
            continue
        if source_atom.atom_kind == "evidence" or target_atom.atom_kind == "evidence":
            continue
        influence = _required_float(item.get("influence"))
        confidence = _required_float(item.get("confidence"))
        if influence is None or confidence is None:
            continue
        if not _signed_unit_interval(influence) or not _unit_interval(confidence):
            continue
        edges.append(
            MemoryEdge(
                edge_id=f"edge_{uuid4().hex[:12]}",
                subject_id=subject_id,
                source_atom_id=source_atom.atom_id,
                target_atom_id=target_atom.atom_id,
                influence=influence,
                confidence=confidence,
                created_at=now_ts,
            )
        )
    return tuple(edges)


def _json_object(raw: str) -> dict[str, object]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("compiler must return a JSON object")
    return {str(key): value for key, value in data.items()}


def _object_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [
        {str(key): item for key, item in value_item.items()}
        for value_item in value
        if isinstance(value_item, dict)
    ]


def _atom_kind(value: object) -> AtomKind | None:
    text = _string(value)
    if text not in {"memory", "episode", "inhibition"}:
        return None
    return cast(AtomKind, text)


def _string(value: object) -> str:
    return str(value).strip()


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _required_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _unit_interval(value: float) -> bool:
    return 0.0 <= value <= 1.0


def _signed_unit_interval(value: float) -> bool:
    return -1.0 <= value <= 1.0
