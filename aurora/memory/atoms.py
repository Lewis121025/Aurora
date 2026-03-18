"""Memory atom helpers and derived views."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from aurora.runtime.contracts import FactKind, FactRecord, FactStatus, MemoryAtom, OpenLoop, RelationField, clamp

_DEFAULT_FACT_KIND: FactKind = "profile"


def atom_text(atom: MemoryAtom) -> str:
    """Return the text used for scoring and prompt projection."""
    payload = atom.payload
    if atom.atom_type == "fact":
        return str(payload.get("content", "")).strip()
    if atom.atom_type == "rule":
        return str(payload.get("text", "")).strip()
    if atom.atom_type == "lexicon":
        terms = payload.get("terms", [])
        if isinstance(terms, list):
            return " ".join(str(item).strip() for item in terms if str(item).strip())
        return str(payload.get("text", "")).strip()
    if atom.atom_type == "loop":
        return str(payload.get("summary", "")).strip()
    if atom.atom_type == "revision":
        return str(payload.get("content", "")).strip() or str(payload.get("summary", "")).strip()
    if atom.atom_type == "forget":
        return str(payload.get("matcher", "")).strip() or str(payload.get("summary", "")).strip()
    return ""


def effective_atoms(atoms: Iterable[MemoryAtom]) -> tuple[MemoryAtom, ...]:
    """Derive lifecycle-effective atom state from the append-only atom log."""
    atom_list = tuple(sorted(atoms, key=lambda item: (item.updated_at, item.created_at, item.atom_id)))
    superseded_ids = {
        atom.supersedes_atom_id
        for atom in atom_list
        if atom.atom_type == "fact" and atom.supersedes_atom_id
    }
    hidden_ids = {
        affected_id
        for atom in atom_list
        if atom.atom_type == "forget" and atom.status == "active" and atom.visibility > 0.0
        for affected_id in atom.affects_atom_ids
    }

    derived: list[MemoryAtom] = []
    for atom in atom_list:
        status = atom.status
        visibility = atom.visibility
        if atom.atom_id in hidden_ids and atom.atom_type != "forget":
            status = "hidden"
            visibility = 0.0
        elif atom.atom_type == "fact" and atom.atom_id in superseded_ids:
            status = "superseded"
            visibility = min(visibility, 0.12)
        derived.append(
            MemoryAtom(
                atom_id=atom.atom_id,
                relation_id=atom.relation_id,
                atom_type=atom.atom_type,
                payload=atom.payload,
                status=status,
                confidence=atom.confidence,
                salience=atom.salience,
                visibility=visibility,
                evidence_event_ids=atom.evidence_event_ids,
                affects_atom_ids=atom.affects_atom_ids,
                supersedes_atom_id=atom.supersedes_atom_id,
                created_at=atom.created_at,
                updated_at=atom.updated_at,
            )
        )
    return tuple(sorted(derived, key=lambda item: (item.updated_at, item.created_at), reverse=True))


def active_atoms(atoms: Iterable[MemoryAtom], *, min_visibility: float = 0.05) -> tuple[MemoryAtom, ...]:
    """Return atoms that still participate in default continuity."""
    effective = effective_atoms(atoms)
    return tuple(
        atom
        for atom in effective
        if atom.status == "active" and atom.visibility >= min_visibility
    )


def derive_open_loops(atoms: Iterable[MemoryAtom]) -> tuple[OpenLoop, ...]:
    """Derive open loop view from loop atoms."""
    loop_map: dict[str, OpenLoop] = {}
    for atom in sorted(effective_atoms(atoms), key=lambda item: (item.updated_at, item.created_at)):
        if atom.atom_type != "loop":
            continue
        if atom.status == "hidden" or atom.visibility <= 0.0:
            continue
        payload = atom.payload
        loop_ref = str(payload.get("loop_ref") or atom.atom_id)
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            continue
        state = str(payload.get("state", "active")).strip().lower()
        loop_map[loop_ref] = OpenLoop(
            loop_id=loop_ref,
            relation_id=atom.relation_id,
            loop_type=str(payload.get("loop_type", "unfinished_thread")),  # type: ignore[arg-type]
            status="resolved" if state == "resolved" or atom.status == "resolved" else "active",
            summary=summary,
            urgency=clamp(float(payload.get("urgency", atom.salience or 0.5))),
            opened_at=float(payload.get("opened_at", atom.created_at)),
            updated_at=atom.updated_at,
            evidence_refs=atom.evidence_event_ids,
        )
    ordered = sorted(loop_map.values(), key=lambda item: item.updated_at, reverse=True)
    return tuple(ordered)


def derive_facts(atoms: Iterable[MemoryAtom]) -> tuple[FactRecord, ...]:
    """Derive fact view from visible fact atoms."""
    facts: list[FactRecord] = []
    for atom in effective_atoms(atoms):
        if atom.atom_type != "fact":
            continue
        content = str(atom.payload.get("content", "")).strip()
        if not content:
            continue
        facts.append(
            FactRecord(
                fact_id=atom.atom_id,
                relation_id=atom.relation_id,
                content=content,
                fact_kind=_coerce_fact_kind(atom.payload.get("fact_kind")),
                document_date=atom.updated_at,
                event_date=float(atom.payload.get("event_date", atom.created_at)),
                status=_fact_status(atom.status),
                supersedes=atom.supersedes_atom_id,
                confidence=atom.confidence,
                visibility=atom.visibility,
                evidence_refs=atom.evidence_event_ids,
            )
        )
    return tuple(facts)


def derive_relation_field(
    relation_id: str,
    atoms: Iterable[MemoryAtom],
) -> RelationField:
    """Reduce current visible atoms into a hidden relation field."""
    atom_list = effective_atoms(atoms)
    visible = active_atoms(atom_list)
    rules = _collect_unique(
        str(atom.payload.get("text", "")).strip()
        for atom in visible
        if atom.atom_type == "rule"
    )
    lexicon = _collect_unique(
        term
        for atom in visible
        if atom.atom_type == "lexicon"
        for term in _lexicon_terms(atom)
    )
    loops = derive_open_loops(atom_list)
    facts = derive_facts(atom_list)
    revisions = [atom for atom in atom_list if atom.atom_type == "revision" and atom.visibility >= 0.05]
    forgets = [atom for atom in atom_list if atom.atom_type == "forget" and atom.visibility >= 0.05]

    trust = 0.35 + min(0.22, 0.03 * len(rules) + 0.015 * len([fact for fact in facts if fact.status == "active"]))
    trust += min(0.08, 0.02 * len(forgets))
    distance = 0.65 - min(0.24, 0.04 * len(rules) + 0.02 * len(lexicon))
    warmth = 0.30 + min(0.20, 0.02 * len(lexicon) + 0.03 * len([loop for loop in loops if loop.loop_type == "commitment" and loop.status == "active"]))
    tension = min(
        0.90,
        0.12 * len([loop for loop in loops if loop.loop_type == "contradiction" and loop.status == "active"])
        + 0.06 * len(revisions),
    )
    repair_debt = min(0.80, 0.08 * len(revisions))

    last_compiled_at = max((atom.updated_at for atom in atom_list), default=0.0)
    return RelationField(
        relation_id=relation_id,
        trust=clamp(trust),
        distance=clamp(distance),
        warmth=clamp(warmth),
        tension=clamp(tension),
        repair_debt=clamp(repair_debt),
        shared_lexicon=list(lexicon),
        interaction_rules=list(rules),
        last_compiled_at=last_compiled_at,
    )


def _coerce_fact_kind(value: object) -> FactKind:
    if value in {"profile", "preference", "current_state", "biographical"}:
        return value  # type: ignore[return-value]
    return _DEFAULT_FACT_KIND


def _fact_status(status: str) -> FactStatus:
    if status in {"active", "superseded", "hidden"}:
        return cast(FactStatus, status)
    return "active"


def _lexicon_terms(atom: MemoryAtom) -> tuple[str, ...]:
    terms = atom.payload.get("terms", [])
    if isinstance(terms, list):
        return tuple(str(item).strip() for item in terms if str(item).strip())
    text = str(atom.payload.get("text", "")).strip()
    return (text,) if text else ()


def _collect_unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return tuple(values)
