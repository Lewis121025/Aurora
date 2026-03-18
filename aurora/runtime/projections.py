"""Aurora runtime projection helpers for the memory field."""

from __future__ import annotations

from collections.abc import Iterable

from aurora.runtime.contracts import ActivatedAtom, ActivatedEdge, RecallResult, SubjectMemoryState

_MAINLINE_LIMIT = 4
_QUERY_LIMIT = 4
_RECENT_LIMIT = 3
_TENSION_LIMIT = 3
_COMMITMENT_LIMIT = 3


def build_memory_brief(
    state: SubjectMemoryState,
    recall: RecallResult | None = None,
) -> str:
    """Build a query-shaped memory brief for response generation."""
    recall_atoms = () if recall is None else recall.atoms
    recall_edges = () if recall is None else recall.edges
    current_mainline = _current_mainline(state.atoms)
    query_relevant = _query_relevant(recall_atoms)
    recent_changes = _recent_changes(state.atoms, recall_atoms)
    active_tensions = _active_tensions(state.atoms, recall_atoms, state.edges, recall_edges)
    ongoing_commitments = _ongoing_commitments(state.atoms, recall_atoms)
    return _render_brief(
        current_mainline=current_mainline,
        query_relevant=query_relevant,
        recent_changes=recent_changes,
        active_tensions=active_tensions,
        ongoing_commitments=ongoing_commitments,
    )


def _current_mainline(atoms: tuple[ActivatedAtom, ...]) -> tuple[ActivatedAtom, ...]:
    selected = _dedupe_atoms(
        atom
        for atom in atoms
        if atom.atom_kind == "memory" and not _is_commitment(atom.text)
    )
    if selected:
        return selected[:_MAINLINE_LIMIT]
    return _dedupe_atoms(atom for atom in atoms if atom.atom_kind == "episode")[:_MAINLINE_LIMIT]


def _query_relevant(atoms: tuple[ActivatedAtom, ...]) -> tuple[ActivatedAtom, ...]:
    return _dedupe_atoms(
        atom
        for atom in atoms
        if atom.atom_kind != "inhibition" and not _is_commitment(atom.text)
    )[:_QUERY_LIMIT]


def _recent_changes(
    state_atoms: tuple[ActivatedAtom, ...],
    recall_atoms: tuple[ActivatedAtom, ...],
) -> tuple[ActivatedAtom, ...]:
    merged = _dedupe_atoms(
        atom
        for atom in (*recall_atoms, *state_atoms)
        if atom.atom_kind == "episode"
    )
    ordered = sorted(
        merged,
        key=lambda atom: (atom.created_at, atom.activation, atom.atom_id),
        reverse=True,
    )
    return tuple(ordered[:_RECENT_LIMIT])


def _active_tensions(
    state_atoms: tuple[ActivatedAtom, ...],
    recall_atoms: tuple[ActivatedAtom, ...],
    state_edges: tuple[ActivatedEdge, ...],
    recall_edges: tuple[ActivatedEdge, ...],
) -> tuple[str, ...]:
    atoms_by_id = {atom.atom_id: atom for atom in _dedupe_atoms((*recall_atoms, *state_atoms))}
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    ordered_edges = sorted(
        (edge for edge in (*recall_edges, *state_edges) if edge.influence < 0.0),
        key=lambda edge: (
            abs(edge.influence) * edge.confidence,
            edge.confidence,
            edge.source_atom_id,
            edge.target_atom_id,
        ),
        reverse=True,
    )
    for edge in ordered_edges:
        source = atoms_by_id.get(edge.source_atom_id)
        target = atoms_by_id.get(edge.target_atom_id)
        if source is None or target is None:
            continue
        key = (edge.source_atom_id, edge.target_atom_id)
        if key in seen:
            continue
        seen.add(key)
        if source.atom_kind == "inhibition":
            lines.append(f"{target.text} 受到 {source.text} 的抑制")
        else:
            lines.append(f"{source.text} 与 {target.text} 存在张力")
        if len(lines) >= _TENSION_LIMIT:
            break
    return tuple(lines)


def _ongoing_commitments(
    state_atoms: tuple[ActivatedAtom, ...],
    recall_atoms: tuple[ActivatedAtom, ...],
) -> tuple[ActivatedAtom, ...]:
    return _dedupe_atoms(
        atom
        for atom in (*recall_atoms, *state_atoms)
        if atom.atom_kind == "memory" and _is_commitment(atom.text)
    )[:_COMMITMENT_LIMIT]


def _render_brief(
    *,
    current_mainline: tuple[ActivatedAtom, ...],
    query_relevant: tuple[ActivatedAtom, ...],
    recent_changes: tuple[ActivatedAtom, ...],
    active_tensions: tuple[str, ...],
    ongoing_commitments: tuple[ActivatedAtom, ...],
) -> str:
    lines = ["[MEMORY_BRIEF]"]
    _append_atom_section(lines, "current_mainline", current_mainline)
    _append_atom_section(lines, "query_relevant", query_relevant)
    _append_atom_section(lines, "recent_changes", recent_changes)
    _append_text_section(lines, "active_tensions", active_tensions)
    _append_atom_section(lines, "ongoing_commitments", ongoing_commitments)
    return "\n".join(lines)


def _append_atom_section(lines: list[str], heading: str, atoms: tuple[ActivatedAtom, ...]) -> None:
    lines.append(f"{heading}:")
    if not atoms:
        lines.append("- none")
        return
    for atom in atoms:
        lines.append(f"- {atom.text}")


def _append_text_section(lines: list[str], heading: str, items: tuple[str, ...]) -> None:
    lines.append(f"{heading}:")
    if not items:
        lines.append("- none")
        return
    for item in items:
        lines.append(f"- {item}")


def _dedupe_atoms(atoms: Iterable[ActivatedAtom]) -> tuple[ActivatedAtom, ...]:
    seen: set[str] = set()
    ordered: list[ActivatedAtom] = []
    for atom in atoms:
        if atom.atom_id in seen:
            continue
        seen.add(atom.atom_id)
        ordered.append(atom)
    return tuple(ordered)


def _is_commitment(text: str) -> bool:
    return text.startswith("Aurora承诺：")
