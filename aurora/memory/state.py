"""Activated memory-field projection."""

from __future__ import annotations

from aurora.runtime.contracts import ActivatedAtom, ActivatedEdge, MemoryAtom, MemoryEdge, SubjectMemoryState, atom_text

_STATE_LIMIT = 12
_VISIBLE_FLOOR = 0.05


def project_subject_state(
    *,
    subject_id: str,
    atoms: tuple[MemoryAtom, ...],
    edges: tuple[MemoryEdge, ...],
    activations: dict[str, float],
    limit: int = _STATE_LIMIT,
) -> SubjectMemoryState:
    """Project the current memory field into a read view."""
    selected_atoms = _select_atoms(atoms, activations, limit=limit)
    selected_edges = _select_edges(edges, activations, atom_ids={atom.atom_id for atom in selected_atoms}, limit=limit * 2)
    return SubjectMemoryState(
        subject_id=subject_id,
        summary=_summary(selected_atoms, selected_edges),
        atoms=selected_atoms,
        edges=selected_edges,
    )


def _select_atoms(
    atoms: tuple[MemoryAtom, ...],
    activations: dict[str, float],
    *,
    limit: int,
) -> tuple[ActivatedAtom, ...]:
    ordered = sorted(
        (
            atom
            for atom in atoms
            if atom.atom_kind != "evidence" and activations.get(atom.atom_id, 0.0) > _VISIBLE_FLOOR
        ),
        key=lambda atom: (
            activations.get(atom.atom_id, 0.0),
            atom.salience,
            atom.created_at,
            atom.atom_id,
        ),
        reverse=True,
    )[:limit]
    return tuple(
        ActivatedAtom(
            atom_id=atom.atom_id,
            atom_kind=atom.atom_kind,
            text=atom_text(atom),
            activation=activations.get(atom.atom_id, 0.0),
            confidence=atom.confidence,
            salience=atom.salience,
            created_at=atom.created_at,
        )
        for atom in ordered
    )


def _select_edges(
    edges: tuple[MemoryEdge, ...],
    activations: dict[str, float],
    *,
    atom_ids: set[str],
    limit: int,
) -> tuple[ActivatedEdge, ...]:
    ordered = sorted(
        (
            edge
            for edge in edges
            if edge.source_atom_id in atom_ids and edge.target_atom_id in atom_ids
        ),
        key=lambda edge: (
            abs(edge.influence) * edge.confidence * activations.get(edge.source_atom_id, 0.0),
            edge.created_at,
            edge.edge_id,
        ),
        reverse=True,
    )[:limit]
    return tuple(
        ActivatedEdge(
            source_atom_id=edge.source_atom_id,
            target_atom_id=edge.target_atom_id,
            influence=edge.influence,
            confidence=edge.confidence,
        )
        for edge in ordered
    )


def _summary(atoms: tuple[ActivatedAtom, ...], edges: tuple[ActivatedEdge, ...]) -> str:
    lines = ["[CURRENT_MEMORY_FIELD]"]
    if not atoms:
        lines.append("- none")
        return "\n".join(lines)
    for atom in atoms:
        lines.append(f"- {atom.atom_kind} ({atom.activation:.3f}): {atom.text}")
    if edges:
        lines.append("[CONNECTIONS]")
        for edge in edges[: max(1, min(8, len(edges)))]:
            lines.append(f"- {edge.source_atom_id} -> {edge.target_atom_id}: {edge.influence:.3f}")
    return "\n".join(lines)
