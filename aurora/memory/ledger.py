"""Memory-field evolution and recall for Aurora."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import numpy.typing as npt

from aurora.memory.store import SQLiteMemoryStore
from aurora.runtime.contracts import ActivatedAtom, ActivatedEdge, MemoryAtom, MemoryEdge, RecallResult, atom_text

EMBEDDING_DIM = 384
FloatVector = npt.NDArray[np.float32]
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_STATE_LIMIT = 12
_RECALL_LIMIT = 8
_VISIBLE_FLOOR = 0.05
_SOLVER_EPSILON = 1e-6


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize mixed CJK and latin text."""
    tokens: list[str] = []
    for chunk in _TOKEN_PATTERN.findall(text.lower()):
        if _CJK_PATTERN.search(chunk):
            tokens.extend(char for char in chunk if _CJK_PATTERN.match(char))
            continue
        tokens.append(chunk)
    return tuple(tokens)


class HashEmbeddingEncoder:
    """Stable hash encoder without external dependencies."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def encode(self, text: str) -> FloatVector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


class MemoryField:
    """Conservative evolution for Aurora's memory field."""

    __slots__ = ("store", "encoder", "iterations")

    def __init__(self, store: SQLiteMemoryStore, encoder: HashEmbeddingEncoder | None = None, iterations: int = 8) -> None:
        self.store = store
        self.encoder = encoder or HashEmbeddingEncoder()
        self.iterations = iterations

    def evolve(self, subject_id: str) -> dict[str, float]:
        atoms = self.store.list_atoms(subject_id)
        edges = self.store.list_edges(subject_id)
        activations = _solve_activation(atoms, edges, iterations=self.iterations)
        updated_at = _latest_timestamp(atoms, edges)
        self.store.replace_activation_cache(subject_id, activations, updated_at=updated_at)
        return activations

    def state_slice(
        self,
        subject_id: str,
        *,
        limit: int = _STATE_LIMIT,
    ) -> tuple[tuple[ActivatedAtom, ...], tuple[ActivatedEdge, ...]]:
        atoms = self.store.list_atoms(subject_id)
        edges = self.store.list_edges(subject_id)
        activations = self.store.list_activation_cache(subject_id)
        if atoms and not activations:
            activations = self.evolve(subject_id)
        return _select_slice(atoms, edges, activations, limit=limit)

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        limit: int = _RECALL_LIMIT,
    ) -> RecallResult:
        atoms = self.store.list_atoms(subject_id)
        if not atoms:
            return RecallResult(subject_id=subject_id, query=query, summary="", atoms=(), edges=())
        edges = self.store.list_edges(subject_id)
        activations = self.store.list_activation_cache(subject_id)
        if not activations:
            activations = self.evolve(subject_id)
        query_bias = _query_bias(query, atoms, activations, self.encoder)
        local = _solve_activation(
            atoms,
            edges,
            seed_bias=query_bias,
            initial_activation=activations,
            iterations=self.iterations,
        )
        selected_atoms, selected_edges = _select_slice(atoms, edges, local, limit=limit)
        return RecallResult(
            subject_id=subject_id,
            query=query,
            summary=_field_summary(selected_atoms, selected_edges, heading="[QUERY_MEMORY_FIELD]"),
            atoms=selected_atoms,
            edges=selected_edges,
        )


def _solve_activation(
    atoms: tuple[MemoryAtom, ...],
    edges: tuple[MemoryEdge, ...],
    *,
    seed_bias: dict[str, float] | None = None,
    initial_activation: dict[str, float] | None = None,
    iterations: int,
) -> dict[str, float]:
    """Solve one bounded memory-field fixed point.

    Kernel axioms mirrored by docs and tests:
    - Every node has an intrinsic activation determined only by local retention and query seed.
    - Positive edges transmit only activation above the intrinsic baseline, so the resting field does not self-amplify.
    - Negative edges transmit inhibitory pressure in proportion to source accessibility.
    - Recall uses the cached field as its initial state, not as a hard lower bound.
    - Relaxation scales with outgoing mass to keep the iteration damped.
    """
    if not atoms:
        return {}

    atom_ids = [atom.atom_id for atom in atoms]
    index = {atom_id: position for position, atom_id in enumerate(atom_ids)}
    retention = np.array(
        [
            max(
                atom.confidence * atom.salience,
                0.0,
            )
            for atom in atoms
        ],
        dtype=np.float32,
    )
    seed = np.zeros(len(atoms), dtype=np.float32)
    if seed_bias:
        for atom_id, value in seed_bias.items():
            if atom_id in index:
                seed[index[atom_id]] = max(seed[index[atom_id]], float(value))
    intrinsic = _intrinsic_activation(retention + seed)
    current = np.array(intrinsic, dtype=np.float32)
    if initial_activation:
        for atom_id, value in initial_activation.items():
            if atom_id in index:
                current[index[atom_id]] = float(value)
    current = np.clip(current, 0.0, 1.0)
    weights, relaxation = _normalized_edge_weights(edges, index)
    for _ in range(iterations):
        target = np.array(intrinsic, dtype=np.float32)
        for source_index, target_index, weight in weights:
            source_signal = current[source_index]
            if weight > 0.0:
                # Positive support can only transmit activation above the local resting field.
                source_signal = max(0.0, float(current[source_index] - intrinsic[source_index]))
            else:
                # Inhibitory pressure depends on how reachable the source currently is.
                source_signal = min(
                    1.0,
                    float(current[source_index] / max(float(intrinsic[source_index]), _SOLVER_EPSILON)),
                )
            target[target_index] += weight * source_signal
        target = np.clip(target, 0.0, 1.0)
        updated = np.clip(current + relaxation * (target - current), 0.0, 1.0)
        if np.allclose(updated, current, atol=1e-4):
            current = updated
            break
        current = updated
    return {atom_id: float(current[index[atom_id]]) for atom_id in atom_ids}


def _normalized_edge_weights(
    edges: tuple[MemoryEdge, ...],
    index: dict[str, int],
) -> tuple[tuple[tuple[int, int, float], ...], float]:
    positive_mass: dict[str, float] = {}
    for edge in edges:
        if edge.source_atom_id not in index or edge.target_atom_id not in index:
            continue
        if edge.influence <= 0.0:
            continue
        positive_mass[edge.source_atom_id] = positive_mass.get(edge.source_atom_id, 0.0) + (
            edge.influence * edge.confidence
        )

    normalized: list[tuple[int, int, float]] = []
    source_mass: dict[int, float] = {}
    for edge in edges:
        source_index = index.get(edge.source_atom_id)
        target_index = index.get(edge.target_atom_id)
        if source_index is None or target_index is None:
            continue
        raw_weight = edge.influence * edge.confidence
        if raw_weight > 0.0:
            scale = positive_mass.get(edge.source_atom_id, 0.0)
            if scale <= 0.0:
                continue
            weight = raw_weight / scale
        else:
            weight = raw_weight
        normalized.append((source_index, target_index, weight))
        source_mass[source_index] = source_mass.get(source_index, 0.0) + abs(weight)

    normalized_mass = max(source_mass.values(), default=0.0)
    relaxation = 1.0 if normalized_mass <= 0.0 else 1.0 / (1.0 + normalized_mass)
    return tuple(normalized), float(relaxation)


def _query_bias(
    query: str,
    atoms: tuple[MemoryAtom, ...],
    activations: dict[str, float],
    encoder: HashEmbeddingEncoder,
) -> dict[str, float]:
    query_tokens = set(tokenize(query))
    query_vector = encoder.encode(query)
    bias: dict[str, float] = {}
    for atom in atoms:
        if atom.atom_kind == "evidence":
            continue
        text = atom_text(atom)
        if not text:
            continue
        lexical, vector_score = _relevance(query_tokens, query_vector, text, encoder)
        score = max(lexical, vector_score) * _query_availability(atom, activations)
        if score <= 0.0:
            continue
        bias[atom.atom_id] = score
    return bias


def _intrinsic_activation(value: FloatVector) -> FloatVector:
    bounded = np.maximum(value, 0.0)
    intrinsic = bounded / (1.0 + bounded)
    return np.clip(intrinsic, _SOLVER_EPSILON, 1.0 - _SOLVER_EPSILON)


def _query_availability(atom: MemoryAtom, activations: dict[str, float]) -> float:
    baseline = float(_intrinsic_activation(np.array([atom.confidence * atom.salience], dtype=np.float32))[0])
    if baseline <= 0.0:
        return 0.0
    current = max(0.0, activations.get(atom.atom_id, 0.0))
    return min(1.0, current / baseline)


def _relevance(
    query_tokens: set[str],
    query_vector: FloatVector,
    content: str,
    encoder: HashEmbeddingEncoder,
) -> tuple[float, float]:
    content_tokens = set(tokenize(content))
    lexical = 0.0
    if query_tokens and content_tokens:
        lexical = len(query_tokens & content_tokens) / len(query_tokens | content_tokens)

    content_vector = encoder.encode(content)
    vector_score = 0.0
    if float(np.linalg.norm(content_vector)) > 0.0 and float(np.linalg.norm(query_vector)) > 0.0:
        vector_score = max(0.0, float(np.dot(query_vector, content_vector)))
    return lexical, vector_score


def _select_slice(
    atoms: tuple[MemoryAtom, ...],
    edges: tuple[MemoryEdge, ...],
    activations: dict[str, float],
    *,
    limit: int,
) -> tuple[tuple[ActivatedAtom, ...], tuple[ActivatedEdge, ...]]:
    ordered_atoms = sorted(
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
    atom_ids = {atom.atom_id for atom in ordered_atoms}
    selected_atoms = tuple(
        ActivatedAtom(
            atom_id=atom.atom_id,
            atom_kind=atom.atom_kind,
            text=atom_text(atom),
            activation=activations.get(atom.atom_id, 0.0),
            confidence=atom.confidence,
            salience=atom.salience,
            created_at=atom.created_at,
        )
        for atom in ordered_atoms
    )
    ranked_edges = sorted(
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
    )[: limit * 2]
    selected_edges = tuple(
        ActivatedEdge(
            source_atom_id=edge.source_atom_id,
            target_atom_id=edge.target_atom_id,
            influence=edge.influence,
            confidence=edge.confidence,
        )
        for edge in ranked_edges
    )
    return selected_atoms, selected_edges


def _field_summary(
    atoms: tuple[ActivatedAtom, ...],
    edges: tuple[ActivatedEdge, ...],
    *,
    heading: str,
) -> str:
    lines = [heading]
    if not atoms:
        lines.append("- none")
        return "\n".join(lines)
    for atom in atoms:
        lines.append(f"- {atom.atom_kind} ({atom.activation:.3f}): {atom.text}")
    if edges:
        lines.append("[LOCAL_CONNECTIONS]")
        for edge in edges[: max(1, min(8, len(edges)))]:
            lines.append(f"- {edge.source_atom_id} -> {edge.target_atom_id}: {edge.influence:.3f}")
    return "\n".join(lines)


def _latest_timestamp(atoms: tuple[MemoryAtom, ...], edges: tuple[MemoryEdge, ...]) -> float:
    latest = 0.0
    for atom in atoms:
        latest = max(latest, atom.created_at)
    for edge in edges:
        latest = max(latest, edge.created_at)
    return latest
