"""Expert-scoped evolution and recall for Aurora."""

from __future__ import annotations

import hashlib
import re
import time
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from aurora.memory.experts import (
    COMPILE_ATOMS_PER_EXPERT,
    EXPERT_SPLIT_ATOMS,
    EXPERT_SPLIT_EDGES,
    GLOBAL_COMMITMENT_LIMIT,
    GLOBAL_EPISODE_LIMIT,
    GLOBAL_EXPERT_ID,
    GLOBAL_MAINLINE_LIMIT,
    GLOBAL_TENSION_LIMIT,
    NEW_EXPERT_SCORE_THRESHOLD,
    ROUTED_TOPIC_EXPERTS,
    ExpertRecord,
    ExpertRoute,
    centroid_from_vectors,
    centroid_similarity,
    split_embeddings,
)
from aurora.memory.state import project_subject_state
from aurora.memory.store import SQLiteMemoryStore
from aurora.runtime.contracts import (
    ActivatedAtom,
    ActivatedEdge,
    MemoryAtom,
    MemoryEdge,
    RecallResult,
    SubjectMemoryState,
    atom_text,
)
from aurora.runtime.projections import build_memory_brief

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
    """Sparse expert-scoped memory field."""

    __slots__ = ("store", "encoder", "iterations")

    def __init__(self, store: SQLiteMemoryStore, encoder: HashEmbeddingEncoder | None = None, iterations: int = 8) -> None:
        self.store = store
        self.encoder = encoder or HashEmbeddingEncoder()
        self.iterations = iterations
        self._bootstrap_existing_subjects()

    def evolve(self, subject_id: str) -> dict[str, float]:
        self.ensure_subject_experts(subject_id)
        topic_ids = tuple(expert.expert_id for expert in self.store.list_experts(subject_id, expert_kind="topic"))
        self._refresh_topics(subject_id, topic_ids, now_ts=time.time())
        self._refresh_global(subject_id, now_ts=time.time())
        return self.store.list_activation_cache(subject_id)

    def state(self, subject_id: str, *, limit: int = _STATE_LIMIT) -> SubjectMemoryState:
        self.ensure_subject_experts(subject_id)
        global_id = self._global_expert(subject_id).expert_id
        atoms, edges, activations = self._load_expert_scope(subject_id, (global_id,))
        if atoms and not activations:
            self.evolve(subject_id)
            atoms, edges, activations = self._load_expert_scope(subject_id, (global_id,))
        return project_subject_state(
            subject_id=subject_id,
            atoms=atoms,
            edges=edges,
            activations=activations,
            limit=limit,
        )

    def build_compile_context(
        self,
        subject_id: str,
        query: str,
        route: ExpertRoute,
    ) -> tuple[tuple[MemoryAtom, ...], str]:
        state = self.state(subject_id)
        ordered: list[MemoryAtom] = []
        seen: set[str] = set()
        for expert_id in route.expert_ids:
            atoms = self.store.list_expert_atoms(subject_id, expert_id)
            activations = self.store.list_expert_activation_cache(subject_id, expert_id)
            if not atoms or not activations:
                continue
            selected_atoms = _compile_atoms(
                atoms,
                self._edges_for_atoms(subject_id, atoms),
                activations,
                limit=COMPILE_ATOMS_PER_EXPERT,
            )
            for activated in selected_atoms:
                if activated.atom_id in seen:
                    continue
                seen.add(activated.atom_id)
                atom = self.store.get_atom(activated.atom_id)
                if atom is not None:
                    ordered.append(atom)
        return tuple(ordered), build_memory_brief(state, heading="[STABLE_FIELD_SUMMARY]")

    def route(
        self,
        subject_id: str,
        query: str,
        *,
        now_ts: float | None = None,
        touch: bool,
    ) -> ExpertRoute:
        self.ensure_subject_experts(subject_id)
        timestamp = time.time() if now_ts is None else now_ts
        global_id = self._global_expert(subject_id).expert_id
        topics = self.store.list_experts(subject_id, expert_kind="topic")
        if len(topics) == 1 and topics[0].atom_count == 0:
            route = ExpertRoute(subject_id=subject_id, global_expert_id=global_id, topic_expert_ids=(topics[0].expert_id,))
            if touch:
                self.store.touch_experts(subject_id, route.expert_ids, activated_at=timestamp)
            return route

        query_tokens = set(tokenize(query))
        query_vector = self.encoder.encode(query)
        token_scores = self.store.expert_token_overlap(subject_id, query_tokens, expert_kind="topic")
        latest_activation = max((expert.last_activated_at for expert in topics), default=0.0)
        scored = sorted(
            (
                (
                    0.55 * min(1.0, token_scores.get(expert.expert_id, 0.0))
                    + 0.35 * centroid_similarity(query_vector, expert.centroid)
                    + 0.10 * (1.0 if latest_activation > 0.0 and expert.last_activated_at >= latest_activation else 0.0),
                    expert,
                )
                for expert in topics
            ),
            key=lambda item: (item[0], item[1].updated_at, item[1].expert_id),
            reverse=True,
        )
        topic_ids = tuple(expert.expert_id for _, expert in scored[:ROUTED_TOPIC_EXPERTS])
        route = ExpertRoute(subject_id=subject_id, global_expert_id=global_id, topic_expert_ids=topic_ids)
        if touch:
            self.store.touch_experts(subject_id, route.expert_ids, activated_at=timestamp)
        return route

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        limit: int = _RECALL_LIMIT,
        route: ExpertRoute | None = None,
    ) -> RecallResult:
        self.ensure_subject_experts(subject_id)
        scoped_route = route or self.route(subject_id, query, touch=False)
        atoms, edges, activations = self._load_expert_scope(subject_id, scoped_route.expert_ids)
        if not atoms:
            return RecallResult(subject_id=subject_id, query=query, summary="", atoms=(), edges=())
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

    def integrate(
        self,
        subject_id: str,
        route: ExpertRoute,
        atoms: tuple[MemoryAtom, ...],
        edges: tuple[MemoryEdge, ...],
        *,
        now_ts: float,
    ) -> None:
        if not atoms and not edges:
            return
        assignments = self._assign_primary_topics(subject_id, route, atoms, edges, now_ts=now_ts)
        for atom in atoms:
            primary_expert_id = assignments[atom.atom_id]
            self.store.set_atom_memberships(subject_id, atom.atom_id, primary_expert_id=primary_expert_id)
        affected_topics = tuple(dict.fromkeys(assignments.values()))
        affected_topics = self._split_topics(subject_id, affected_topics, now_ts=now_ts)
        self._refresh_topics(subject_id, affected_topics, now_ts=now_ts)
        self._refresh_global(subject_id, now_ts=now_ts)

    def ensure_subject_experts(self, subject_id: str) -> None:
        experts = self.store.list_experts(subject_id)
        if experts:
            return
        self._bootstrap_subject(subject_id)

    def _bootstrap_existing_subjects(self) -> None:
        for subject_id in self.store.list_subject_ids():
            self.ensure_subject_experts(subject_id)

    def _bootstrap_subject(self, subject_id: str) -> None:
        timestamp = time.time()
        with self.store.transaction():
            global_record = ExpertRecord(
                expert_id=GLOBAL_EXPERT_ID,
                subject_id=subject_id,
                expert_kind="global",
                centroid=(),
                atom_count=0,
                updated_at=timestamp,
                last_activated_at=0.0,
            )
            self.store.add_expert(global_record)
            atoms = tuple(atom for atom in self.store.list_atoms(subject_id) if atom.atom_kind != "evidence")
            if atoms:
                initial_topic_id = self._create_topic_expert(subject_id, timestamp)
                for atom in atoms:
                    self.store.set_atom_memberships(subject_id, atom.atom_id, primary_expert_id=initial_topic_id)
                self._refresh_topics(subject_id, (initial_topic_id,), now_ts=timestamp)
            self._refresh_global(subject_id, now_ts=timestamp)

    def _create_topic_expert(self, subject_id: str, now_ts: float) -> str:
        expert_id = f"expert_topic_{uuid4().hex[:12]}"
        record = ExpertRecord(
            expert_id=expert_id,
            subject_id=subject_id,
            expert_kind="topic",
            centroid=(),
            atom_count=0,
            updated_at=now_ts,
            last_activated_at=0.0,
        )
        self.store.add_expert(record)
        return expert_id

    def _global_expert(self, subject_id: str) -> ExpertRecord:
        experts = self.store.list_experts(subject_id, expert_kind="global")
        if len(experts) != 1:
            raise RuntimeError(f"subject {subject_id} must have exactly one global expert")
        return experts[0]

    def _assign_primary_topics(
        self,
        subject_id: str,
        route: ExpertRoute,
        atoms: tuple[MemoryAtom, ...],
        edges: tuple[MemoryEdge, ...],
        *,
        now_ts: float,
    ) -> dict[str, str]:
        assignments: dict[str, str] = {}
        topic_ids = route.topic_expert_ids or (self._create_topic_expert(subject_id, now_ts),)
        for atom in atoms:
            if atom.atom_kind == "inhibition":
                continue
            if atom.atom_kind == "episode":
                assignments[atom.atom_id] = topic_ids[0]
                continue
            assignments[atom.atom_id] = self._best_topic_for_text(subject_id, atom_text(atom), topic_ids, now_ts=now_ts)
        for atom in atoms:
            if atom.atom_kind != "inhibition":
                continue
            target_topic = self._inhibition_target_topic(subject_id, atom, edges, assignments)
            assignments[atom.atom_id] = target_topic or self._best_topic_for_text(subject_id, atom_text(atom), topic_ids, now_ts=now_ts)
        return assignments

    def _best_topic_for_text(
        self,
        subject_id: str,
        text: str,
        candidate_ids: tuple[str, ...],
        *,
        now_ts: float,
    ) -> str:
        query_tokens = set(tokenize(text))
        query_vector = self.encoder.encode(text)
        token_scores = self.store.expert_token_overlap(subject_id, query_tokens, expert_kind="topic")
        scored: list[tuple[float, str]] = []
        for expert_id in candidate_ids:
            expert = self.store.get_expert(subject_id, expert_id)
            if expert is None:
                continue
            score = 0.55 * min(1.0, token_scores.get(expert_id, 0.0)) + 0.35 * centroid_similarity(query_vector, expert.centroid)
            scored.append((score, expert_id))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= NEW_EXPERT_SCORE_THRESHOLD:
            return scored[0][1]
        return self._create_topic_expert(subject_id, now_ts)

    def _inhibition_target_topic(
        self,
        subject_id: str,
        atom: MemoryAtom,
        edges: tuple[MemoryEdge, ...],
        assignments: dict[str, str],
    ) -> str | None:
        for edge in edges:
            if edge.source_atom_id != atom.atom_id:
                continue
            if edge.target_atom_id in assignments:
                return assignments[edge.target_atom_id]
            primary = self.store.primary_expert_id(subject_id, edge.target_atom_id)
            if primary is not None:
                return primary
        return None

    def _split_topics(self, subject_id: str, topic_ids: tuple[str, ...], *, now_ts: float) -> tuple[str, ...]:
        final_topics: list[str] = []
        for topic_id in topic_ids:
            expert = self.store.get_expert(subject_id, topic_id)
            if expert is None:
                continue
            atoms = self.store.list_expert_atoms(subject_id, topic_id, membership_role="primary")
            edges = self._edges_for_atoms(subject_id, atoms)
            if len(atoms) <= EXPERT_SPLIT_ATOMS and len(edges) <= EXPERT_SPLIT_EDGES:
                final_topics.append(topic_id)
                continue
            vectors = {
                atom.atom_id: self.encoder.encode(atom_text(atom))
                for atom in atoms
                if atom_text(atom)
            }
            left_ids, right_ids = split_embeddings(vectors)
            if not left_ids or not right_ids:
                final_topics.append(topic_id)
                continue
            left_topic_id = self._create_topic_expert(subject_id, now_ts)
            right_topic_id = self._create_topic_expert(subject_id, now_ts)
            for atom in atoms:
                if atom.atom_id in left_ids:
                    self.store.set_atom_memberships(subject_id, atom.atom_id, primary_expert_id=left_topic_id)
                elif atom.atom_id in right_ids:
                    self.store.set_atom_memberships(subject_id, atom.atom_id, primary_expert_id=right_topic_id)
            self.store.delete_expert(subject_id, topic_id)
            final_topics.extend((left_topic_id, right_topic_id))
        return tuple(final_topics)

    def _refresh_topics(self, subject_id: str, topic_ids: tuple[str, ...], *, now_ts: float) -> None:
        for topic_id in dict.fromkeys(topic_ids):
            expert = self.store.get_expert(subject_id, topic_id)
            if expert is None:
                continue
            atoms = self.store.list_expert_atoms(subject_id, topic_id, membership_role="primary")
            edges = self._edges_for_atoms(subject_id, atoms)
            activations = _solve_activation(atoms, edges, iterations=self.iterations)
            self.store.replace_expert_activation_cache(subject_id, topic_id, activations, updated_at=_latest_timestamp(atoms, edges, now_ts))
            self.store.replace_expert_terms(subject_id, topic_id, _token_weights(atoms))
            self.store.update_expert(
                subject_id,
                topic_id,
                centroid=_centroid(self.encoder, atoms),
                atom_count=len(atoms),
                updated_at=now_ts,
            )

    def _refresh_global(self, subject_id: str, *, now_ts: float) -> None:
        global_id = self._global_expert(subject_id).expert_id
        anchor_ids = self._global_anchor_ids(subject_id)
        self.store.replace_secondary_memberships(subject_id, global_id, anchor_ids)
        atoms = self.store.list_expert_atoms(subject_id, global_id)
        edges = self._edges_for_atoms(subject_id, atoms)
        activations = _solve_activation(atoms, edges, iterations=self.iterations)
        self.store.replace_expert_activation_cache(subject_id, global_id, activations, updated_at=_latest_timestamp(atoms, edges, now_ts))
        self.store.replace_expert_terms(subject_id, global_id, _token_weights(atoms))
        self.store.update_expert(
            subject_id,
            global_id,
            centroid=_centroid(self.encoder, atoms),
            atom_count=len(atoms),
            updated_at=now_ts,
        )

    def _global_anchor_ids(self, subject_id: str) -> tuple[str, ...]:
        atoms_by_id: dict[str, MemoryAtom] = {}
        activation_by_id: dict[str, float] = {}
        episode_ids: list[str] = []
        commitment_ids: list[str] = []
        mainline_ids: list[str] = []
        tension_edges: list[tuple[float, str, str]] = []

        topic_ids = tuple(expert.expert_id for expert in self.store.list_experts(subject_id, expert_kind="topic"))
        for topic_id in topic_ids:
            atoms = self.store.list_expert_atoms(subject_id, topic_id, membership_role="primary")
            if not atoms:
                continue
            edges = self._edges_for_atoms(subject_id, atoms)
            activations = self.store.list_expert_activation_cache(subject_id, topic_id)
            selected_atoms, selected_edges = _select_slice(atoms, edges, activations, limit=_STATE_LIMIT)
            local_by_id = {atom.atom_id: atom for atom in atoms}
            atoms_by_id.update(local_by_id)
            for atom_id, activation in activations.items():
                activation_by_id[atom_id] = max(activation_by_id.get(atom_id, 0.0), activation)
            episode_ids.extend(atom.atom_id for atom in atoms if atom.atom_kind == "episode")
            commitment_ids.extend(atom.atom_id for atom in atoms if _is_commitment(atom_text(atom)))
            mainline_ids.extend(
                atom.atom_id
                for atom in selected_atoms
                if atom.atom_kind == "memory" and not _is_commitment(atom.text)
            )
            tension_edges.extend(
                (
                    abs(edge.influence) * edge.confidence * activations.get(edge.source_atom_id, 0.0),
                    edge.source_atom_id,
                    edge.target_atom_id,
                )
                for edge in selected_edges
                if edge.influence < 0.0
            )

        selected_ids: list[str] = []
        selected_ids.extend(_top_atom_ids(episode_ids, atoms_by_id, activation_by_id, limit=GLOBAL_EPISODE_LIMIT, sort_by_recency=True))
        selected_ids.extend(_top_atom_ids(commitment_ids, atoms_by_id, activation_by_id, limit=GLOBAL_COMMITMENT_LIMIT))
        selected_ids.extend(_top_atom_ids(mainline_ids, atoms_by_id, activation_by_id, limit=GLOBAL_MAINLINE_LIMIT))
        for _, source_atom_id, target_atom_id in sorted(tension_edges, reverse=True):
            for atom_id in (source_atom_id, target_atom_id):
                if atom_id not in selected_ids:
                    selected_ids.append(atom_id)
                if len([item for item in selected_ids if item in atoms_by_id]) >= (
                    GLOBAL_EPISODE_LIMIT + GLOBAL_COMMITMENT_LIMIT + GLOBAL_MAINLINE_LIMIT + GLOBAL_TENSION_LIMIT
                ):
                    break
            if len(selected_ids) >= GLOBAL_EPISODE_LIMIT + GLOBAL_COMMITMENT_LIMIT + GLOBAL_MAINLINE_LIMIT + GLOBAL_TENSION_LIMIT:
                break
        ordered: list[str] = []
        seen: set[str] = set()
        for atom_id in selected_ids:
            if atom_id in atoms_by_id and atom_id not in seen:
                seen.add(atom_id)
                ordered.append(atom_id)
        return tuple(ordered)

    def _load_expert_scope(
        self,
        subject_id: str,
        expert_ids: tuple[str, ...],
    ) -> tuple[tuple[MemoryAtom, ...], tuple[MemoryEdge, ...], dict[str, float]]:
        ordered_atoms: list[MemoryAtom] = []
        seen: set[str] = set()
        activations: dict[str, float] = {}
        for expert_id in dict.fromkeys(expert_ids):
            for atom in self.store.list_expert_atoms(subject_id, expert_id):
                if atom.atom_id in seen:
                    continue
                seen.add(atom.atom_id)
                ordered_atoms.append(atom)
            for atom_id, activation in self.store.list_expert_activation_cache(subject_id, expert_id).items():
                activations[atom_id] = max(activations.get(atom_id, 0.0), activation)
        atom_ids = tuple(atom.atom_id for atom in ordered_atoms)
        edges = tuple(
            edge
            for edge in self.store.list_edges(subject_id, atom_ids=atom_ids)
            if edge.source_atom_id in seen and edge.target_atom_id in seen
        )
        return tuple(ordered_atoms), edges, activations

    def _edges_for_atoms(self, subject_id: str, atoms: tuple[MemoryAtom, ...]) -> tuple[MemoryEdge, ...]:
        atom_ids = tuple(atom.atom_id for atom in atoms)
        atom_id_set = set(atom_ids)
        if not atom_ids:
            return ()
        return tuple(
            edge
            for edge in self.store.list_edges(subject_id, atom_ids=atom_ids)
            if edge.source_atom_id in atom_id_set and edge.target_atom_id in atom_id_set
        )


def _centroid(encoder: HashEmbeddingEncoder, atoms: tuple[MemoryAtom, ...]) -> tuple[float, ...]:
    vectors = tuple(encoder.encode(atom_text(atom)) for atom in atoms if atom_text(atom))
    return centroid_from_vectors(vectors)


def _token_weights(atoms: tuple[MemoryAtom, ...]) -> dict[str, float]:
    counts: dict[str, float] = {}
    total = 0.0
    for atom in atoms:
        text = atom_text(atom)
        if not text:
            continue
        for token in set(tokenize(text)):
            counts[token] = counts.get(token, 0.0) + 1.0
            total += 1.0
    if total <= 0.0:
        return {}
    return {token: value / total for token, value in counts.items()}


def _is_commitment(text: str) -> bool:
    return text.startswith("Aurora commitment:")


def _top_atom_ids(
    atom_ids: list[str],
    atoms_by_id: dict[str, MemoryAtom],
    activation_by_id: dict[str, float],
    *,
    limit: int,
    sort_by_recency: bool = False,
) -> list[str]:
    unique = [atom_id for atom_id in dict.fromkeys(atom_ids) if atom_id in atoms_by_id]
    if sort_by_recency:
        unique.sort(
            key=lambda atom_id: (
                atoms_by_id[atom_id].created_at,
                activation_by_id.get(atom_id, 0.0),
                atom_id,
            ),
            reverse=True,
        )
    else:
        unique.sort(
            key=lambda atom_id: (
                activation_by_id.get(atom_id, 0.0),
                atoms_by_id[atom_id].created_at,
                atom_id,
            ),
            reverse=True,
        )
    return unique[:limit]


def _solve_activation(
    atoms: tuple[MemoryAtom, ...],
    edges: tuple[MemoryEdge, ...],
    *,
    seed_bias: dict[str, float] | None = None,
    initial_activation: dict[str, float] | None = None,
    iterations: int,
) -> dict[str, float]:
    if not atoms:
        return {}

    atom_ids = [atom.atom_id for atom in atoms]
    index = {atom_id: position for position, atom_id in enumerate(atom_ids)}
    retention = np.array(
        [max(atom.confidence * atom.salience, 0.0) for atom in atoms],
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
                source_signal = max(0.0, float(current[source_index] - intrinsic[source_index]))
            else:
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
        if edge.source_atom_id not in index or edge.target_atom_id not in index or edge.influence <= 0.0:
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


def _compile_atoms(
    atoms: tuple[MemoryAtom, ...],
    edges: tuple[MemoryEdge, ...],
    activations: dict[str, float],
    *,
    limit: int,
) -> tuple[ActivatedAtom, ...]:
    selected_atoms, _ = _select_slice(atoms, edges, activations, limit=limit)
    if not selected_atoms:
        return ()
    threshold = max(0.1, selected_atoms[0].activation * 0.5)
    return tuple(
        atom
        for atom in selected_atoms
        if atom.activation >= threshold or atom.atom_id == selected_atoms[0].atom_id
    )


def _latest_timestamp(atoms: tuple[MemoryAtom, ...], edges: tuple[MemoryEdge, ...], fallback: float) -> float:
    latest = fallback
    for atom in atoms:
        latest = max(latest, atom.created_at)
    for edge in edges:
        latest = max(latest, edge.created_at)
    return latest
