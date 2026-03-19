"""Expert-scoped routing helpers for Aurora."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

ExpertKind = Literal["global", "topic"]
MembershipRole = Literal["primary", "secondary"]
FloatVector = npt.NDArray[np.float32]

GLOBAL_EXPERT_ID = "expert_global"
ROUTED_TOPIC_EXPERTS = 2
COMPILE_ATOMS_PER_EXPERT = 16
NEW_EXPERT_SCORE_THRESHOLD = 0.35
EXPERT_SPLIT_ATOMS = 128
EXPERT_SPLIT_EDGES = 256

GLOBAL_EPISODE_LIMIT = 6
GLOBAL_MAINLINE_LIMIT = 6
GLOBAL_COMMITMENT_LIMIT = 4
GLOBAL_TENSION_LIMIT = 4


@dataclass(frozen=True, slots=True)
class ExpertRecord:
    expert_id: str
    subject_id: str
    expert_kind: ExpertKind
    centroid: tuple[float, ...]
    atom_count: int
    updated_at: float
    last_activated_at: float


@dataclass(frozen=True, slots=True)
class ExpertRoute:
    subject_id: str
    global_expert_id: str
    topic_expert_ids: tuple[str, ...]

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return (self.global_expert_id, *self.topic_expert_ids)

    @property
    def primary_topic_id(self) -> str | None:
        return self.topic_expert_ids[0] if self.topic_expert_ids else None


def centroid_from_vectors(vectors: tuple[FloatVector, ...]) -> tuple[float, ...]:
    if not vectors:
        return ()
    stacked = np.stack(vectors)
    centroid = np.mean(stacked, axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return tuple(float(value) for value in np.asarray(centroid).tolist())
    normalized = centroid / norm
    return tuple(float(value) for value in np.asarray(normalized).tolist())


def centroid_similarity(query_vector: FloatVector, centroid: tuple[float, ...]) -> float:
    if not centroid:
        return 0.0
    centroid_vector = np.array(centroid, dtype=np.float32)
    if float(np.linalg.norm(query_vector)) == 0.0 or float(np.linalg.norm(centroid_vector)) == 0.0:
        return 0.0
    return max(0.0, float(np.dot(query_vector, centroid_vector)))


def split_embeddings(vectors_by_id: dict[str, FloatVector]) -> tuple[set[str], set[str]]:
    if len(vectors_by_id) < 2:
        return set(vectors_by_id), set()

    ordered_ids = tuple(vectors_by_id)
    first = ordered_ids[0]
    second = max(ordered_ids[1:], key=lambda atom_id: _distance(vectors_by_id[first], vectors_by_id[atom_id]))
    center_a = _normalize(vectors_by_id[first])
    center_b = _normalize(vectors_by_id[second])

    group_a: set[str] = set()
    group_b: set[str] = set()
    for _ in range(6):
        next_a: set[str] = set()
        next_b: set[str] = set()
        for atom_id, vector in vectors_by_id.items():
            score_a = float(np.dot(vector, center_a))
            score_b = float(np.dot(vector, center_b))
            if score_a >= score_b:
                next_a.add(atom_id)
            else:
                next_b.add(atom_id)
        if not next_a or not next_b:
            midpoint = len(ordered_ids) // 2
            return set(ordered_ids[:midpoint]), set(ordered_ids[midpoint:])
        group_a, group_b = next_a, next_b
        center_a = _normalize(np.mean(np.stack([vectors_by_id[atom_id] for atom_id in group_a]), axis=0))
        center_b = _normalize(np.mean(np.stack([vectors_by_id[atom_id] for atom_id in group_b]), axis=0))
    return group_a, group_b


def _normalize(vector: FloatVector) -> FloatVector:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.array(vector, dtype=np.float32)
    return np.array(vector / norm, dtype=np.float32)


def _distance(left: FloatVector, right: FloatVector) -> float:
    return 1.0 - max(-1.0, min(1.0, float(np.dot(_normalize(left), _normalize(right)))))
