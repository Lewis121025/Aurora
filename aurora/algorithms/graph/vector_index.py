"""AURORA Vector Index - Brute-force baseline for testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.utils.math_utils import cosine_sim


class VectorIndex:
    """Brute-force vector index with kind filtering.
    
    Use FAISSVectorIndex for production (100x faster).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.kinds: List[str] = []

    def add(self, _id: str, vec: np.ndarray, kind: str) -> None:
        vec = vec.astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"vector dim mismatch: {vec.shape} vs {(self.dim,)}")
        self.ids.append(_id)
        self.vecs.append(vec)
        self.kinds.append(kind)

    def remove(self, _id: str) -> None:
        if _id not in self.ids:
            return
        i = self.ids.index(_id)
        self.ids.pop(i)
        self.vecs.pop(i)
        self.kinds.pop(i)

    def search(self, q: np.ndarray, k: int = 10, kind: Optional[str] = None) -> List[Tuple[str, float]]:
        if not self.vecs:
            return []
        q = q.astype(np.float32)
        hits = [
            (_id, cosine_sim(q, v))
            for _id, v, kd in zip(self.ids, self.vecs, self.kinds)
            if kind is None or kd == kind
        ]
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "ids": self.ids,
            "vecs": [v.tolist() for v in self.vecs],
            "kinds": self.kinds,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "VectorIndex":
        obj = cls(dim=d["dim"])
        obj.ids = d["ids"]
        obj.vecs = [np.array(v, dtype=np.float32) for v in d["vecs"]]
        obj.kinds = d["kinds"]
        return obj
