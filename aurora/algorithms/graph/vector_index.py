"""
AURORA Vector Index
====================

Brute-force vector index for similarity search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.utils.math_utils import cosine_sim


class VectorIndex:
    """Brute-force vector index with kind filtering.

    Replace with FAISS/pgvector for production.

    DEPRECATED: Use aurora.storage.vector_store.VectorStore instead for production.
    This class is kept for backward compatibility and testing.

    Attributes:
        dim: Vector dimension
        ids: List of vector IDs
        vecs: List of vectors
        kinds: List of vector kinds
    """

    def __init__(self, dim: int):
        """Initialize an empty vector index.

        Args:
            dim: Vector dimension
        """
        self.dim = dim
        self.ids: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.kinds: List[str] = []

    def add(self, _id: str, vec: np.ndarray, kind: str) -> None:
        """Add a vector to the index.

        Args:
            _id: Unique identifier
            vec: Vector to add
            kind: Kind label for filtering

        Raises:
            ValueError: If vector dimension doesn't match
        """
        vec = vec.astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"vector dim mismatch: {vec.shape} vs {(self.dim,)}")
        self.ids.append(_id)
        self.vecs.append(vec)
        self.kinds.append(kind)

    def remove(self, _id: str) -> None:
        """Remove a vector from the index.

        Args:
            _id: ID of vector to remove
        """
        if _id not in self.ids:
            return
        i = self.ids.index(_id)
        self.ids.pop(i)
        self.vecs.pop(i)
        self.kinds.pop(i)

    def search(self, q: np.ndarray, k: int = 10, kind: Optional[str] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors.

        Args:
            q: Query vector
            k: Number of results to return
            kind: Optional kind filter

        Returns:
            List of (id, similarity) tuples, sorted by similarity descending
        """
        if not self.vecs:
            return []
        q = q.astype(np.float32)
        hits: List[Tuple[str, float]] = []
        for _id, v, kd in zip(self.ids, self.vecs, self.kinds):
            if kind is not None and kd != kind:
                continue
            hits.append((_id, cosine_sim(q, v)))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "dim": self.dim,
            "ids": self.ids,
            "vecs": [v.tolist() for v in self.vecs],
            "kinds": self.kinds,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "VectorIndex":
        """Reconstruct from state dict."""
        obj = cls(dim=d["dim"])
        obj.ids = d["ids"]
        obj.vecs = [np.array(v, dtype=np.float32) for v in d["vecs"]]
        obj.kinds = d["kinds"]
        return obj
