"""
AURORA FAISS Vector Index
=========================

High-performance vector index using FAISS HNSW.
Provides sub-millisecond retrieval for production use.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length for cosine similarity."""
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-12 else v.astype(np.float32)


class FAISSVectorIndex:
    """FAISS HNSW vector index with kind filtering.
    
    Drop-in replacement for VectorIndex with ~100x faster search.
    """
    
    def __init__(
        self,
        dim: int,
        M: int = 32,
        ef_construction: int = 64,
        ef_search: int = 32,
        rebuild_threshold: int = 100,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required. Install: pip install faiss-cpu")
        
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.rebuild_threshold = rebuild_threshold
        
        self.ids: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.kinds: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        
        self._index: Optional[faiss.IndexHNSWFlat] = None
        self._dirty_count = 0
        self._build_index()
    
    def _build_index(self) -> None:
        """Build or rebuild FAISS index."""
        self._index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.hnsw.efSearch = self.ef_search
        
        if self.vecs:
            vectors = np.array([_l2_normalize(v) for v in self.vecs], dtype=np.float32)
            self._index.add(vectors)
        
        self._dirty_count = 0
    
    def _maybe_rebuild(self) -> None:
        if self._dirty_count >= self.rebuild_threshold:
            self._build_index()
    
    def add(self, _id: str, vec: np.ndarray, kind: str) -> None:
        vec = vec.astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"vector dim mismatch: {vec.shape} vs {(self.dim,)}")
        
        if _id in self._id_to_idx:
            idx = self._id_to_idx[_id]
            self.vecs[idx] = vec
            self.kinds[idx] = kind
            self._dirty_count += 1
        else:
            self.ids.append(_id)
            self.vecs.append(vec)
            self.kinds.append(kind)
            self._id_to_idx[_id] = len(self.ids) - 1
            self._index.add(_l2_normalize(vec).reshape(1, -1))
        
        self._maybe_rebuild()
    
    def remove(self, _id: str) -> None:
        if _id not in self._id_to_idx:
            return
        
        idx = self._id_to_idx[_id]
        last_idx = len(self.ids) - 1
        
        if idx != last_idx:
            self.ids[idx] = self.ids[last_idx]
            self.vecs[idx] = self.vecs[last_idx]
            self.kinds[idx] = self.kinds[last_idx]
            self._id_to_idx[self.ids[idx]] = idx
        
        self.ids.pop()
        self.vecs.pop()
        self.kinds.pop()
        del self._id_to_idx[_id]
        
        self._dirty_count += 1
        self._maybe_rebuild()
    
    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        kind: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        if not self.vecs or self._index is None:
            return []
        
        q = _l2_normalize(q.astype(np.float32)).reshape(1, -1)
        
        if kind is None:
            search_k = min(k, len(self.vecs))
            similarities, indices = self._index.search(q, search_k)
            return [
                (self.ids[idx], float(sim))
                for sim, idx in zip(similarities[0], indices[0])
                if 0 <= idx < len(self.ids)
            ]
        
        # With kind filtering
        kind_count = sum(1 for kd in self.kinds if kd == kind)
        if kind_count == 0:
            return []
        
        search_k = min(max(k * len(self.vecs) // kind_count + k, k * 3), len(self.vecs))
        similarities, indices = self._index.search(q, search_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if 0 <= idx < len(self.ids) and self.kinds[idx] == kind:
                results.append((self.ids[idx], float(sim)))
                if len(results) >= k:
                    break
        return results
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "ids": self.ids,
            "vecs": [v.tolist() for v in self.vecs],
            "kinds": self.kinds,
        }
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "FAISSVectorIndex":
        obj = cls(
            dim=d["dim"],
            M=d.get("M", 32),
            ef_construction=d.get("ef_construction", 64),
            ef_search=d.get("ef_search", 32),
        )
        obj.ids = d["ids"]
        obj.vecs = [np.array(v, dtype=np.float32) for v in d["vecs"]]
        obj.kinds = d["kinds"]
        obj._id_to_idx = {_id: i for i, _id in enumerate(obj.ids)}
        obj._build_index()
        return obj
