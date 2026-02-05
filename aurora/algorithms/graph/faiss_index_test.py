"""Tests for FAISS vector index."""

import time
import numpy as np
import pytest

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from aurora.algorithms.graph.vector_index import VectorIndex

if FAISS_AVAILABLE:
    from aurora.algorithms.graph.faiss_index import FAISSVectorIndex


def _random_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestFAISSVectorIndex:
    
    def test_add_search(self):
        index = FAISSVectorIndex(dim=128)
        vecs = _random_vectors(100, 128)
        
        for i, vec in enumerate(vecs):
            index.add(f"id_{i}", vec, "plot")
        
        assert len(index) == 100
        
        results = index.search(vecs[0], k=5)
        assert len(results) == 5
        assert results[0][0] == "id_0"
        assert results[0][1] > 0.99
    
    def test_kind_filtering(self):
        index = FAISSVectorIndex(dim=128)
        vecs = _random_vectors(100, 128)
        
        for i, vec in enumerate(vecs):
            index.add(f"id_{i}", vec, "plot" if i % 2 == 0 else "story")
        
        results = index.search(vecs[0], k=10, kind="plot")
        for _id, _ in results:
            assert int(_id.split("_")[1]) % 2 == 0
    
    def test_remove(self):
        index = FAISSVectorIndex(dim=128)
        vecs = _random_vectors(10, 128)
        
        for i, vec in enumerate(vecs):
            index.add(f"id_{i}", vec, "plot")
        
        index.remove("id_5")
        assert len(index) == 9
        assert "id_5" not in [r[0] for r in index.search(vecs[5], k=10)]
    
    def test_serialization(self):
        index = FAISSVectorIndex(dim=128)
        vecs = _random_vectors(50, 128)
        
        for i, vec in enumerate(vecs):
            index.add(f"id_{i}", vec, "plot")
        
        restored = FAISSVectorIndex.from_state_dict(index.to_state_dict())
        assert len(restored) == 50
        
        orig = [r[0] for r in index.search(vecs[0], k=5)]
        rest = [r[0] for r in restored.search(vecs[0], k=5)]
        assert orig == rest
    
    def test_performance(self):
        dim, n_vecs, n_queries = 384, 5000, 100
        vecs = _random_vectors(n_vecs, dim)
        queries = _random_vectors(n_queries, dim, seed=999)
        
        brute = VectorIndex(dim=dim)
        faiss_idx = FAISSVectorIndex(dim=dim)
        
        for i, vec in enumerate(vecs):
            brute.add(f"id_{i}", vec, "plot")
            faiss_idx.add(f"id_{i}", vec, "plot")
        
        start = time.perf_counter()
        for q in queries:
            brute.search(q, k=10)
        brute_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for q in queries:
            faiss_idx.search(q, k=10)
        faiss_time = time.perf_counter() - start
        
        print(f"\n{n_vecs} vectors, {n_queries} queries:")
        print(f"  Brute: {brute_time/n_queries*1000:.2f}ms/query")
        print(f"  FAISS: {faiss_time/n_queries*1000:.2f}ms/query")
        print(f"  Speedup: {brute_time/faiss_time:.1f}x")
        
        assert faiss_time < brute_time
