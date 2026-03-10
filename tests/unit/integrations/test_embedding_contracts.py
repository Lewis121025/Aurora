from __future__ import annotations

import numpy as np

from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding


class TestEmbeddingContract:
    def test_local_semantic_returns_ndarray(self):
        embedder = LocalSemanticEmbedding(dim=64, seed=42)
        vec = embedder.embed("我住在北京")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_hash_returns_ndarray(self):
        embedder = HashEmbedding(dim=64, seed=42)
        vec = embedder.embed("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_batch_contract_is_consistent(self):
        embedder = LocalSemanticEmbedding(dim=32, seed=7)
        batch = embedder.embed_batch(["alpha", "beta"])
        assert len(batch) == 2
        assert all(isinstance(vec, np.ndarray) for vec in batch)
        assert all(vec.shape == (32,) for vec in batch)
