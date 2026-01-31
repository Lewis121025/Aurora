"""
AURORA Hash Embedding
=====================

Deterministic pseudo-random embedding for testing.
Produces consistent embeddings based on text hash.

Usage:
    from aurora.embeddings import HashEmbedding
    
    embedder = HashEmbedding(dim=384)
    vec = embedder.embed("Hello world")
"""

from __future__ import annotations

from typing import List

import numpy as np

from aurora.embeddings.base import EmbeddingProvider
from aurora.utils.math_utils import l2_normalize
from aurora.utils.id_utils import stable_hash


class HashEmbedding(EmbeddingProvider):
    """Deterministic pseudo-random embedding for testing.
    
    Produces consistent embeddings based on text hash.
    Replace with real embedding model in production.
    
    Attributes:
        dim: Embedding dimension
        seed: Random seed for reproducibility
    """

    def __init__(self, dim: int = 384, seed: int = 7):
        self.dim = dim
        self.seed = seed

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text."""
        rng = np.random.default_rng(stable_hash(text) ^ self.seed)
        v = rng.normal(size=self.dim).astype(np.float32)
        return l2_normalize(v)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]
