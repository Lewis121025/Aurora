"""
AURORA Embedding
=================

Embedding utilities for the memory system.
"""

from __future__ import annotations

import numpy as np

from aurora.utils.math_utils import l2_normalize
from aurora.utils.id_utils import stable_hash


class HashEmbedding:
    """Dependency-free embedding model for local testing.

    Replace with a real embedding model in production.

    This produces deterministic pseudo-random embeddings based on text content,
    useful for testing and development without external embedding services.

    Attributes:
        dim: Embedding dimension
        seed: Random seed for consistent embeddings
    """

    def __init__(self, dim: int = 384, seed: int = 7):
        """Initialize the hash embedding model.

        Args:
            dim: Embedding dimension
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.seed = seed

    def embed(self, text: str) -> np.ndarray:
        """Generate a deterministic embedding for text.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector
        """
        rng = np.random.default_rng(stable_hash(text) ^ self.seed)
        v = rng.normal(size=self.dim).astype(np.float32)
        return l2_normalize(v)
