from __future__ import annotations

import math
import numpy as np
from typing import List

from .base import EmbeddingProvider


class HashEmbedding(EmbeddingProvider):
    """A deterministic embedding for offline tests.

    Production should replace this with OpenAI / sentence-transformers, etc.
    """

    def __init__(self, dim: int = 384, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def embed(self, text: str) -> List[float]:
        # simple signed hashing into buckets
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = (hash(tok) + self.seed) % self.dim
            sign = -1.0 if (hash(tok + "!") & 1) else 1.0
            v[h] += sign
        # normalize
        n = float(np.linalg.norm(v)) or 1.0
        v = (v / n).astype(np.float32)
        return v.tolist()
