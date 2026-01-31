"""
AURORA Density Estimation
==========================

Online kernel density estimator for surprise computation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from aurora.algorithms.constants import DEFAULT_COLD_START_SURPRISE, DENSITY_MIN_SAMPLES


class OnlineKDE:
    """Kernel density estimator in embedding space.

    Used to compute surprise = -log p(x) without any similarity thresholds.

    This implements a reservoir-sampled KDE that maintains a bounded memory
    footprint while providing density estimates for incoming embeddings.

    Attributes:
        dim: Embedding dimension
        reservoir: Maximum number of vectors to retain
        k_sigma: Number of nearest neighbors for bandwidth estimation
    """

    def __init__(self, dim: int, reservoir: int = 4096, k_sigma: int = 20, seed: int = 0):
        """Initialize the online KDE.

        Args:
            dim: Embedding dimension
            reservoir: Maximum reservoir size for sampling
            k_sigma: k-nearest neighbors for adaptive bandwidth (default: 20)
            seed: Random seed for reservoir sampling
        
        Benchmark optimization:
        - Lower k_sigma (20 vs 25) produces sharper surprise peaks
        - This helps AR by making novel/important information more distinguishable
        - Also helps CR by making contradictory information stand out more
        """
        self.dim = dim
        self.reservoir = reservoir
        self.k_sigma = k_sigma
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._vecs: List[np.ndarray] = []

    def add(self, x: np.ndarray) -> None:
        """Add a vector to the density estimate.

        Uses reservoir sampling to maintain bounded memory.

        Args:
            x: Embedding vector to add
        """
        x = x.astype(np.float32)
        if len(self._vecs) < self.reservoir:
            self._vecs.append(x)
        else:
            # Reservoir sampling (capacity-limited memory)
            j = int(self.rng.integers(0, len(self._vecs) + 1))
            if j < len(self._vecs):
                self._vecs[j] = x

    def _sigma(self, x: np.ndarray) -> float:
        """Compute adaptive bandwidth using k-nearest neighbors.

        Args:
            x: Query vector

        Returns:
            Bandwidth estimate (median distance to k nearest neighbors)
        """
        if not self._vecs:
            return 1.0
        dists = [float(np.linalg.norm(x - v)) for v in self._vecs]
        dists.sort()
        k = min(self.k_sigma, len(dists))
        med = float(np.median(dists[:k])) if k > 0 else float(np.median(dists))
        return med + 1e-6

    def log_density(self, x: np.ndarray) -> float:
        """Compute log density at a point.

        Args:
            x: Query vector

        Returns:
            Log density estimate
        """
        if not self._vecs:
            # Weak prior: very low density
            return -10.0
        sigma = self._sigma(x)
        inv2 = 1.0 / (2.0 * sigma * sigma)
        vals = []
        for v in self._vecs:
            d2 = float(np.dot(x - v, x - v))
            vals.append(math.exp(-d2 * inv2))
        p = sum(vals) / len(vals)
        return math.log(p + 1e-12)

    def surprise(self, x: np.ndarray) -> float:
        """Compute surprise (negative log density) at a point.

        Cold start protection:
        - When samples < DENSITY_MIN_SAMPLES, KDE estimates are unreliable
        - Returns DEFAULT_COLD_START_SURPRISE to encourage early storage
        - This prevents losing critical early information (names, preferences, etc.)

        Args:
            x: Query vector

        Returns:
            Surprise value (higher = more surprising)
        """
        # Cold start protection: use default surprise when insufficient samples
        if len(self._vecs) < DENSITY_MIN_SAMPLES:
            return DEFAULT_COLD_START_SURPRISE
        return -self.log_density(x)

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "dim": self.dim,
            "reservoir": self.reservoir,
            "k_sigma": self.k_sigma,
            "seed": self._seed,
            "vecs": [v.tolist() for v in self._vecs],
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "OnlineKDE":
        """Reconstruct from state dict."""
        obj = cls(
            dim=d["dim"],
            reservoir=d["reservoir"],
            k_sigma=d["k_sigma"],
            seed=d["seed"],
        )
        obj._vecs = [np.array(v, dtype=np.float32) for v in d.get("vecs", [])]
        return obj
