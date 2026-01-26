"""
AURORA Metric Learning
=======================

Low-rank Mahalanobis metric for adaptive similarity learning.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


class LowRankMetric:
    """Low-rank Mahalanobis metric: d(x,y)^2 = ||L(x-y)||^2.

    Interpretable as learning a task- and user-adapted "information geometry" metric
    from retrieval feedback.

    The metric learns to map the embedding space such that relevant items
    are close and irrelevant items are far apart.

    Parameter stability:
        - window_size: Sliding window for Adagrad accumulator reset.
          Every `window_size` updates, the accumulator G is rescaled to prevent
          the learning rate from decaying to near-zero over time.
        - decay_factor: Applied to G during periodic recomputation (default 0.5).

    Attributes:
        dim: Input embedding dimension
        rank: Rank of the metric (output dimension of L)
        L: The low-rank projection matrix (rank x dim)
        G: Adagrad accumulator for adaptive learning rates
        t: Update counter
    """

    def __init__(
        self,
        dim: int,
        rank: int = 64,
        seed: int = 0,
        window_size: int = 10000,
        decay_factor: float = 0.5,
    ):
        """Initialize the low-rank metric.

        Args:
            dim: Embedding dimension
            rank: Rank of the learned metric
            seed: Random seed for initialization
            window_size: Adagrad accumulator reset window
            decay_factor: Decay factor for accumulator rescaling
        """
        self.dim = dim
        self.rank = min(rank, dim)
        self._seed = seed
        self.window_size = window_size
        self.decay_factor = decay_factor

        rng = np.random.default_rng(seed)
        self.L = np.eye(dim, dtype=np.float32)[: self.rank].copy()
        self.L += (0.01 * rng.normal(size=self.L.shape)).astype(np.float32)

        self.G = np.zeros_like(self.L)  # Adagrad accumulator
        self.t = 0

        # Statistics for monitoring
        self._total_loss = 0.0
        self._update_count = 0

    def d2(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute squared distance in the learned metric.

        Args:
            x: First embedding
            y: Second embedding

        Returns:
            Squared distance d(x,y)^2 = ||L(x-y)||^2
        """
        z = (x - y).astype(np.float32)
        p = self.L @ z
        return float(np.dot(p, p))

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute similarity in the learned metric.

        Args:
            x: First embedding
            y: Second embedding

        Returns:
            Similarity in (0, 1], where 1 means identical
        """
        return 1.0 / (1.0 + self.d2(x, y))

    def update_triplet(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        margin: float = 1.0,
    ) -> float:
        """Online OASIS-like update with Adagrad.

        Updates the metric to make anchor closer to positive and
        farther from negative.

        margin is not a threshold on similarity; it's a geometric separation unit.

        Includes sliding window mechanism: periodically rescales the Adagrad
        accumulator to prevent learning rate from vanishing.

        Args:
            anchor: Anchor embedding
            positive: Positive (similar) embedding
            negative: Negative (dissimilar) embedding
            margin: Margin for triplet loss

        Returns:
            Loss value (0 if constraint already satisfied)
        """
        self.t += 1
        ap = (anchor - positive).astype(np.float32)
        an = (anchor - negative).astype(np.float32)
        Lap = self.L @ ap
        Lan = self.L @ an
        dap = float(np.dot(Lap, Lap))
        dan = float(np.dot(Lan, Lan))
        loss = max(0.0, margin + dap - dan)
        if loss <= 0:
            return 0.0

        grad = 2.0 * (np.outer(Lap, ap) - np.outer(Lan, an)).astype(np.float32)
        self.G += grad * grad
        # Self-tuning learning rate: decays with t automatically
        base = 1.0 / math.sqrt(self.t + 1.0)
        step = base * grad / (np.sqrt(self.G) + 1e-8)
        self.L -= step

        # Track statistics
        self._total_loss += loss
        self._update_count += 1

        # Sliding window: periodically rescale G to maintain plasticity
        # This prevents the Adagrad accumulator from growing unboundedly
        # which would cause the learning rate to approach zero
        if self.t > 0 and self.t % self.window_size == 0:
            self._rescale_accumulator()

        return float(loss)

    def _rescale_accumulator(self) -> None:
        """Rescale the Adagrad accumulator to maintain learning capacity.

        This implements a "soft reset" that preserves learned structure
        while preventing the accumulator from growing too large.
        """
        self.G *= self.decay_factor

    def average_loss(self) -> float:
        """Return average triplet loss over all updates."""
        return self._total_loss / self._update_count if self._update_count > 0 else 0.0

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "dim": self.dim,
            "rank": self.rank,
            "seed": self._seed,
            "window_size": self.window_size,
            "decay_factor": self.decay_factor,
            "L": self.L.tolist(),
            "G": self.G.tolist(),
            "t": self.t,
            "total_loss": self._total_loss,
            "update_count": self._update_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "LowRankMetric":
        """Reconstruct from state dict."""
        obj = cls(
            dim=d["dim"],
            rank=d["rank"],
            seed=d.get("seed", 0),
            window_size=d.get("window_size", 10000),
            decay_factor=d.get("decay_factor", 0.5),
        )
        obj.L = np.array(d["L"], dtype=np.float32)
        obj.G = np.array(d["G"], dtype=np.float32)
        obj.t = d["t"]
        obj._total_loss = d.get("total_loss", 0.0)
        obj._update_count = d.get("update_count", 0)
        return obj
