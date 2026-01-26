"""
AURORA Bandit Components
=========================

Thompson sampling based decision making for stochastic encoding.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from aurora.utils.math_utils import sigmoid


class ThompsonBernoulliGate:
    """Stochastic encode policy with Thompson sampling.

    We do not decide by "score > threshold". We sample a parameter vector w and
    encode with probability sigmoid(w·x). The mapping is learned from delayed rewards.

    This implements a Bayesian approach to the encode/skip decision, allowing
    the system to explore different encoding strategies while learning from
    downstream task success.

    Parameter stability:
        - forgetting_factor (lambda): Prevents precision from infinite accumulation.
          prec = lambda * prec + grad * grad, where lambda=0.99 ensures ~1% decay per update.
        - This keeps the system "plastic" and able to adapt to distribution shifts.

    Attributes:
        d: Feature dimension
        w_mean: Mean of the weight distribution
        prec: Precision (inverse variance) of the weight distribution
        grad2: RMSprop accumulator for adaptive learning
        t: Update counter
    """

    def __init__(self, feature_dim: int, seed: int = 0, forgetting_factor: float = 0.99):
        """Initialize the Thompson sampling gate.

        Args:
            feature_dim: Dimension of the feature vector
            seed: Random seed
            forgetting_factor: Decay factor for precision accumulation
        """
        self.d = feature_dim
        self._seed = seed
        self.lambda_ = forgetting_factor  # Forgetting factor for precision
        self.rng = np.random.default_rng(seed)

        self.w_mean = np.zeros(self.d, dtype=np.float32)
        self.prec = np.ones(self.d, dtype=np.float32) * 1e-2  # Weak precision
        self.grad2 = np.zeros(self.d, dtype=np.float32)  # RMS

        self.t = 0

        # Statistics tracking for monitoring
        self._encode_count = 0
        self._skip_count = 0

    def _sample_w(self) -> np.ndarray:
        """Sample a weight vector from the posterior.

        Returns:
            Sampled weight vector
        """
        std = np.sqrt(1.0 / (self.prec + 1e-9))
        return self.w_mean + self.rng.normal(size=self.d).astype(np.float32) * std

    def prob(self, x: np.ndarray) -> float:
        """Compute encoding probability for a feature vector.

        Args:
            x: Feature vector

        Returns:
            Probability of encoding (Thompson sampled)
        """
        w = self._sample_w()
        return sigmoid(float(np.dot(w, x)))

    def decide(self, x: np.ndarray) -> bool:
        """Make a stochastic encoding decision.

        Args:
            x: Feature vector

        Returns:
            True if should encode, False if should skip
        """
        result = bool(self.rng.random() < self.prob(x))
        if result:
            self._encode_count += 1
        else:
            self._skip_count += 1
        return result

    def update(self, x: np.ndarray, reward: float) -> None:
        """Bandit update: reward in [-1, 1] from downstream task success.

        Uses forgetting factor to prevent precision from accumulating indefinitely,
        which would cause variance to approach zero and freeze the policy.

        Args:
            x: Feature vector that was used for the decision
            reward: Reward signal from downstream task
        """
        self.t += 1
        y = 1.0 if reward > 0 else 0.0
        p = sigmoid(float(np.dot(self.w_mean, x)))
        grad = (y - p) * x  # Ascent

        # RMS with self-tuning step size
        self.grad2 = 0.99 * self.grad2 + 0.01 * (grad * grad)
        base = 1.0 / math.sqrt(self.t + 1.0)
        step = base * grad / (np.sqrt(self.grad2) + 1e-6)
        self.w_mean += step

        # Apply forgetting factor to prevent precision from growing unboundedly
        # This ensures the system remains "plastic" and can adapt to distribution shifts
        self.prec = self.lambda_ * self.prec + grad * grad

    def pass_rate(self) -> float:
        """Return the gate pass rate (encode / total decisions)."""
        total = self._encode_count + self._skip_count
        return self._encode_count / total if total > 0 else 0.5

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "d": self.d,
            "seed": self._seed,
            "lambda": self.lambda_,
            "w_mean": self.w_mean.tolist(),
            "prec": self.prec.tolist(),
            "grad2": self.grad2.tolist(),
            "t": self.t,
            "encode_count": self._encode_count,
            "skip_count": self._skip_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "ThompsonBernoulliGate":
        """Reconstruct from state dict."""
        obj = cls(
            feature_dim=d["d"],
            seed=d.get("seed", 0),
            forgetting_factor=d.get("lambda", 0.99),
        )
        obj.w_mean = np.array(d["w_mean"], dtype=np.float32)
        obj.prec = np.array(d["prec"], dtype=np.float32)
        obj.grad2 = np.array(d["grad2"], dtype=np.float32)
        obj.t = d["t"]
        obj._encode_count = d.get("encode_count", 0)
        obj._skip_count = d.get("skip_count", 0)
        return obj
