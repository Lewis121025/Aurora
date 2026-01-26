"""
AURORA Math Utilities
=====================

Numerical utility functions for vector operations and probability calculations.
All functions are numerically stable and type-annotated.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2 normalize a vector.

    Args:
        v: Input vector
        eps: Epsilon for numerical stability (avoids division by zero)

    Returns:
        Unit vector in the same direction as v, or copy of v if norm < eps
    """
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return (v / n).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.

    Args:
        x: Input value

    Returns:
        sigmoid(x) in (0, 1)
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(logits: Sequence[float]) -> List[float]:
    """
    Numerically stable softmax function.

    Args:
        logits: Sequence of log-probabilities

    Returns:
        List of probabilities that sum to 1
    """
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    Z = sum(exps) + 1e-12
    return [e / Z for e in exps]


def log_sum_exp(logits: Sequence[float]) -> float:
    """
    Numerically stable log-sum-exp.

    Args:
        logits: Sequence of log values

    Returns:
        log(sum(exp(logits)))
    """
    m = max(logits)
    return m + math.log(sum(math.exp(x - m) for x in logits) + 1e-12)
