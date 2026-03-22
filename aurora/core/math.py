"""Numerical helpers for the Aurora trace field."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
EPS = 1e-8
LOG_2PI = math.log(2.0 * math.pi)


def as_float_array(x: Sequence[float] | NDArray[np.floating]) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1D vector, got shape={arr.shape}")
    return arr


def safe_variance(
    x: Sequence[float] | NDArray[np.floating],
    *,
    min_value: float = 1e-4,
    max_value: float = 1e4,
) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    return np.clip(arr, min_value, max_value)


def l2_normalize(x: Sequence[float] | NDArray[np.floating], *, eps: float = EPS) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < eps:
        return arr.astype(np.float64, copy=True)
    return np.asarray(arr / norm, dtype=np.float64)


def cosine_similarity(
    x: Sequence[float] | NDArray[np.floating],
    y: Sequence[float] | NDArray[np.floating],
    *,
    eps: float = EPS,
) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    denom = max(np.linalg.norm(x_arr) * np.linalg.norm(y_arr), eps)
    return float(np.dot(x_arr, y_arr) / denom)


def diag_gaussian_logpdf(
    x: Sequence[float] | NDArray[np.floating],
    mean: Sequence[float] | NDArray[np.floating],
    sigma_diag: Sequence[float] | NDArray[np.floating],
    *,
    min_var: float = 1e-4,
    max_var: float = 1e4,
) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    var = safe_variance(sigma_diag, min_value=min_var, max_value=max_var)
    diff = x_arr - mean_arr
    mahal = np.sum((diff * diff) / var)
    log_det = np.sum(np.log(var))
    return float(-0.5 * (mahal + log_det + x_arr.size * LOG_2PI))


def squared_mahalanobis(
    x: Sequence[float] | NDArray[np.floating],
    mean: Sequence[float] | NDArray[np.floating],
    sigma_diag: Sequence[float] | NDArray[np.floating],
    *,
    min_var: float = 1e-4,
    max_var: float = 1e4,
) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    var = safe_variance(sigma_diag, min_value=min_var, max_value=max_var)
    diff = x_arr - mean_arr
    return float(np.sum((diff * diff) / var))


def logsumexp(values: Sequence[float] | NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("-inf")
    max_val = float(np.max(arr))
    if not math.isfinite(max_val):
        return max_val
    return float(max_val + np.log(np.sum(np.exp(arr - max_val))))


def softmax(values: Sequence[float] | NDArray[np.floating], *, temperature: float = 1.0) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    temp = max(float(temperature), EPS)
    shifted = arr / temp
    shifted = shifted - np.max(shifted)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if denom <= EPS:
        return np.full_like(arr, fill_value=1.0 / max(arr.size, 1), dtype=np.float64)
    return np.asarray(exps / denom, dtype=np.float64)


def entropy(probs: Sequence[float] | NDArray[np.floating], *, eps: float = EPS) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    arr = arr / np.sum(arr)
    return float(-np.sum(arr * np.log(arr)))


def weighted_mean(
    vectors: Iterable[Sequence[float] | NDArray[np.floating]],
    weights: Sequence[float] | NDArray[np.floating],
) -> Array:
    vectors_arr = [np.asarray(v, dtype=np.float64) for v in vectors]
    if not vectors_arr:
        raise ValueError("weighted_mean received no vectors")
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= EPS:
        return np.asarray(np.mean(np.stack(vectors_arr, axis=0), axis=0), dtype=np.float64)
    stacked = np.stack(vectors_arr, axis=0)
    return np.asarray(np.sum(stacked * w[:, None], axis=0) / w_sum, dtype=np.float64)


def project_top_k(probs: Sequence[float] | NDArray[np.floating], k: int) -> Array:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.size <= k:
        total = float(np.sum(arr))
        if total <= EPS:
            return np.full_like(arr, fill_value=1.0 / max(arr.size, 1), dtype=np.float64)
        return arr / total
    idx = np.argpartition(arr, -k)[-k:]
    out = np.zeros_like(arr, dtype=np.float64)
    out[idx] = arr[idx]
    total = float(np.sum(out))
    if total <= EPS:
        out[int(np.argmax(arr))] = 1.0
        return out
    return out / total


def entmax_bisect(
    logits: Sequence[float] | NDArray[np.floating],
    *,
    alpha: float = 1.5,
    n_iter: int = 50,
) -> Array:
    if alpha <= 1.0:
        return softmax(logits)
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError(f"expected 1D logits, got {z.shape}")
    z = z - np.max(z)
    scale = alpha - 1.0
    power = 1.0 / scale
    tau_lo = float(np.min(z) - 1.0 / scale)
    tau_hi = float(np.max(z))

    def phi(tau: float) -> float:
        p = np.maximum(scale * (z - tau), 0.0) ** power
        return float(np.sum(p) - 1.0)

    lo, hi = tau_lo, tau_hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        value = phi(mid)
        if value > 0.0:
            lo = mid
        else:
            hi = mid

    tau = lo
    p = np.maximum(scale * (z - tau), 0.0) ** power
    total = float(np.sum(p))
    if total <= EPS:
        out = np.zeros_like(z)
        out[int(np.argmax(z))] = 1.0
        return out
    return p / total
