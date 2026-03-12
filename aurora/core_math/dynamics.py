from __future__ import annotations
import numpy as np
from aurora.core_math.state import MetricState, cosine, normalize


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def prediction_error(cue: np.ndarray, latent: np.ndarray) -> float:
    return float(max(0.0, 1.0 - cosine(cue, normalize(latent))))


def update_metric(metric: MetricState, cue: np.ndarray, error: float, lr: float = 0.05) -> MetricState:
    """空间折叠：极大惊奇会瞬间拉伸空间矩阵，形成不可见的引力黑洞"""
    dim = metric.dim
    plasticity = float(np.clip(lr + 2.0 * (error ** 2), 0.01, 1.5))
    centered_outer = np.outer(cue, cue)
    centered_outer -= np.trace(centered_outer) / float(dim) * np.eye(dim)

    full = metric.matrix() + plasticity * centered_outer
    eigvals, eigvecs = np.linalg.eigh(full)
    order = np.argsort(np.abs(eigvals - 1.0))[::-1]
    chosen = order[:metric.lambdas.shape[0]]
    lambdas = np.clip(eigvals[chosen] - 1.0, -0.95, 12.0)
    return MetricState(dim=dim, basis=eigvecs[:, chosen], lambdas=lambdas)


def advance_latent_ou(
    vector: np.ndarray,
    metric: MetricState,
    dt_hours: float,
    rng: np.random.Generator,
    drift: float = 0.12,
    temperature: float = 0.08,
) -> np.ndarray:
    if dt_hours <= 0.0:
        return vector.copy()
    exp_term = float(np.exp(-drift * dt_hours))
    variance = (temperature / drift) * (1.0 - np.exp(-2.0 * drift * dt_hours))
    chol = np.linalg.cholesky(metric.matrix() + 1e-6 * np.eye(metric.dim))
    noise = chol @ rng.normal(size=metric.dim)
    return normalize(exp_term * vector + np.sqrt(max(variance, 1e-8)) * noise)


def boundary_budget(error: float, basin_pressure: float) -> float:
    return sigmoid(2.2 * error + 0.45 * basin_pressure - 1.2)


def verbosity_budget(error: float, recalled_count: int) -> float:
    return float(np.clip(0.35 + 0.45 * error + 0.08 * recalled_count, 0.1, 1.0))
