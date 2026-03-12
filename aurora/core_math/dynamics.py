from __future__ import annotations

import numpy as np

from aurora.core_math.state import MetricState, cosine, normalize


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


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
    stationary = temperature / max(drift, 1e-6)
    variance = stationary * (1.0 - np.exp(-2.0 * drift * dt_hours))
    diffusion = metric.matrix()
    chol = np.linalg.cholesky(diffusion + 1e-6 * np.eye(metric.dim))
    noise = chol @ rng.normal(size=metric.dim)
    return exp_term * vector + np.sqrt(max(variance, 1e-8)) * noise


def anisotropic_gradient(
    cue_embedding: np.ndarray,
    latent_vector: np.ndarray,
    sampled_embeddings: list[np.ndarray],
) -> np.ndarray:
    latent = normalize(latent_vector)
    if sampled_embeddings:
        memory_pull = normalize(np.mean(np.stack(sampled_embeddings, axis=0), axis=0))
    else:
        memory_pull = np.zeros_like(cue_embedding)
    return normalize(cue_embedding - latent + 0.65 * memory_pull)


def update_metric(metric: MetricState, gradient: np.ndarray, lr: float = 0.08) -> MetricState:
    dim = metric.dim
    centered_outer = np.outer(gradient, gradient)
    centered_outer -= np.trace(centered_outer) / float(dim) * np.eye(dim)
    full = metric.matrix() + lr * centered_outer
    eigvals, eigvecs = np.linalg.eigh(full)
    order = np.argsort(np.abs(eigvals - 1.0))[::-1]
    rank = metric.lambdas.shape[0]
    chosen = order[:rank]
    basis = eigvecs[:, chosen]
    lambdas = np.clip(eigvals[chosen] - 1.0, -0.95, 4.0)
    return MetricState(dim=dim, basis=basis, lambdas=lambdas)


def prediction_error(cue_embedding: np.ndarray, latent_vector: np.ndarray) -> float:
    return float(max(0.0, 1.0 - cosine(cue_embedding, normalize(latent_vector))))


def boundary_budget(error: float, basin_pressure: float) -> float:
    return sigmoid(2.2 * error + 0.45 * basin_pressure - 1.2)


def verbosity_budget(error: float, recalled_count: int) -> float:
    return float(np.clip(0.35 + 0.45 * error + 0.08 * recalled_count, 0.1, 1.0))
