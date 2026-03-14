from __future__ import annotations
import numpy as np
from aurora.core_math.state import MetricState, cosine, normalize


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def prediction_error(cue: np.ndarray, latent: np.ndarray, vad_cue: list[float] | None = None, vad_latent: list[float] | None = None) -> float:
    semantic_err = float(max(0.0, 1.0 - cosine(cue, normalize(latent))))
    
    if vad_cue and vad_latent and len(vad_cue) == 3 and len(vad_latent) == 3:
        # VAD vectors are roughly [-1, 1], compute Euclidean distance normalized
        vad_err = sum((c - l) ** 2 for c, l in zip(vad_cue, vad_latent)) / 12.0
        return semantic_err * 0.6 + vad_err * 0.4
    return semantic_err


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


def advance_latent_ou_twin(
    core: np.ndarray,
    surface: np.ndarray,
    metric: MetricState,
    dt_hours: float,
    rng: np.random.Generator,
    drift: float = 0.12,
    temperature: float = 0.08,
    spring_k: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-vector drift: Surface wanders but is pulled by Core. Core is rigid."""
    if dt_hours <= 0.0:
        return core.copy(), surface.copy()
        
    # Core drifts almost imperceptibly
    core_drift = 0.001
    core_exp = float(np.exp(-core_drift * dt_hours))
    new_core = normalize(core_exp * core + np.sqrt(max(0.0001, 1e-8)) * rng.normal(size=metric.dim))

    # Surface drifts and is pulled toward Core
    exp_term = float(np.exp(-drift * dt_hours))
    variance = (temperature / drift) * (1.0 - np.exp(-2.0 * drift * dt_hours))
    chol = np.linalg.cholesky(metric.matrix() + 1e-6 * np.eye(metric.dim))
    noise = chol @ rng.normal(size=metric.dim)
    
    # Spring tension from core
    pull = spring_k * dt_hours * (new_core - surface)
    
    new_surface = normalize(exp_term * surface + pull + np.sqrt(max(variance, 1e-8)) * noise)
    return new_core, new_surface


def boundary_budget(error: float, basin_pressure: float, mutual_respect: float = 0.0) -> float:
    # High respect lowers the boundary threshold
    base = sigmoid(2.2 * error + 0.45 * basin_pressure - 1.2)
    return float(np.clip(base - (0.3 * mutual_respect), 0.0, 1.0))


def verbosity_budget(error: float, recalled_count: int) -> float:
    return float(np.clip(0.35 + 0.45 * error + 0.08 * recalled_count, 0.1, 1.0))
