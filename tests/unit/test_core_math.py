from __future__ import annotations

import numpy as np

from aurora.core_math.dynamics import advance_latent_ou, update_metric
from aurora.core_math.state import MetricState


def test_ou_advance_is_reproducible_with_fixed_seed() -> None:
    metric = MetricState.isotropic(dim=8, rank=3)
    vector = np.linspace(-0.4, 0.4, num=8)
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    a = advance_latent_ou(vector, metric, dt_hours=4.0, rng=rng_a)
    b = advance_latent_ou(vector, metric, dt_hours=4.0, rng=rng_b)
    assert np.allclose(a, b)


def test_metric_update_stays_spd() -> None:
    metric = MetricState.isotropic(dim=10, rank=4)
    for idx in range(20):
        gradient = np.zeros(10, dtype=np.float64)
        gradient[idx % 10] = 1.0
        metric = update_metric(metric, gradient, lr=0.05)
        eigvals = np.linalg.eigvalsh(metric.matrix())
        assert np.all(eigvals > 0.0)


def test_repeated_direction_breaks_symmetry() -> None:
    metric = MetricState.isotropic(dim=12, rank=4)
    gradient = np.zeros(12, dtype=np.float64)
    gradient[0] = 1.0
    for _ in range(40):
        metric = update_metric(metric, gradient, lr=0.08)
    eigvals = np.sort(np.linalg.eigvalsh(metric.matrix()))[::-1]
    assert eigvals[0] > eigvals[1]
