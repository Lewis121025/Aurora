from __future__ import annotations

import numpy as np
import pytest

from aurora.soul.models import HOMEOSTATIC_AXES, IdentityState
from aurora.soul.retrieval import LowRankMetric


@pytest.fixture
def metric() -> LowRankMetric:
    return LowRankMetric(dim=64, rank=16, seed=42)


@pytest.fixture
def identity_state() -> IdentityState:
    axis_state = {name: 0.2 for name, _, _, _ in HOMEOSTATIC_AXES}
    self_vector = np.ones(64, dtype=np.float32)
    self_vector /= np.linalg.norm(self_vector)
    return IdentityState(
        self_vector=self_vector,
        axis_state=axis_state,
        intuition_axes={name: 0.0 for name in axis_state},
        current_mode_label="origin",
    )
