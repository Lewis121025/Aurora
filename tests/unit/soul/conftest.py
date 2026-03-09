from __future__ import annotations

import pytest

from aurora.soul.models import BELIEF_ORDER, TRAIT_ORDER, IdentityState
from aurora.soul.retrieval import LowRankMetric


@pytest.fixture
def metric() -> LowRankMetric:
    return LowRankMetric(dim=64, rank=16, seed=42)


@pytest.fixture
def identity_state() -> IdentityState:
    return IdentityState(
        traits={name: 0.5 for name in TRAIT_ORDER},
        beliefs={name: 0.5 for name in BELIEF_ORDER},
        phase="dependent_child",
    )

