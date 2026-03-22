"""Action primitives for Aurora proposal decisions."""

from __future__ import annotations

import numpy as np

from aurora.core.types import ActionName

ACTION_ORDER: tuple[ActionName, ...] = (
    "ASSIMILATE",
    "ATTACH",
    "SPLIT",
    "BIRTH",
    "INHIBIT",
)
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_ORDER)}


def action_vector(action: ActionName) -> np.ndarray:
    vec = np.zeros(len(ACTION_ORDER), dtype=np.float64)
    vec[ACTION_TO_INDEX[action]] = 1.0
    return vec
