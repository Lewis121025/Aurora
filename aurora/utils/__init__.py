"""
AURORA Utilities
================

Common utility functions for the AURORA memory system.
"""

from aurora.utils.math_utils import (
    l2_normalize,
    cosine_sim,
    sigmoid,
    softmax,
    log_sum_exp,
)
from aurora.utils.id_utils import (
    AURORA_NAMESPACE,
    det_id,
    stable_hash,
    content_hash,
)
from aurora.utils.time_utils import (
    now_ts,
    set_mock_time,
    age_hours,
    age_days,
)

__all__ = [
    # math_utils
    "l2_normalize",
    "cosine_sim",
    "sigmoid",
    "softmax",
    "log_sum_exp",
    # id_utils
    "AURORA_NAMESPACE",
    "det_id",
    "stable_hash",
    "content_hash",
    # time_utils
    "now_ts",
    "set_mock_time",
    "age_hours",
    "age_days",
]
