"""Backward-compatible constants exports.

Deprecated:
    Import from ``aurora.core.config`` or a specific
    ``aurora.core.config.*`` module instead.
"""

from __future__ import annotations

import warnings

from aurora.core.config import *  # noqa: F401,F403
from aurora.core.config import __all__ as _CONFIG_ALL

warnings.warn(
    (
        "aurora.core.constants is deprecated; import from aurora.core.config "
        "or a specific aurora.core.config.* module instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_CONFIG_ALL)
