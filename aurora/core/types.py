"""
Type aliases for AURORA algorithms.

This module defines common type aliases used across the AURORA algorithms
to improve type safety and code readability.
"""

from typing import Union

from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme

# Type alias for any memory element (Plot, StoryArc, or Theme)
# Using Union instead of TypeAlias for broader Python version compatibility
MemoryElement = Union[Plot, StoryArc, Theme]

__all__ = ["MemoryElement"]
