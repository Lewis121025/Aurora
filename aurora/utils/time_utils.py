"""
AURORA Time Utilities
=====================

Time-related utility functions.
Supports mocking for deterministic testing.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

# Optional mock function for testing
_mock_time: Optional[Callable[[], float]] = None


def now_ts() -> float:
    """
    Get current timestamp in seconds since epoch.

    Returns:
        Current time as float (seconds.microseconds)

    Note:
        Can be mocked for testing using set_mock_time()
    """
    if _mock_time is not None:
        return _mock_time()
    return time.time()


def set_mock_time(mock_fn: Optional[Callable[[], float]]) -> None:
    """
    Set a mock time function for testing.

    Args:
        mock_fn: Function that returns mock timestamp, or None to disable mocking

    Example:
        >>> counter = [0.0]
        >>> set_mock_time(lambda: (counter[0] := counter[0] + 1.0))
        >>> now_ts()
        1.0
        >>> now_ts()
        2.0
        >>> set_mock_time(None)  # Restore real time
    """
    global _mock_time
    _mock_time = mock_fn


def age_hours(ts: float) -> float:
    """
    Calculate age in hours from a timestamp.

    Args:
        ts: Timestamp to compare against current time

    Returns:
        Age in hours (can be negative if ts is in the future)
    """
    return (now_ts() - ts) / 3600.0


def age_days(ts: float) -> float:
    """
    Calculate age in days from a timestamp.

    Args:
        ts: Timestamp to compare against current time

    Returns:
        Age in days (can be negative if ts is in the future)
    """
    return (now_ts() - ts) / 86400.0
