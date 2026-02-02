"""
AURORA Retrieval
=================

Field-based retrieval with attractor dynamics and query type awareness.

Time as First-Class Citizen: Temporal retrieval with anchor detection.
"""

from aurora.algorithms.retrieval.field_retriever import (
    FieldRetriever,
    QueryType,
    TimeAnchor,
)
from aurora.algorithms.retrieval.time_filter import (
    TimeRange,
    TimeRangeExtractor,
)

__all__ = [
    "FieldRetriever",
    "QueryType",
    "TimeAnchor",
    "TimeRange",
    "TimeRangeExtractor",
]
