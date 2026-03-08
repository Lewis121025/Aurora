from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from aurora.core.models.trace import QueryHit


@dataclass
class IngestResult:
    event_id: str
    plot_id: str
    story_id: Optional[str]
    encoded: bool
    tension: float
    surprise: float
    pred_error: float
    redundancy: float


@dataclass
class QueryResult:
    query: str
    attractor_path_len: int
    hits: List[QueryHit]


@dataclass
class CoherenceResult:
    overall_score: float
    conflict_count: int
    unfinished_story_count: int
    recommendations: List[str]
