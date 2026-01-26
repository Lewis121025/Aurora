"""
AURORA Plot Model
==================

Atomic interaction/event memory - the fundamental unit of AURORA memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts


@dataclass
class Plot:
    """
    Atomic interaction/event memory.

    In production, this should be extracted from interaction using an LLM
    or a structured parser. Here we keep it generic and embed the full
    interaction text.

    Attributes:
        id: Unique identifier
        ts: Timestamp when the plot was created
        text: Full interaction text
        actors: Tuple of actor identifiers involved
        embedding: Vector embedding of the interaction

    Signals (computed online, no fixed mixing weights):
        surprise: -log p(x) under OnlineKDE
        pred_error: Mismatch with best story predictor
        redundancy: Max similarity to existing plots
        goal_relevance: Similarity to query/goal context
        tension: Free-energy proxy

    Assignment:
        story_id: ID of the story this plot belongs to

    Usage stats:
        access_count: Number of times accessed
        last_access_ts: Last access timestamp
        status: Current status (active, absorbed, archived)
    """

    id: str
    ts: float
    text: str
    actors: Tuple[str, ...]
    embedding: np.ndarray

    # Signals computed online (no fixed mixing weights)
    surprise: float = 0.0
    pred_error: float = 0.0
    redundancy: float = 0.0
    goal_relevance: float = 0.0
    tension: float = 0.0

    # Assignment
    story_id: Optional[str] = None

    # Usage stats -> "mass" emerges
    access_count: int = 0
    last_access_ts: float = field(default_factory=now_ts)
    status: Literal["active", "absorbed", "archived"] = "active"

    def mass(self) -> float:
        """
        Emergent inertia: increases with access frequency, decreases with age.

        Returns:
            Mass value combining freshness and access count
        """
        age = max(1.0, now_ts() - self.ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(self.access_count + 1)

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "ts": self.ts,
            "text": self.text,
            "actors": list(self.actors),
            "embedding": self.embedding.tolist(),
            "surprise": self.surprise,
            "pred_error": self.pred_error,
            "redundancy": self.redundancy,
            "goal_relevance": self.goal_relevance,
            "tension": self.tension,
            "story_id": self.story_id,
            "access_count": self.access_count,
            "last_access_ts": self.last_access_ts,
            "status": self.status,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "Plot":
        """Reconstruct from state dict."""
        return cls(
            id=d["id"],
            ts=d["ts"],
            text=d["text"],
            actors=tuple(d["actors"]),
            embedding=np.array(d["embedding"], dtype=np.float32),
            surprise=d.get("surprise", 0.0),
            pred_error=d.get("pred_error", 0.0),
            redundancy=d.get("redundancy", 0.0),
            goal_relevance=d.get("goal_relevance", 0.0),
            tension=d.get("tension", 0.0),
            story_id=d.get("story_id"),
            access_count=d.get("access_count", 0),
            last_access_ts=d.get("last_access_ts", now_ts()),
            status=d.get("status", "active"),
        )
