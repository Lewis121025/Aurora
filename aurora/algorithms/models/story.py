"""
AURORA StoryArc Model
======================

Mesoscale narrative unit - a self-organizing cluster of plots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from aurora.utils.time_utils import now_ts


@dataclass
class StoryArc:
    """
    Mesoscale narrative unit: a self-organizing cluster of plots.

    Stories emerge from plots through Chinese Restaurant Process clustering.
    They maintain online statistics for generative modeling.

    Attributes:
        id: Unique identifier
        created_ts: Creation timestamp
        updated_ts: Last update timestamp
        plot_ids: List of plot IDs belonging to this story

    Generative parameters:
        centroid: Mean embedding of plots in this story
        dist_mean, dist_m2, dist_n: Welford stats for semantic dispersion
        gap_mean, gap_m2, gap_n: Welford stats for temporal gaps

    Metadata:
        actor_counts: Count of each actor's appearances
        tension_curve: History of tension values
        status: Story lifecycle status
        reference_count: How often this story is referenced
    """

    id: str
    created_ts: float
    updated_ts: float
    plot_ids: List[str] = field(default_factory=list)

    # Online generative parameters
    centroid: Optional[np.ndarray] = None

    # Stats for semantic dispersion (Welford's algorithm)
    dist_mean: float = 0.0
    dist_m2: float = 0.0
    dist_n: int = 0

    # Stats for temporal gaps (Welford's algorithm)
    gap_mean: float = 0.0
    gap_m2: float = 0.0
    gap_n: int = 0

    actor_counts: Dict[str, int] = field(default_factory=dict)
    tension_curve: List[float] = field(default_factory=list)

    status: Literal["developing", "resolved", "abandoned"] = "developing"
    reference_count: int = 0

    def _update_stats(self, name: str, x: float) -> None:
        """
        Update running statistics using Welford's algorithm.

        Args:
            name: Either "dist" (distance) or "gap" (temporal gap)
            x: New observation value
        """
        if name == "dist":
            self.dist_n += 1
            delta = x - self.dist_mean
            self.dist_mean += delta / self.dist_n
            self.dist_m2 += delta * (x - self.dist_mean)
        elif name == "gap":
            self.gap_n += 1
            delta = x - self.gap_mean
            self.gap_mean += delta / self.gap_n
            self.gap_m2 += delta * (x - self.gap_mean)
        else:
            raise ValueError(f"Unknown stat name: {name}")

    def dist_var(self) -> float:
        """Get variance of distance statistics."""
        return self.dist_m2 / (self.dist_n - 1) if self.dist_n > 1 else 1.0

    def gap_mean_safe(self, default: float = 3600.0) -> float:
        """Get gap mean with safe default."""
        return self.gap_mean if self.gap_n > 0 and self.gap_mean > 0 else default

    def activity_probability(self, ts: Optional[float] = None) -> float:
        """
        Probability the story is still active under a learned temporal hazard model.

        If a story usually gets updates every ~gap_mean seconds, then being idle
        >> gap_mean should reduce activity probability smoothly (not via a fixed
        threshold).

        Args:
            ts: Current timestamp (defaults to now)

        Returns:
            Activity probability in (0, 1)
        """
        ts = ts or now_ts()
        idle = max(0.0, ts - self.updated_ts)
        tau = self.gap_mean_safe()
        # Survival function of exponential: P(active) ~ exp(-idle/tau)
        return math.exp(-idle / max(tau, 1e-6))

    def mass(self) -> float:
        """
        Emergent importance at story level.

        Returns:
            Mass value combining freshness, size, and references
        """
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        size = math.log1p(len(self.plot_ids))
        return freshness * (size + math.log1p(self.reference_count + 1))

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "plot_ids": self.plot_ids,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "dist_mean": self.dist_mean,
            "dist_m2": self.dist_m2,
            "dist_n": self.dist_n,
            "gap_mean": self.gap_mean,
            "gap_m2": self.gap_m2,
            "gap_n": self.gap_n,
            "actor_counts": self.actor_counts,
            "tension_curve": self.tension_curve,
            "status": self.status,
            "reference_count": self.reference_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "StoryArc":
        """Reconstruct from state dict."""
        centroid = d.get("centroid")
        return cls(
            id=d["id"],
            created_ts=d["created_ts"],
            updated_ts=d["updated_ts"],
            plot_ids=d.get("plot_ids", []),
            centroid=np.array(centroid, dtype=np.float32) if centroid is not None else None,
            dist_mean=d.get("dist_mean", 0.0),
            dist_m2=d.get("dist_m2", 0.0),
            dist_n=d.get("dist_n", 0),
            gap_mean=d.get("gap_mean", 0.0),
            gap_m2=d.get("gap_m2", 0.0),
            gap_n=d.get("gap_n", 0),
            actor_counts=d.get("actor_counts", {}),
            tension_curve=d.get("tension_curve", []),
            status=d.get("status", "developing"),
            reference_count=d.get("reference_count", 0),
        )
