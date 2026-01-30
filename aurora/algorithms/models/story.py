"""
AURORA StoryArc Model
======================

Mesoscale narrative unit - now organized around RELATIONSHIPS, not just semantic clusters.

Key insight: Identity is defined in relationships. A Story is no longer just
"semantically similar events" but "the narrative of a relationship".

The primary organizational dimension is now `relationship_with`, not semantic similarity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts


# -----------------------------------------------------------------------------
# Relationship Trajectory Data Structure
# -----------------------------------------------------------------------------

@dataclass
class RelationshipMoment:
    """
    A moment in the relationship trajectory.
    
    Captures how the relationship evolved at a specific point in time.
    """
    ts: float                   # Timestamp
    event_summary: str          # Brief summary of what happened
    trust_level: float          # Trust level at this moment [0, 1]
    my_role: str                # My role in this moment
    quality_delta: float = 0.0  # Change in relationship quality
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "event_summary": self.event_summary,
            "trust_level": self.trust_level,
            "my_role": self.my_role,
            "quality_delta": self.quality_delta,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationshipMoment":
        return cls(
            ts=d["ts"],
            event_summary=d["event_summary"],
            trust_level=d.get("trust_level", 0.5),
            my_role=d.get("my_role", "assistant"),
            quality_delta=d.get("quality_delta", 0.0),
        )


# -----------------------------------------------------------------------------
# StoryArc Model
# -----------------------------------------------------------------------------

@dataclass
class StoryArc:
    """
    Mesoscale narrative unit: now a RELATIONSHIP NARRATIVE, not just a semantic cluster.

    Key paradigm shift:
    - Old: Stories group semantically similar plots
    - New: Stories represent relationships, and identity emerges within them

    The primary organizational dimension is `relationship_with`.
    "Who I am" is answered per-relationship via `my_identity_in_this_relationship`.

    Attributes:
        id: Unique identifier
        created_ts: Creation timestamp
        updated_ts: Last update timestamp
        plot_ids: List of plot IDs belonging to this story

    Relationship-centric:
        relationship_with: The entity ID this story is about (primary key)
        relationship_type: Type of relationship ("user", "system", "concept")
        relationship_arc: Trajectory of relationship moments
        my_identity_in_this_relationship: "Who I am in this relationship"
        lessons_from_relationship: What this relationship has taught me
        relationship_health: Current health/quality of the relationship [0, 1]

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

    # === Relationship-centric fields (NEW - primary organizational dimension) ===
    relationship_with: Optional[str] = None         # Entity ID (primary key for organization)
    relationship_type: str = "user"                 # "user", "system", "concept"
    relationship_arc: List[RelationshipMoment] = field(default_factory=list)
    my_identity_in_this_relationship: str = ""      # "Who I am in this relationship"
    lessons_from_relationship: List[str] = field(default_factory=list)
    relationship_health: float = 0.5                # Current relationship quality [0, 1]

    # Online generative parameters (kept for compatibility)
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
    
    # -------------------------------------------------------------------------
    # Relationship-centric Methods
    # -------------------------------------------------------------------------
    
    def is_relationship_story(self) -> bool:
        """Check if this story is organized around a relationship."""
        return self.relationship_with is not None
    
    def add_relationship_moment(
        self,
        event_summary: str,
        trust_level: float,
        my_role: str,
        quality_delta: float = 0.0,
        ts: Optional[float] = None
    ) -> None:
        """Add a moment to the relationship trajectory."""
        moment = RelationshipMoment(
            ts=ts or now_ts(),
            event_summary=event_summary,
            trust_level=trust_level,
            my_role=my_role,
            quality_delta=quality_delta,
        )
        self.relationship_arc.append(moment)
        
        # Update relationship health based on quality delta
        self.relationship_health = max(0.0, min(1.0, 
            self.relationship_health + quality_delta * 0.1
        ))
    
    def get_trust_trend(self, window: int = 10) -> float:
        """
        Get the trend of trust level over recent interactions.
        
        Returns:
            Positive if trust is increasing, negative if decreasing.
        """
        if len(self.relationship_arc) < 2:
            return 0.0
        
        recent = self.relationship_arc[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Simple linear trend
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        avg_first = sum(m.trust_level for m in first_half) / len(first_half)
        avg_second = sum(m.trust_level for m in second_half) / len(second_half)
        
        return avg_second - avg_first
    
    def get_role_consistency(self, window: int = 10) -> float:
        """
        Get the consistency of my role in this relationship.
        
        Returns:
            1.0 if perfectly consistent, lower if role varies.
        """
        if len(self.relationship_arc) < 2:
            return 1.0
        
        recent = self.relationship_arc[-window:]
        roles = [m.my_role for m in recent]
        
        # Count most common role
        role_counts: Dict[str, int] = {}
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        
        max_count = max(role_counts.values()) if role_counts else 0
        return max_count / len(roles) if roles else 1.0
    
    def update_identity_in_relationship(self, new_identity: str) -> None:
        """Update my identity in this relationship."""
        self.my_identity_in_this_relationship = new_identity
    
    def add_lesson(self, lesson: str) -> None:
        """Add a lesson learned from this relationship."""
        if lesson not in self.lessons_from_relationship:
            self.lessons_from_relationship.append(lesson)
    
    def get_current_trust(self) -> float:
        """Get the current trust level in this relationship."""
        if not self.relationship_arc:
            return 0.5  # Neutral default
        return self.relationship_arc[-1].trust_level
    
    def to_relationship_narrative(self) -> str:
        """Generate a natural language narrative of this relationship."""
        if not self.is_relationship_story():
            return f"Story {self.id} (not a relationship story)"
        
        parts = []
        
        # Relationship identity
        if self.my_identity_in_this_relationship:
            parts.append(f"在与 {self.relationship_with} 的关系中，我是{self.my_identity_in_this_relationship}。")
        
        # Trust and health
        trust = self.get_current_trust()
        if trust > 0.7:
            parts.append("我们建立了良好的信任关系。")
        elif trust > 0.4:
            parts.append("我们的关系正在发展中。")
        else:
            parts.append("我们的关系还需要培养。")
        
        # Lessons
        if self.lessons_from_relationship:
            lessons = "、".join(self.lessons_from_relationship[:3])
            parts.append(f"这段关系教会我：{lessons}。")
        
        return "".join(parts)

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "plot_ids": self.plot_ids,
            # Relationship-centric fields
            "relationship_with": self.relationship_with,
            "relationship_type": self.relationship_type,
            "relationship_arc": [m.to_dict() for m in self.relationship_arc],
            "my_identity_in_this_relationship": self.my_identity_in_this_relationship,
            "lessons_from_relationship": self.lessons_from_relationship,
            "relationship_health": self.relationship_health,
            # Generative parameters
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
        
        # Parse relationship arc
        relationship_arc = []
        if "relationship_arc" in d:
            relationship_arc = [
                RelationshipMoment.from_dict(m) for m in d["relationship_arc"]
            ]
        
        return cls(
            id=d["id"],
            created_ts=d["created_ts"],
            updated_ts=d["updated_ts"],
            plot_ids=d.get("plot_ids", []),
            # Relationship-centric fields
            relationship_with=d.get("relationship_with"),
            relationship_type=d.get("relationship_type", "user"),
            relationship_arc=relationship_arc,
            my_identity_in_this_relationship=d.get("my_identity_in_this_relationship", ""),
            lessons_from_relationship=d.get("lessons_from_relationship", []),
            relationship_health=d.get("relationship_health", 0.5),
            # Generative parameters
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
