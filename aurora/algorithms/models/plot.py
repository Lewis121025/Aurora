"""
AURORA Plot Model
==================

Atomic interaction/event memory - the fundamental unit of AURORA memory.

Enhanced with relational and identity layers:
- Layer 1 (Factual): What happened (immutable, but can be forgotten)
- Layer 2 (Relational): The relational context and meaning
- Layer 3 (Identity): Impact on self-identity (can evolve over time)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts


# -----------------------------------------------------------------------------
# Relational and Identity Layer Data Structures
# -----------------------------------------------------------------------------

@dataclass
class RelationalContext:
    """
    The relational layer of a Plot - captures "who I am in this relationship".
    
    This is the core innovation: memory is not just about what happened,
    but about what it means for the relationship and my role in it.
    """
    with_whom: str                      # Relationship entity ID
    my_role_in_relation: str            # "Who I am in this relationship"
    relationship_quality_delta: float   # Impact on relationship quality [-1, 1]
    what_this_says_about_us: str        # Natural language description of relational meaning
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "with_whom": self.with_whom,
            "my_role_in_relation": self.my_role_in_relation,
            "relationship_quality_delta": self.relationship_quality_delta,
            "what_this_says_about_us": self.what_this_says_about_us,
        }
    
    # Backward compatibility alias
    to_dict = to_state_dict
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "RelationalContext":
        """Reconstruct from state dict."""
        return cls(
            with_whom=d["with_whom"],
            my_role_in_relation=d.get("my_role_in_relation", "assistant"),
            relationship_quality_delta=d.get("relationship_quality_delta", 0.0),
            what_this_says_about_us=d.get("what_this_says_about_us", ""),
        )
    
    # Backward compatibility alias
    from_dict = from_state_dict


@dataclass
class IdentityImpact:
    """
    The identity layer of a Plot - captures how this experience affects "who I am".
    
    Key insight: The meaning of an experience can evolve over time.
    What seemed like a failure may later be understood as a turning point.
    """
    when_formed: float                              # When this interpretation was formed
    initial_meaning: str                            # Initial understanding
    current_meaning: str                            # Current understanding (can be updated)
    identity_dimensions_affected: List[str]         # Which identity dimensions are affected
    evolution_history: List[Tuple[float, str]]      # (timestamp, meaning) evolution history
    
    def update_meaning(self, new_meaning: str) -> None:
        """Update the current meaning and record the evolution."""
        if new_meaning != self.current_meaning:
            self.evolution_history.append((now_ts(), self.current_meaning))
            self.current_meaning = new_meaning
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "when_formed": self.when_formed,
            "initial_meaning": self.initial_meaning,
            "current_meaning": self.current_meaning,
            "identity_dimensions_affected": self.identity_dimensions_affected,
            "evolution_history": self.evolution_history,
        }
    
    # Backward compatibility alias
    to_dict = to_state_dict
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "IdentityImpact":
        """Reconstruct from state dict."""
        return cls(
            when_formed=d["when_formed"],
            initial_meaning=d["initial_meaning"],
            current_meaning=d.get("current_meaning", d["initial_meaning"]),
            identity_dimensions_affected=d.get("identity_dimensions_affected", []),
            evolution_history=d.get("evolution_history", []),
        )
    
    # Backward compatibility alias
    from_dict = from_state_dict


# -----------------------------------------------------------------------------
# Plot Model
# -----------------------------------------------------------------------------

@dataclass
class Plot:
    """
    Atomic interaction/event memory with three layers:
    
    Layer 1 - Factual (immutable):
        What happened - the objective event record.
        
    Layer 2 - Relational (core):
        The relational context - "who I am in this relationship".
        This is the primary organizational dimension.
        
    Layer 3 - Identity (evolvable):
        The identity impact - how this affects "who I am".
        This can evolve as understanding deepens.

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

    Relational Layer:
        relational: RelationalContext capturing relationship meaning
        
    Identity Layer:
        identity_impact: IdentityImpact capturing self-identity effects

    Assignment:
        story_id: ID of the story this plot belongs to

    Usage stats:
        access_count: Number of times accessed
        last_access_ts: Last access timestamp
        status: Current status (active, absorbed, archived)
    """

    # === Layer 1: Factual (immutable, but can be forgotten) ===
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

    # === Layer 2: Relational (core - "who I am in this relationship") ===
    relational: Optional[RelationalContext] = None

    # === Layer 3: Identity (evolvable - "how this affects who I am") ===
    identity_impact: Optional[IdentityImpact] = None

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
    
    def get_relationship_entity(self) -> Optional[str]:
        """Get the relationship entity ID from relational context."""
        return self.relational.with_whom if self.relational else None
    
    def get_my_role(self) -> str:
        """Get my role in the relationship."""
        return self.relational.my_role_in_relation if self.relational else "assistant"
    
    def has_identity_impact(self) -> bool:
        """Check if this plot has identity impact."""
        return self.identity_impact is not None
    
    def get_identity_dimensions(self) -> List[str]:
        """Get the identity dimensions affected by this plot."""
        if self.identity_impact:
            return self.identity_impact.identity_dimensions_affected
        return []

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        result = {
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
        
        # Add relational context if present
        if self.relational is not None:
            result["relational"] = self.relational.to_state_dict()
        
        # Add identity impact if present
        if self.identity_impact is not None:
            result["identity_impact"] = self.identity_impact.to_state_dict()
        
        return result

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "Plot":
        """Reconstruct from state dict."""
        # Parse relational context if present
        relational = None
        if "relational" in d and d["relational"] is not None:
            relational = RelationalContext.from_state_dict(d["relational"])
        
        # Parse identity impact if present
        identity_impact = None
        if "identity_impact" in d and d["identity_impact"] is not None:
            identity_impact = IdentityImpact.from_state_dict(d["identity_impact"])
        
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
            relational=relational,
            identity_impact=identity_impact,
            story_id=d.get("story_id"),
            access_count=d.get("access_count", 0),
            last_access_ts=d.get("last_access_ts", now_ts()),
            status=d.get("status", "active"),
        )
