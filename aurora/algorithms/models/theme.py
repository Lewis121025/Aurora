"""
AURORA Theme Model
===================

Macroscale stable pattern (attractor) emerging from stories.

NOW: Theme as IDENTITY DIMENSION - directly answers "who I am".

Key philosophy change:
- Old: Theme = abstract pattern extracted from stories
- New: Theme = identity dimension, a partial answer to "who I am"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from aurora.utils.time_utils import now_ts


@dataclass
class Theme:
    """
    Macroscale stable pattern (attractor) - NOW AN IDENTITY DIMENSION.

    Themes represent answers to "who I am", not just abstract patterns.
    Each Theme is a dimension of identity, supported by relationship evidence.

    Key paradigm shift:
    - Old: "This is a pattern I've observed"
    - New: "This is part of who I am"

    Attributes:
        id: Unique identifier
        created_ts: Creation timestamp
        updated_ts: Last update timestamp
        story_ids: List of supporting story IDs
        prototype: Mean embedding representing this theme

    Identity Dimension:
        identity_dimension: The identity dimension name (e.g., "作为解释者的我")
        supporting_relationships: Relationships that provide evidence for this dimension
        strength: How much evidence supports this dimension [0, 1]

    Functional Contradiction Management:
        tensions_with: Other identity dimensions in tension with this one
        harmonizes_with: Other identity dimensions that complement this one

    Epistemic confidence (Beta posterior):
        a: Alpha parameter (successes + 1)
        b: Beta parameter (failures + 1)

    Metadata:
        name: Human-readable name
        description: Detailed description
        theme_type: Classification of theme type
    """

    id: str
    created_ts: float
    updated_ts: float
    story_ids: List[str] = field(default_factory=list)
    prototype: Optional[np.ndarray] = None

    # === Identity Dimension (NEW - core paradigm shift) ===
    identity_dimension: str = ""                                    # e.g., "作为解释者的我"
    supporting_relationships: List[str] = field(default_factory=list)  # Relationship entity IDs
    strength: float = 0.5                                           # How strong is the evidence [0, 1]

    # === Functional Contradiction Management ===
    tensions_with: List[str] = field(default_factory=list)          # Theme IDs in tension
    harmonizes_with: List[str] = field(default_factory=list)        # Theme IDs that complement

    # Epistemic confidence as Beta posterior (evidence from applications)
    a: float = 1.0
    b: float = 1.0

    name: str = ""
    description: str = ""
    theme_type: Literal[
        "pattern", "lesson", "preference", "causality", "capability", "limitation",
        "identity"  # Identity dimension type
    ] = "pattern"

    def confidence(self) -> float:
        """
        Get confidence level as Beta posterior mean.

        Returns:
            Confidence in (0, 1), where higher means more evidence of success
        """
        return self.a / (self.a + self.b)

    def update_evidence(self, success: bool) -> None:
        """
        Update epistemic confidence based on outcome.

        Args:
            success: Whether the theme application was successful
        """
        if success:
            self.a += 1.0
        else:
            self.b += 1.0
        self.updated_ts = now_ts()

    def mass(self) -> float:
        """
        Emergent importance at theme level.

        Returns:
            Mass value combining freshness, supporting stories, and confidence
        """
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(len(self.story_ids) + 1) * self.confidence()
    
    # -------------------------------------------------------------------------
    # Identity Dimension Methods
    # -------------------------------------------------------------------------
    
    def is_identity_dimension(self) -> bool:
        """Check if this theme represents an identity dimension."""
        return bool(self.identity_dimension) or self.theme_type == "identity"
    
    def identity_strength(self) -> float:
        """
        Get the strength of this identity dimension.
        
        Combines multiple signals:
        - Evidence strength (a / (a + b))
        - Relationship support
        - Explicit strength setting
        """
        evidence_strength = self.confidence()
        relationship_support = min(1.0, len(self.supporting_relationships) * 0.2)
        
        return 0.5 * evidence_strength + 0.3 * relationship_support + 0.2 * self.strength
    
    def add_supporting_relationship(self, relationship_entity: str) -> None:
        """Add a relationship that supports this identity dimension."""
        if relationship_entity not in self.supporting_relationships:
            self.supporting_relationships.append(relationship_entity)
            self._update_strength()
    
    def _update_strength(self) -> None:
        """Update strength based on supporting evidence."""
        # More relationships = stronger identity dimension
        relationship_factor = min(1.0, len(self.supporting_relationships) * 0.15)
        story_factor = min(1.0, len(self.story_ids) * 0.1)
        
        self.strength = 0.5 * self.confidence() + 0.3 * relationship_factor + 0.2 * story_factor
    
    def add_tension(self, other_theme_id: str) -> None:
        """Record a tension with another identity dimension."""
        if other_theme_id not in self.tensions_with:
            self.tensions_with.append(other_theme_id)
    
    def add_harmony(self, other_theme_id: str) -> None:
        """Record harmony with another identity dimension."""
        if other_theme_id not in self.harmonizes_with:
            self.harmonizes_with.append(other_theme_id)
    
    def has_significant_tensions(self) -> bool:
        """Check if this dimension has significant unresolved tensions."""
        return len(self.tensions_with) > 2
    
    def to_identity_narrative(self) -> str:
        """
        Generate a narrative for this identity dimension.
        
        This directly answers part of "who I am".
        """
        if not self.identity_dimension:
            return f"我有一个特点：{self.name or self.description}"
        
        strength_desc = ""
        if self.identity_strength() > 0.7:
            strength_desc = "我坚定地"
        elif self.identity_strength() > 0.4:
            strength_desc = "我"
        else:
            strength_desc = "我正在成为"
        
        narrative = f"{strength_desc}是{self.identity_dimension}。"
        
        # Add relationship context
        if self.supporting_relationships:
            rel_str = "、".join(self.supporting_relationships[:2])
            narrative += f"这在与{rel_str}的互动中得到了体现。"
        
        # Add tension acknowledgment
        if self.tensions_with:
            narrative += "这与我其他的一些特质有时会产生张力，但这是健康的。"
        
        return narrative

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "story_ids": self.story_ids,
            "prototype": self.prototype.tolist() if self.prototype is not None else None,
            # Identity dimension fields
            "identity_dimension": self.identity_dimension,
            "supporting_relationships": self.supporting_relationships,
            "strength": self.strength,
            "tensions_with": self.tensions_with,
            "harmonizes_with": self.harmonizes_with,
            # Original fields
            "a": self.a,
            "b": self.b,
            "name": self.name,
            "description": self.description,
            "theme_type": self.theme_type,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "Theme":
        """Reconstruct from state dict."""
        prototype = d.get("prototype")
        return cls(
            id=d["id"],
            created_ts=d["created_ts"],
            updated_ts=d["updated_ts"],
            story_ids=d.get("story_ids", []),
            prototype=np.array(prototype, dtype=np.float32) if prototype is not None else None,
            # Identity dimension fields
            identity_dimension=d.get("identity_dimension", ""),
            supporting_relationships=d.get("supporting_relationships", []),
            strength=d.get("strength", 0.5),
            tensions_with=d.get("tensions_with", []),
            harmonizes_with=d.get("harmonizes_with", []),
            # Original fields
            a=d.get("a", 1.0),
            b=d.get("b", 1.0),
            name=d.get("name", ""),
            description=d.get("description", ""),
            theme_type=d.get("theme_type", "pattern"),
        )
