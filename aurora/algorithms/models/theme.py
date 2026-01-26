"""
AURORA Theme Model
===================

Macroscale stable pattern (attractor) emerging from stories.
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
    Macroscale stable pattern (attractor) emerging from stories.

    Themes represent learned patterns, lessons, preferences, or causal
    relationships that emerge from accumulated narrative experience.

    Attributes:
        id: Unique identifier
        created_ts: Creation timestamp
        updated_ts: Last update timestamp
        story_ids: List of supporting story IDs
        prototype: Mean embedding representing this theme

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

    # Epistemic confidence as Beta posterior (evidence from applications)
    a: float = 1.0
    b: float = 1.0

    name: str = ""
    description: str = ""
    theme_type: Literal[
        "pattern", "lesson", "preference", "causality", "capability", "limitation"
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

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "story_ids": self.story_ids,
            "prototype": self.prototype.tolist() if self.prototype is not None else None,
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
            a=d.get("a", 1.0),
            b=d.get("b", 1.0),
            name=d.get("name", ""),
            description=d.get("description", ""),
            theme_type=d.get("theme_type", "pattern"),
        )
