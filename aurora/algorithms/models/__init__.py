"""
AURORA Data Models
==================

Core data structures for the AURORA memory system.

The memory model has three layers:
- Factual: What happened (Plot core fields)
- Relational: Who I am in this relationship (RelationalContext)
- Identity: How this affects who I am (IdentityImpact)
"""

from aurora.algorithms.models.plot import Plot, RelationalContext, IdentityImpact
from aurora.algorithms.models.story import StoryArc, RelationshipMoment
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.models.config import AlgorithmConfig, MemoryConfig
from aurora.algorithms.models.trace import (
    RetrievalTrace,
    EvolutionSnapshot,
    EvolutionPatch,
    QueryHit,
    KnowledgeTimeline,
    TimelineGroup,
)

__all__ = [
    # Plot and its layers
    "Plot",
    "RelationalContext",
    "IdentityImpact",
    # Story and its relationship structures
    "StoryArc",
    "RelationshipMoment",
    # Theme
    "Theme",
    # Config
    "AlgorithmConfig",
    "MemoryConfig",
    # Traces and Query Results
    "RetrievalTrace",
    "EvolutionSnapshot",
    "EvolutionPatch",
    "QueryHit",
    # Timeline-based retrieval (First Principles: superseded ≠ deleted)
    "KnowledgeTimeline",
    "TimelineGroup",
]
