"""
AURORA Trace and Snapshot Models
=================================

Data structures for retrieval traces and evolution snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RetrievalTrace:
    """
    Trace of a retrieval operation with relationship-centric extensions.

    Captures the query, intermediate states, and ranked results
    for analysis and learning.

    Attributes:
        query: Original query text
        query_emb: Query embedding vector
        attractor_path: Path of embeddings during mean-shift
        ranked: List of (id, score, kind) tuples for ranked results
        
    Relationship-centric extensions:
        asker_id: The ID of the entity asking the query
        activated_identity: The identity activated for this relationship
        relationship_context: Summary of relationship context
    """

    query: str
    query_emb: np.ndarray
    attractor_path: List[np.ndarray]
    ranked: List[Tuple[str, float, str]]  # (id, score, kind)
    
    # Relationship-centric extensions
    asker_id: Optional[str] = None
    activated_identity: Optional[str] = None
    relationship_context: Optional[str] = None


@dataclass
class EvolutionSnapshot:
    """
    Read-only snapshot of memory state for background evolution.

    Captures everything needed to compute evolution without modifying state.
    This enables concurrent evolution computation.

    Attributes:
        story_ids: List of all story IDs
        story_statuses: Mapping of story ID to status
        story_centroids: Mapping of story ID to centroid embedding
        story_tension_curves: Mapping of story ID to tension history
        story_updated_ts: Mapping of story ID to last update timestamp
        story_gap_means: Mapping of story ID to mean gap

        theme_ids: List of all theme IDs
        theme_story_counts: Mapping of theme ID to number of stories
        theme_prototypes: Mapping of theme ID to prototype embedding

        crp_theme_alpha: CRP concentration parameter for themes
        rng_state: Random number generator state for reproducibility
    """

    # Story data
    story_ids: List[str]
    story_statuses: Dict[str, str]
    story_centroids: Dict[str, Optional[np.ndarray]]
    story_tension_curves: Dict[str, List[float]]
    story_updated_ts: Dict[str, float]
    story_gap_means: Dict[str, float]

    # Theme data
    theme_ids: List[str]
    theme_story_counts: Dict[str, int]
    theme_prototypes: Dict[str, Optional[np.ndarray]]

    # CRP parameters
    crp_theme_alpha: float

    # RNG state for reproducibility
    rng_state: Dict[str, Any]


@dataclass
class EvolutionPatch:
    """
    Computed changes from evolution, to be applied atomically.

    Represents a diff that can be applied to the memory state
    after evolution computation completes.

    Attributes:
        status_changes: Mapping of story ID to new status
        theme_assignments: List of (story_id, theme_id) assignments
        new_themes: List of (theme_id, prototype) for new themes
    """

    # Story status changes: story_id -> new_status
    status_changes: Dict[str, str] = field(default_factory=dict)

    # Theme assignments: [(story_id, theme_id)]
    theme_assignments: List[Tuple[str, str]] = field(default_factory=list)

    # New themes to create: [(theme_id, prototype)]
    new_themes: List[Tuple[str, np.ndarray]] = field(default_factory=list)
