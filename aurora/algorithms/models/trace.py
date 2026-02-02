"""
AURORA Trace and Snapshot Models
=================================

Data structures for retrieval traces and evolution snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from aurora.algorithms.retrieval.field_retriever import QueryType
    from aurora.algorithms.abstention import AbstentionResult


@dataclass
class RetrievalTrace:
    """
    Trace of a retrieval operation with relationship-centric and timeline extensions.

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
        
    Query type awareness:
        query_type: Detected or specified query type for adaptive retrieval
        
    Timeline extensions (First Principles):
        timeline_group: Organized timelines showing knowledge evolution
        include_historical: Whether historical (superseded) results are included
    """

    query: str
    query_emb: np.ndarray
    attractor_path: List[np.ndarray]
    ranked: List[Tuple[str, float, str]]  # (id, score, kind)
    
    # Relationship-centric extensions
    asker_id: Optional[str] = None
    activated_identity: Optional[str] = None
    relationship_context: Optional[str] = None
    
    # Query type awareness (use Any to avoid circular import at runtime)
    query_type: Optional[Any] = None  # QueryType enum
    
    # Timeline extensions - First Principles: preserve temporal evolution
    timeline_group: Optional[TimelineGroup] = None
    include_historical: bool = True  # By default, include full history
    
    # Abstention detection
    abstention: Optional[Any] = None  # AbstentionResult (use Any to avoid circular import)


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
class QueryHit:
    """
    A single retrieval hit.
    
    Represents one result from a memory query, containing
    the memory unit ID, type, relevance score, and a text snippet.
    
    Attributes:
        id: Unique identifier of the memory unit
        kind: Type of memory ("plot", "story", "theme")
        score: Relevance score (higher is better)
        snippet: Text snippet or summary of the content
        metadata: Optional additional metadata about the hit
    """
    id: str
    kind: str  # "plot", "story", "theme"
    score: float
    snippet: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeTimeline:
    """
    A timeline of knowledge evolution for a single topic/entity.
    
    First Principles:
    - In narrative psychology, past facts are not "deleted" but repositioned with past tense
    - "I lived in Beijing" → "I **used to** live in Beijing" (still true, just temporal)
    - superseded ≠ invalid; superseded = past truth
    
    This structure captures the complete evolution of knowledge, allowing the
    semantic understanding layer (LLM) to make decisions based on full context
    rather than having the retrieval layer arbitrarily filter results.
    
    Attributes:
        chain: List of plot IDs from oldest to newest (chronological order)
        current_id: ID of the currently active plot (None if all superseded)
        topic_signature: Semantic signature for grouping similar updates
        match_score: Best semantic match score from the query
        
    Example Timeline:
        chain: ["plot-001", "plot-002", "plot-003"]
        current_id: "plot-003"
        
        Represents:
        - plot-001: "I live in Beijing" (2024-01) → HISTORICAL
        - plot-002: "I moved to Shanghai" (2024-06) → HISTORICAL  
        - plot-003: "I moved to Shenzhen" (2024-12) → CURRENT
    """
    chain: List[str]                    # [oldest_plot_id, ..., newest_plot_id]
    current_id: Optional[str]           # ID of active plot (newest non-superseded)
    topic_signature: str                # Semantic signature for the topic
    match_score: float = 0.0            # Best match score from query
    
    def __len__(self) -> int:
        """Return the number of plots in the timeline."""
        return len(self.chain)
    
    def is_single_version(self) -> bool:
        """Check if this timeline has only one version (no updates)."""
        return len(self.chain) == 1
    
    def has_evolution(self) -> bool:
        """Check if this timeline shows knowledge evolution."""
        return len(self.chain) > 1
    
    def get_historical_ids(self) -> List[str]:
        """Get IDs of historical (superseded) plots."""
        if self.current_id is None:
            return self.chain  # All are historical
        return [pid for pid in self.chain if pid != self.current_id]


@dataclass
class TimelineGroup:
    """
    A group of related knowledge timelines from a retrieval operation.
    
    First Principles:
    - Retrieval should return structured temporal information
    - Let the LLM see full context with temporal markers
    - Don't filter at retrieval layer; let semantic understanding layer decide
    
    This structure organizes retrieval results into meaningful timelines,
    enabling temporal-reasoning queries ("Where did I used to live?") and
    knowledge-update queries ("Where do I live now?") to both work correctly.
    
    Attributes:
        timelines: List of KnowledgeTimeline objects
        standalone_results: Results that aren't part of any update chain
        total_results: Total number of unique plots across all timelines
    """
    timelines: List[KnowledgeTimeline] = field(default_factory=list)
    standalone_results: List[Tuple[str, float, str]] = field(default_factory=list)
    
    @property
    def total_results(self) -> int:
        """Total number of unique results."""
        timeline_count = sum(len(t.chain) for t in self.timelines)
        return timeline_count + len(self.standalone_results)
    
    def get_all_plot_ids(self) -> List[str]:
        """Get all plot IDs from timelines and standalone results."""
        ids: List[str] = []
        for timeline in self.timelines:
            ids.extend(timeline.chain)
        for nid, _, kind in self.standalone_results:
            if kind == "plot":
                ids.append(nid)
        return ids
    
    def get_current_state_ids(self) -> List[str]:
        """Get IDs representing current state (for knowledge-update queries)."""
        ids: List[str] = []
        for timeline in self.timelines:
            if timeline.current_id:
                ids.append(timeline.current_id)
        for nid, _, kind in self.standalone_results:
            ids.append(nid)
        return ids


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
