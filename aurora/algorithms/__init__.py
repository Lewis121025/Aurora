"""
AURORA Memory Algorithms
========================

Core algorithmic components for the AURORA memory system.

Modules:
- aurora_core: Main memory system (Plot, Story, Theme, retrieval, evolution)
- causal: Causal inference (direction discovery, intervention, counterfactuals)
- coherence: Coherence maintenance (contradiction detection, conflict resolution)
- self_narrative: Self-narrative management (identity, capabilities, relationships)
"""

from aurora.algorithms.aurora_core import (
    # Data structures
    Plot,
    StoryArc,
    Theme,
    EdgeBelief,
    RetrievalTrace,
    MemoryConfig,
    
    # Components
    HashEmbedding,
    OnlineKDE,
    LowRankMetric,
    ThompsonBernoulliGate,
    MemoryGraph,
    VectorIndex,
    CRPAssigner,
    StoryModel,
    ThemeModel,
    FieldRetriever,
    
    # Main system
    AuroraMemory,
    
    # Utilities
    now_ts,
    l2_normalize,
    cosine_sim,
    sigmoid,
    softmax,
)

from aurora.algorithms.causal import (
    # Data structures
    CausalEdgeBelief,
    InterventionResult,
    CounterfactualResult,
    
    # Components
    CausalDiscovery,
    InterventionEngine,
    CounterfactualReasoner,
    CausalMemoryGraph,
)

from aurora.algorithms.coherence import (
    # Data structures
    ConflictType,
    Conflict,
    Resolution,
    CoherenceReport,
    BeliefState,
    
    # Components
    BeliefNetwork,
    ContradictionDetector,
    CoherenceScorer,
    ConflictResolver,
    CoherenceGuardian,
)

from aurora.algorithms.self_narrative import (
    # Data structures
    CapabilityBelief,
    RelationshipBelief,
    SelfNarrative,
    
    # Components
    SelfNarrativeEngine,
    IdentityTracker,
)

__all__ = [
    # Core
    "Plot",
    "StoryArc", 
    "Theme",
    "EdgeBelief",
    "RetrievalTrace",
    "MemoryConfig",
    "HashEmbedding",
    "OnlineKDE",
    "LowRankMetric",
    "ThompsonBernoulliGate",
    "MemoryGraph",
    "VectorIndex",
    "CRPAssigner",
    "StoryModel",
    "ThemeModel",
    "FieldRetriever",
    "AuroraMemory",
    
    # Causal
    "CausalEdgeBelief",
    "InterventionResult",
    "CounterfactualResult",
    "CausalDiscovery",
    "InterventionEngine",
    "CounterfactualReasoner",
    "CausalMemoryGraph",
    
    # Coherence
    "ConflictType",
    "Conflict",
    "Resolution",
    "CoherenceReport",
    "BeliefState",
    "BeliefNetwork",
    "ContradictionDetector",
    "CoherenceScorer",
    "ConflictResolver",
    "CoherenceGuardian",
    
    # Self-Narrative
    "CapabilityBelief",
    "RelationshipBelief",
    "SelfNarrative",
    "SelfNarrativeEngine",
    "IdentityTracker",
    
    # Utilities
    "now_ts",
    "l2_normalize",
    "cosine_sim",
    "sigmoid",
    "softmax",
]
