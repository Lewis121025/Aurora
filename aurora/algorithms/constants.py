"""
AURORA Algorithm Constants
==========================

Centralized constants for the AURORA memory system.
All magic numbers are extracted here for maintainability and documentation.

Naming conventions:
- UPPERCASE_WITH_UNDERSCORES for constants
- Group by category with section comments
"""

from __future__ import annotations

# =============================================================================
# Identity & Relationship Weights
# =============================================================================

# Weights for identity relevance computation
REINFORCEMENT_WEIGHT = 0.4
CHALLENGE_WEIGHT = 0.5
NOVELTY_WEIGHT = 0.3

# Weights for storage decision (identity vs VOI)
IDENTITY_RELEVANCE_WEIGHT = 0.6
VOI_DECISION_WEIGHT = 0.4

# Relationship contribution weights
RELATIONSHIP_HEALTH_WEIGHT = 0.3
RELATIONSHIP_RECENCY_WEIGHT = 0.4
RELATIONSHIP_POSITION_WEIGHT = 0.3

# =============================================================================
# Similarity Thresholds
# =============================================================================

# For identity challenge detection
MODERATE_SIMILARITY_MIN = 0.3
MODERATE_SIMILARITY_MAX = 0.6

# For role consistency
ROLE_CONSISTENCY_THRESHOLD = 0.6

# For identity impact filtering
IDENTITY_RELEVANCE_THRESHOLD = 0.2

# For identity tension analysis
TENSION_SIMILARITY_MIN = 0.2
TENSION_SIMILARITY_MAX = 0.6
HARMONY_SIMILARITY_MIN = 0.7

# =============================================================================
# Capacity & Window Sizes
# =============================================================================

# Recent encoded plots window
RECENT_ENCODED_PLOTS_WINDOW = 200

# For retrieval
MAX_RECENT_PLOTS_FOR_RETRIEVAL = 50
RECENT_PLOTS_FOR_FEEDBACK = 20

# Identity dimensions limit
MAX_IDENTITY_DIMENSIONS = 3

# =============================================================================
# Time Thresholds (in days unless specified)
# =============================================================================

# For meaning reframe triggers
REFRAME_AGE_DAYS_THRESHOLD = 7
REFRAME_ACCESS_COUNT_THRESHOLD = 5

# For periodic reflection
PERIODIC_REFLECTION_AGE_DAYS = 30
PERIODIC_REFLECTION_ACCESS_COUNT = 3

# Growth hindrance check (seconds)
GROWTH_HINDRANCE_AGE_SECONDS = 30 * 24 * 3600  # 30 days

# =============================================================================
# Story Boundary Detection
# =============================================================================

# Abandonment threshold (days without activity)
STORY_ABANDONMENT_THRESHOLD_DAYS = 30.0

# Climax detection - tension peak window
CLIMAX_TENSION_WINDOW = 5
CLIMAX_DECLINE_RATIO = 0.3  # Peak must be at least 30% higher than current

# Resolution detection - conflict resolution threshold
RESOLUTION_TENSION_DROP_RATIO = 0.5  # Tension drops by at least 50%
RESOLUTION_MIN_ARC_LENGTH = 5  # Minimum arc length to consider resolution

# =============================================================================
# Graph Structure Cleanup
# =============================================================================

# Edge cleanup
WEAK_EDGE_MIN_WEIGHT = 0.1  # Edges below this weight are candidates for removal
WEAK_EDGE_MIN_SUCCESSES = 2  # Minimum successes to keep an edge

# Node merging
NODE_MERGE_SIMILARITY_THRESHOLD = 0.95  # Nodes above this similarity may be merged

# Archival
ARCHIVE_STALE_DAYS_THRESHOLD = 90.0  # Content unused for this long may be archived
ARCHIVE_MIN_ACCESS_COUNT = 0  # Content with access count <= this may be archived

# =============================================================================
# Numerical Stability
# =============================================================================

EPSILON = 1e-12
EPSILON_LOG = 1e-12
EPSILON_PRIOR = 1e-6

# =============================================================================
# Retrieval Parameters
# =============================================================================

# Initial search K values
INITIAL_SEARCH_K = 50
ATTRACTOR_SEARCH_K = 60
FEEDBACK_SEARCH_K = 10
NEGATIVE_SAMPLE_SEARCH_K = 30

# PageRank defaults
DEFAULT_PAGERANK_DAMPING = 0.85
DEFAULT_PAGERANK_MAX_ITER = 60
DEFAULT_MEAN_SHIFT_STEPS = 8

# Score bonuses
RELATIONSHIP_BONUS_SCORE = 0.2
STORY_SIMILARITY_BONUS = 0.3
MASS_BONUS_COEFFICIENT = 1e-3

# =============================================================================
# Text Processing
# =============================================================================

TEXT_LENGTH_NORMALIZATION = 512.0
SNIPPET_MAX_LENGTH = 240
EVENT_SUMMARY_MAX_LENGTH = 100
FALLBACK_ACTION_MAX_LENGTH = 120

# =============================================================================
# Quality Delta & Contribution
# =============================================================================

QUALITY_DELTA_COEFFICIENT = 0.1
MAX_QUALITY_DELTA = 0.3

# Identity dimension strengthening
IDENTITY_DIMENSION_GROWTH_RATE = 0.1

# Relationship importance factors
INTERACTION_COUNT_LOG_NORMALIZER = 3.0

# =============================================================================
# LLM Parameters
# =============================================================================

DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_TIMEOUT = 20.0

# =============================================================================
# Cache Parameters
# =============================================================================

DEFAULT_CACHE_TTL = 60.0
DEFAULT_CACHE_MAX_SIZE = 1000

# =============================================================================
# Graph Parameters
# =============================================================================

# Semantic neighbors
SEMANTIC_NEIGHBORS_K = 8

# Trust calculation
TRUST_BASE = 0.5

# =============================================================================
# Narrator Engine Parameters
# =============================================================================

# Perspective selection weights (prior preferences, learned from usage)
PERSPECTIVE_PRIOR_CHRONOLOGICAL = 1.0
PERSPECTIVE_PRIOR_RETROSPECTIVE = 0.8
PERSPECTIVE_PRIOR_CONTRASTIVE = 0.6
PERSPECTIVE_PRIOR_FOCUSED = 0.7
PERSPECTIVE_PRIOR_ABSTRACTED = 0.5

# Context recovery parameters
DEFAULT_CAUSAL_DEPTH = 3
MAX_CAUSAL_CHAIN_LENGTH = 10
TURNING_POINT_TENSION_THRESHOLD_BASE = 0.5  # Base, adjusted probabilistically

# Narrative coherence weights
TEMPORAL_COHERENCE_WEIGHT = 0.3
SEMANTIC_COHERENCE_WEIGHT = 0.4
CAUSAL_COHERENCE_WEIGHT = 0.3

# Narrative segment length
NARRATIVE_SEGMENT_MAX_LENGTH = 500
