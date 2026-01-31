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
# Reinforcement helps AR: familiar topics get stored reliably
REINFORCEMENT_WEIGHT = 0.35
# Challenge helps TTL/CR: contradictory or new rules get attention
CHALLENGE_WEIGHT = 0.45
# Novelty helps AR/LRU: new information is preserved
NOVELTY_WEIGHT = 0.35

# Weights for storage decision (identity vs VOI)
# Benchmark optimization: balance identity with information value
# - Lower identity weight improves AR (factual info matters more)
# - Higher VOI weight improves CR (conflict detection via pred_error)
IDENTITY_RELEVANCE_WEIGHT = 0.45
VOI_DECISION_WEIGHT = 0.55

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
# Cold Start Protection
# =============================================================================

# Number of initial plots to force-store regardless of VOI decision
# This prevents critical early information (names, locations, preferences) from being lost
COLD_START_FORCE_STORE_COUNT = 10

# Minimum KDE samples for reliable density estimation
# Below this threshold, use default surprise value instead of KDE estimate
DENSITY_MIN_SAMPLES = 5

# Default surprise value when KDE has insufficient samples
# Set to a moderately high value to encourage storage during cold start
DEFAULT_COLD_START_SURPRISE = 8.0

# =============================================================================
# Storage Gate Parameters
# =============================================================================

# Minimum storage probability floor for Thompson sampling gate
# Ensures a baseline storage rate even for low-value features
# With COLD_START_FORCE_STORE_COUNT=10 forcing first 10 plots,
# MIN_STORE_PROB=0.6 for remaining plots achieves ~73% overall storage rate
# Formula: (10*1.0 + 20*0.6) / 30 = 22/30 = 73.3% expected
MIN_STORE_PROB = 0.6

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
# Query Type Detection Keywords
# =============================================================================

# Temporal query keywords (时序查询关键词)
TEMPORAL_KEYWORDS = {
    # Chinese
    "什么时候", "之前", "之后", "上次", "最近", "以前", "后来", "第一次", "最后", 
    "多久", "当时", "那时", "几点", "几月", "几号", "哪天", "哪年", "历史",
    # English
    "when", "before", "after", "last", "first", "recently", "earlier", "later",
    "previous", "next", "yesterday", "today", "ago", "since", "until", "during",
    "history", "timeline", "chronological",
}

# Causal query keywords (因果查询关键词)
CAUSAL_KEYWORDS = {
    # Chinese
    "为什么", "原因", "因为", "所以", "导致", "结果", "因此", "由于", "怎么会",
    "为何", "何故", "缘由", "起因", "影响", "后果",
    # English
    "why", "because", "cause", "reason", "result", "therefore", "hence", 
    "consequently", "due to", "leads to", "effect", "impact", "outcome",
}

# Multi-hop query keywords (多跳查询关键词)
MULTI_HOP_KEYWORDS = {
    # Chinese
    "相关", "关联", "联系", "连接", "对比", "比较", "类似", "相似", "区别",
    "所有", "全部", "总结", "概括", "归纳", "涉及", "包含", "关系",
    # English
    "related", "connection", "link", "compare", "contrast", "similar", "difference",
    "all", "every", "summarize", "overview", "involve", "contain", "relationship",
    "between", "across", "through",
}

# Query type retrieval parameters
MULTI_HOP_K_MULTIPLIER = 1.5  # Increase k for multi-hop queries
TEMPORAL_SORT_WEIGHT = 0.3    # Weight for timestamp in temporal ranking
MULTI_HOP_EXTRA_PAGERANK_ITER = 20  # Additional PageRank iterations for multi-hop

# Factual query plot priority boost
# For FACTUAL queries, plots contain specific facts and should be ranked higher than
# aggregate structures (stories/themes) that may have similar semantic embeddings
# but lack the precise answer text
FACTUAL_PLOT_PRIORITY_BOOST = 0.15  # Additive boost for plots in factual queries

# =============================================================================
# Retrieval Parameters
# =============================================================================

# Initial search K values
# Benchmark optimization:
# - Higher initial K improves recall for AR
# - Moderate attractor K balances LRU breadth with AR precision
INITIAL_SEARCH_K = 60      # Increased from 50 for better AR recall
ATTRACTOR_SEARCH_K = 50    # Reduced from 60 to focus on precise matches
FEEDBACK_SEARCH_K = 15     # Increased from 10 for better credit assignment
NEGATIVE_SAMPLE_SEARCH_K = 40  # Increased from 30 for better metric learning

# PageRank defaults
# Benchmark optimization:
# - Slightly lower damping focuses more on direct semantic matches (AR)
# - Fewer mean-shift steps prevents over-smoothing for precise queries
DEFAULT_PAGERANK_DAMPING = 0.80    # Reduced from 0.85 for better AR
DEFAULT_PAGERANK_MAX_ITER = 50     # Reduced from 60, usually converges earlier
DEFAULT_MEAN_SHIFT_STEPS = 6       # Reduced from 8 to preserve query specificity

# Score bonuses
# Benchmark optimization:
# - Lower relationship bonus improves AR (pure semantic match matters more)
# - Lower story bonus prevents over-generalization for precise queries
# - These are additive bonuses, not multipliers
RELATIONSHIP_BONUS_SCORE = 0.12  # Reduced from 0.2 for better AR precision
STORY_SIMILARITY_BONUS = 0.18   # Reduced from 0.3 for better AR precision
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
# Benchmark optimization:
# - More neighbors helps LRU (broader context)
# - But not too many to avoid noise for AR
SEMANTIC_NEIGHBORS_K = 10  # Increased from 8 for better LRU coverage

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

# =============================================================================
# Coherence Module Constants
# =============================================================================

# Opposition/contradiction detection thresholds
# Benchmark optimization for CR:
# - Lower opposition threshold catches more subtle conflicts
# - Lower similarity threshold for detecting contradictions
OPPOSITION_SCORE_THRESHOLD = 0.25  # Reduced from 0.3 for better CR
HIGH_SIMILARITY_THRESHOLD = 0.65   # Reduced from 0.7 for broader conflict detection
ANTI_CORRELATION_THRESHOLD = -0.25 # Raised from -0.3 for more sensitive detection

# Unfinished story detection (hours)
UNFINISHED_STORY_HOURS = 72

# Sampling limit for coherence checks
MAX_COHERENCE_PAIRS = 500

# Belief propagation iterations
BELIEF_PROPAGATION_ITERATIONS = 10

# Weights for overall coherence score (geometric mean)
# Benchmark optimization:
# - Higher factual weight improves AR/CR accuracy
# - Higher causal weight improves LRU narrative coherence
COHERENCE_WEIGHTS = {
    "factual": 0.35,   # Increased from 0.3 for better AR/CR
    "temporal": 0.15,  # Reduced from 0.2
    "causal": 0.30,    # Increased from 0.25 for better LRU
    "thematic": 0.20,  # Reduced from 0.25
}
