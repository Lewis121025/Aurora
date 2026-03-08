"""
AURORA 算法常数
==========================

AURORA 内存系统的集中式常数。
所有魔法数字都在此提取，用于可维护性和文档记录。

命名约定：
- 用于常数的大写加下划线
- 按类别和部分注释分组
"""

from __future__ import annotations

# =============================================================================
# 身份和关系权重
# =============================================================================

# 身份相关性计算的权重
# 强化帮助AR：熟悉主题可靠存储
REINFORCEMENT_WEIGHT = 0.35
# 挑战帮助TTL/CR：矛盾或新规则获得关注
CHALLENGE_WEIGHT = 0.45
# 新颖性帮助AR/LRU：保留新信息
NOVELTY_WEIGHT = 0.35

# 存储决策权重（身份 vs VOI）
# 基准优化：平衡身份和信息值
# - 较低的身份权重改进AR（事实信息更重要）
# - 较高的VOI权重改进CR（通过pred_error进行冲突检测）
IDENTITY_RELEVANCE_WEIGHT = 0.45
VOI_DECISION_WEIGHT = 0.55

# 关系贡献权重
RELATIONSHIP_HEALTH_WEIGHT = 0.3
RELATIONSHIP_RECENCY_WEIGHT = 0.4
RELATIONSHIP_POSITION_WEIGHT = 0.3

# =============================================================================
# 相似性阈值
# =============================================================================

# 用于身份挑战检测
MODERATE_SIMILARITY_MIN = 0.3
MODERATE_SIMILARITY_MAX = 0.6

# 用于角色一致性
ROLE_CONSISTENCY_THRESHOLD = 0.6

# 用于身份影响筛选
IDENTITY_RELEVANCE_THRESHOLD = 0.2

# 用于身份张力分析
TENSION_SIMILARITY_MIN = 0.2
TENSION_SIMILARITY_MAX = 0.6
HARMONY_SIMILARITY_MIN = 0.7

# =============================================================================
# 冷启动保护
# =============================================================================

# 无论VOI决策如何都强制存储的初始情节数
# 这可防止关键早期信息（名称、位置、首选项）丢失
COLD_START_FORCE_STORE_COUNT = 10

# 用于可靠密度估计的最小KDE样本
# 低于此阈值，使用默认惊喜值而不是KDE估计
DENSITY_MIN_SAMPLES = 5

# KDE样本不足时的默认惊喜值
# 设置为相对较高的值，以鼓励冷启动期间的存储
DEFAULT_COLD_START_SURPRISE = 8.0

# =============================================================================
# 存储门参数
# =============================================================================

# Thompson抽样门的最小存储概率下限
# 即使对低价值特征也确保基线存储率
# 使用 COLD_START_FORCE_STORE_COUNT=10 强制前10个情节
# MIN_STORE_PROB=0.6 对其余情节实现约73%的整体存储率
# 公式：(10*1.0 + 20*0.6) / 30 = 22/30 = 73.3% 预期
MIN_STORE_PROB = 0.6

# =============================================================================
# 容量和窗口大小
# =============================================================================

# 最近编码情节窗口
RECENT_ENCODED_PLOTS_WINDOW = 200

# 用于检索
MAX_RECENT_PLOTS_FOR_RETRIEVAL = 50
RECENT_PLOTS_FOR_FEEDBACK = 20

# 身份维度限制
MAX_IDENTITY_DIMENSIONS = 3

# =============================================================================
# 时间阈值（除非指定，否则以天为单位）
# =============================================================================

# 用于意义重新框架触发
REFRAME_AGE_DAYS_THRESHOLD = 7
REFRAME_ACCESS_COUNT_THRESHOLD = 5

# 用于定期反思
PERIODIC_REFLECTION_AGE_DAYS = 30
PERIODIC_REFLECTION_ACCESS_COUNT = 3

# 成长阻碍检查（秒）
GROWTH_HINDRANCE_AGE_SECONDS = 30 * 24 * 3600  # 30 天

# =============================================================================
# 故事边界检测
# =============================================================================

# 放弃阈值（无活动的天数）
STORY_ABANDONMENT_THRESHOLD_DAYS = 30.0

# 高潮检测 - 张力峰值窗口
CLIMAX_TENSION_WINDOW = 5
CLIMAX_DECLINE_RATIO = 0.3  # 峰值必须至少比当前高30%

# 分辨率检测 - 冲突分辨阈值
RESOLUTION_TENSION_DROP_RATIO = 0.5  # 张力下降至少50%
RESOLUTION_MIN_ARC_LENGTH = 5  # 考虑分辨率的最小弧长

# =============================================================================
# 图结构清理
# =============================================================================

# 边清理
WEAK_EDGE_MIN_WEIGHT = 0.1  # 权重低于此值的边是移除候选
WEAK_EDGE_MIN_SUCCESSES = 2  # 保持边的最小成功次数

# 节点合并
NODE_MERGE_SIMILARITY_THRESHOLD = 0.95  # 相似性高于此值的节点可能被合并

# 归档
ARCHIVE_STALE_DAYS_THRESHOLD = 90.0  # 未使用此长时间的内容可能被归档
ARCHIVE_MIN_ACCESS_COUNT = 0  # 访问计数 <= 此值的内容可能被归档

# =============================================================================
# 数值稳定性
# =============================================================================

EPSILON = 1e-12
EPSILON_LOG = 1e-12
EPSILON_PRIOR = 1e-6

# =============================================================================
# Query Type Detection Keywords
# =============================================================================

# Temporal query keywords (时序查询关键词)
# NOTE: Removed ambiguous words like "previous" that often appear in
# "previous conversation" context (factual query) rather than temporal queries.
TEMPORAL_KEYWORDS = {
    # Chinese
    "什么时候", "之前", "之后", "上次", "最近", "以前", "后来", "第一次", "最后", 
    "多久", "当时", "那时", "几点", "几月", "几号", "哪天", "哪年", "历史",
    # English
    "when", "before", "after", "first", "recently", "earlier", "later",
    "next", "yesterday", "today", "ago", "since", "until", "during",
    "history", "timeline", "chronological",
    # Removed: "previous", "last" - too often used in "previous chat" / "last time we talked" contexts
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

# Aggregation query keywords (聚合查询关键词)
# These indicate questions that require aggregating information across multiple sessions
# Examples: "How many books do I have?" requires counting across sessions
AGGREGATION_KEYWORDS = {
    # Chinese
    "多少", "几个", "总数", "总共", "合计", "一共", "累计", "汇总", "总计",
    "所有", "全部", "都", "每", "各",
    # English
    "how many", "how much", "total", "sum", "count", "all", "every", "each",
    "aggregate", "combined", "together", "altogether", "in total", "in all",
    "number of", "amount of", "quantity of",
}

# Query type retrieval parameters
MULTI_HOP_K_MULTIPLIER = 1.5  # Increase k for multi-hop queries
TEMPORAL_SORT_WEIGHT = 0.35   # Weight for timestamp in temporal ranking
                               # Reduced from 0.6 to preserve semantic relevance
                               # 0.35 means 65% semantic, 35% temporal
MULTI_HOP_EXTRA_PAGERANK_ITER = 20  # Additional PageRank iterations for multi-hop

# Aggregation query parameters
# Aggregation queries need more results to cover multiple sessions
# Multi-session questions often need to find ALL instances of a topic across many sessions
AGGREGATION_K_MULTIPLIER = 5.0  # Increase k by 5x for aggregation queries (was 3.0)

# =============================================================================
# Temporal Anchor Detection (时间锚点检测)
# =============================================================================

# Time as First-Class Citizen: Different time anchors require different retrieval strategies
# - RECENT: Return most recent memories first (descending by timestamp)
# - EARLIEST: Return earliest memories first (ascending by timestamp)  
# - SPAN: Return temporally diverse memories across the time range

# Recent time anchor keywords (最近/上次)
# NOTE: "previous" is NOT included because "previous conversation" is context reference,
# not a temporal query. "previous event" would need actual temporal handling.
RECENT_ANCHOR_KEYWORDS = {
    # Chinese
    "最近", "上次", "刚才", "刚刚", "近期", "这段时间", "最新", "最后",
    # English
    "recently", "last time", "just now", "lately", "latest", "most recent",
    "newest", "current",
    # Phrases with "last" that are clearly temporal
    "last thing", "last topic", "last event", "last item", "the last",
    # Phrases with "just" that indicate recency
    "just talked", "just mentioned", "just discussed", "just said",
}

# Earliest time anchor keywords (最早/第一次)
EARLIEST_ANCHOR_KEYWORDS = {
    # Chinese
    "最早", "一开始", "起初", "最初", "开始时", "第一次", "首次", "当初", "最先",
    # English
    "first", "originally", "initially", "earliest", "beginning", "started",
    "first time", "at first", "in the beginning", "original",
}

# Span time anchor keywords (时间跨度/历史)
SPAN_ANCHOR_KEYWORDS = {
    # Chinese
    "一直", "从...到", "之前...之后", "历史", "全部", "所有时候", "整个过程",
    "历程", "演变", "发展过程", "时间线", "变化",
    # English
    "throughout", "over time", "history", "timeline", "evolution",
    "all along", "from start", "progression", "across time", "journey",
    "before and after", "development", "changes over",
}

# Temporal diversity parameters for span queries
TEMPORAL_DIVERSITY_BUCKETS = 5  # Number of time buckets for diversity selection
TEMPORAL_DIVERSITY_MMR_LAMBDA = 0.5  # Balance between relevance and temporal diversity

# Factual query plot priority boost
# For FACTUAL queries, plots contain specific facts and should be ranked higher than
# aggregate structures (stories/themes) that may have similar semantic embeddings
# but lack the precise answer text
FACTUAL_PLOT_PRIORITY_BOOST = 0.15  # Additive boost for plots in factual queries

# Semantic weight for factual queries
# FACTUAL queries require precise semantic matching - PageRank should not distort rankings
# Higher value = more weight on direct semantic similarity vs graph structure
FACTUAL_SEMANTIC_WEIGHT = 0.90  # High semantic weight for factual precision

# Attractor weight for factual queries  
# Lower value = less mean-shift drift, preserving original query intent
FACTUAL_ATTRACTOR_WEIGHT = 0.25  # Reduced attractor influence for factual queries

# =============================================================================
# Knowledge Classification Constants
# =============================================================================

# Confidence thresholds for knowledge classification
KNOWLEDGE_CLASSIFICATION_MIN_CONFIDENCE = 0.5  # Below this, treat as UNKNOWN

# Semantic similarity thresholds for trait complementarity detection
COMPLEMENTARY_TRAIT_SIM_MIN = 0.2  # Traits with sim in [0.2, 0.7] may be complementary
COMPLEMENTARY_TRAIT_SIM_MAX = 0.7
CONTRADICTORY_TRAIT_SIM_THRESHOLD = -0.3  # Similarity below this suggests contradiction

# Weight for knowledge type in ingest decision
KNOWLEDGE_TYPE_WEIGHT_STATE = 0.7  # States are moderately important to store
KNOWLEDGE_TYPE_WEIGHT_STATIC = 0.9  # Static facts are very important
KNOWLEDGE_TYPE_WEIGHT_TRAIT = 0.8  # Traits are important for identity
KNOWLEDGE_TYPE_WEIGHT_VALUE = 0.95  # Values are most important
KNOWLEDGE_TYPE_WEIGHT_PREFERENCE = 0.6  # Preferences less critical
KNOWLEDGE_TYPE_WEIGHT_BEHAVIOR = 0.5  # Behaviors least critical

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
# Knowledge Update Detection
# =============================================================================

# Time gap threshold for potential update detection (in seconds)
# If time_gap > this threshold AND high semantic similarity, likely an update
UPDATE_TIME_GAP_THRESHOLD = 3600.0  # 1 hour

# Reinforcement time window (in seconds)
# Short time repetition = reinforcement, not update
REINFORCEMENT_TIME_WINDOW = 300.0  # 5 minutes

# Similarity thresholds for redundancy classification
# CRITICAL FIX: Knowledge updates typically have similarity 0.6-0.7 (not 0.75+)
# because "I live in Beijing" and "I moved to Shanghai" are semantically different
# but represent the SAME entity's state change. Lowering from 0.75 to 0.55
# allows proper update detection.
UPDATE_HIGH_SIMILARITY_THRESHOLD = 0.55  # Above this, check for update vs redundancy
UPDATE_MODERATE_SIMILARITY_THRESHOLD = 0.35  # Above this, may be reinforcement

# Update signal keywords (时态更新指示词)
UPDATE_KEYWORDS_ZH = {
    # State change indicators
    "现在", "已经", "换了", "搬到", "不再", "改成", "变成", "转为",
    "升级", "降级", "更新", "修改", "改为", "目前", "最新", "新的",
    # Temporal contrast
    "之前是", "以前是", "原来是", "曾经是", "过去是",
    # Correction indicators
    "其实", "实际上", "事实上", "纠正", "更正", "应该是",
}

UPDATE_KEYWORDS_EN = {
    # State change indicators
    "now", "already", "changed", "moved", "no longer", "switched", "updated",
    "upgraded", "downgraded", "modified", "currently", "latest", "new",
    "quit", "joined", "left", "started", "stopped", "became", "turned",
    "relocated", "transferred", "promoted", "demoted", "resigned",
    # Temporal contrast
    "used to", "was previously", "before was", "formerly", "previously",
    "but now", "these days", "nowadays", "at present", "as of now",
    # Correction indicators
    "actually", "in fact", "correction", "should be", "is now",
    "not anymore", "no more", "instead", "rather", "different now",
}

# Combined update keywords
UPDATE_KEYWORDS = UPDATE_KEYWORDS_ZH | UPDATE_KEYWORDS_EN

# Numeric value change detection pattern components
# Used to detect patterns like "5 -> 10" or "from 5 to 10"
NUMERIC_CHANGE_INDICATORS = {"->", "→", "=>", "变为", "改为", "from", "to"}

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

# =============================================================================
# Conflict Detection & Resolution (CoherenceGuardian Integration)
# =============================================================================

# Similarity threshold for triggering conflict check
# Only plots with similarity >= this are candidates for conflict detection
CONFLICT_CHECK_SIMILARITY_THRESHOLD = 0.65

# Number of similar plots to check for conflicts
CONFLICT_CHECK_K = 10

# Minimum contradiction probability to register as conflict
CONFLICT_PROBABILITY_THRESHOLD = 0.5

# Maximum number of conflicts to handle per ingest (for performance)
MAX_CONFLICTS_PER_INGEST = 5

# Weight for semantic vs knowledge-type in conflict resolution
SEMANTIC_CONFLICT_WEIGHT = 0.4
KNOWLEDGE_TYPE_CONFLICT_WEIGHT = 0.6

# Time threshold for considering plots as "concurrent" vs "sequential" (seconds)
CONCURRENT_TIME_THRESHOLD = 60.0  # 1 minute

# =============================================================================
# Benchmark Mode Constants
# =============================================================================

# In benchmark mode, we use larger k values to ensure comprehensive retrieval
# LongMemEval multi-session questions need aggregation across many turns
BENCHMARK_DEFAULT_K = 15          # Default k for benchmark queries
BENCHMARK_MULTI_SESSION_K = 30    # K for multi-session/multi-hop queries (was 20)
BENCHMARK_AGGREGATION_K = 50      # K when aggregation is detected (was 25)

# Maximum results to include in context for aggregation queries
# This is separate from retrieval k - we may retrieve 50 but include 20 in context
AGGREGATION_CONTEXT_MAX_RESULTS = 20

# =============================================================================
# Phase 5: Fact-Enhanced Indexing Constants
# =============================================================================

# Fact key matching boost for multi-session recall enhancement
# When a query fact matches a plot's fact_keys, boost the score
FACT_KEY_BOOST_MAX = 0.15  # Maximum boost score for fact key matches
FACT_KEY_MATCH_THRESHOLD = 0.7  # Minimum similarity for fact key matching (0.0-1.0)

# =============================================================================
# Single-Session-User Enhancement Constants
# =============================================================================

# For single-session-user questions, the semantic similarity between the question
# and the answer-containing content is often low (e.g., asking about "internet speed"
# when the user mentioned "500 Mbps" in a Netflix discussion).
# 
# Solution: Increase retrieval coverage and prioritize user statements.

# Retrieval k multiplier for single-session-user questions
# Higher k ensures more sessions are retrieved, increasing chance of finding the answer
SINGLE_SESSION_USER_K_MULTIPLIER = 2.0

# Maximum context length for single-session-user (longer to capture more user statements)
SINGLE_SESSION_USER_MAX_CONTEXT = 15000

# Boost score for plots containing extracted question keywords
# Applied to plots that match question entities (e.g., "internet", "speed", "plan")
KEYWORD_MATCH_BOOST = 0.20

# Minimum keyword match ratio to apply boost (0.0-1.0)
# If question has 3 keywords, need at least 1 to match (0.33)
KEYWORD_MATCH_MIN_RATIO = 0.25

# Role-based priority for single-session-user questions
# User statements are prioritized over assistant statements
USER_ROLE_PRIORITY_BOOST = 0.15

# Common stop words to exclude from keyword extraction
QUESTION_STOP_WORDS = {
    # English
    'what', 'where', 'when', 'how', 'why', 'who', 'which', 'whose', 'whom',
    'is', 'are', 'was', 'were', 'did', 'do', 'does', 'done', 'been', 'being',
    'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'my', 'your', 'i', 'you', 'me', 'we', 'they', 'it', 'our', 'their', 'its',
    'have', 'has', 'had', 'can', 'could', 'would', 'should', 'will', 'shall',
    'about', 'that', 'this', 'these', 'those', 'there', 'here',
    # Chinese
    '什么', '哪里', '哪个', '谁', '怎么', '为什么', '是', '的', '了', '吗', '我', '你',
}

# Question type hint mappings for benchmark integration
# Maps benchmark-provided question type strings to internal QueryType
QUESTION_TYPE_HINT_MAPPINGS = {
    # Single-session-user questions need USER_FACT treatment
    'single-session-user': 'USER_FACT',
    'single_session_user': 'USER_FACT',
    'singlesessionuser': 'USER_FACT',
    'user': 'USER_FACT',
    # Multi-session questions
    'multi-session': 'MULTI_HOP',
    'multi_session': 'MULTI_HOP',
    'multisession': 'MULTI_HOP',
    # Temporal questions
    'temporal-reasoning': 'TEMPORAL',
    'temporal_reasoning': 'TEMPORAL',
    'temporal': 'TEMPORAL',
    # Standard single-session assistant questions
    'single-session-assistant': 'FACTUAL',
    'single_session_assistant': 'FACTUAL',
}
