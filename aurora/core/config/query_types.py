"""
查询类型检测相关配置常量
==========================

控制查询类型分类和自适应检索策略的参数。
"""

# =============================================================================
# 查询类型检测关键词
# =============================================================================

# 时序查询关键词
TEMPORAL_KEYWORDS = {
    # Chinese
    "什么时候", "之前", "之后", "上次", "最近", "以前", "后来", "第一次", "最后",
    "多久", "当时", "那时", "几点", "几月", "几号", "哪天", "哪年", "历史",
    # English
    "when", "before", "after", "first", "recently", "earlier", "later",
    "next", "yesterday", "today", "ago", "since", "until", "during",
    "history", "timeline", "chronological",
}

# 因果查询关键词
CAUSAL_KEYWORDS = {
    # Chinese
    "为什么", "原因", "因为", "所以", "导致", "结果", "因此", "由于", "怎么会",
    "为何", "何故", "缘由", "起因", "影响", "后果",
    # English
    "why", "because", "cause", "reason", "result", "therefore", "hence",
    "consequently", "due to", "leads to", "effect", "impact", "outcome",
}

# 多跳查询关键词
MULTI_HOP_KEYWORDS = {
    # Chinese
    "相关", "关联", "联系", "连接", "对比", "比较", "类似", "相似", "区别",
    "所有", "全部", "总结", "概括", "归纳", "涉及", "包含", "关系",
    # English
    "related", "connection", "link", "compare", "contrast", "similar", "difference",
    "all", "every", "summarize", "overview", "involve", "contain", "relationship",
    "between", "across", "through",
}

# 聚合查询关键词
AGGREGATION_KEYWORDS = {
    # Chinese
    "多少", "几个", "总数", "总共", "合计", "一共", "累计", "汇总", "总计",
    "所有", "全部", "都", "每", "各",
    # English
    "how many", "how much", "total", "sum", "count", "all", "every", "each",
    "aggregate", "combined", "together", "altogether", "in total", "in all",
    "number of", "amount of", "quantity of",
}

# =============================================================================
# 查询类型检索参数
# =============================================================================

MULTI_HOP_K_MULTIPLIER = 1.5
TEMPORAL_SORT_WEIGHT = 0.35
MULTI_HOP_EXTRA_PAGERANK_ITER = 20
AGGREGATION_K_MULTIPLIER = 5.0

# =============================================================================
# 时间锚点检测
# =============================================================================

# 最近时间锚点关键词
RECENT_ANCHOR_KEYWORDS = {
    # Chinese
    "最近", "上次", "刚才", "刚刚", "近期", "这段时间", "最新", "最后",
    # English
    "recently", "last time", "just now", "lately", "latest", "most recent",
    "newest", "current",
    "last thing", "last topic", "last event", "last item", "the last",
    "just talked", "just mentioned", "just discussed", "just said",
}

# 最早时间锚点关键词
EARLIEST_ANCHOR_KEYWORDS = {
    # Chinese
    "最早", "一开始", "起初", "最初", "开始时", "第一次", "首次", "当初", "最先",
    # English
    "first", "originally", "initially", "earliest", "beginning", "started",
    "first time", "at first", "in the beginning", "original",
}

# 时间跨度锚点关键词
SPAN_ANCHOR_KEYWORDS = {
    # Chinese
    "一直", "从...到", "之前...之后", "历史", "全部", "所有时候", "整个过程",
    "历程", "演变", "发展过程", "时间线", "变化",
    # English
    "throughout", "over time", "history", "timeline", "evolution",
    "all along", "from start", "progression", "across time", "journey",
    "before and after", "development", "changes over",
}

# 时间多样性参数
TEMPORAL_DIVERSITY_BUCKETS = 5
TEMPORAL_DIVERSITY_MMR_LAMBDA = 0.5

# =============================================================================
# 事实查询优化
# =============================================================================

FACTUAL_PLOT_PRIORITY_BOOST = 0.15
FACTUAL_SEMANTIC_WEIGHT = 0.90
FACTUAL_ATTRACTOR_WEIGHT = 0.25

# =============================================================================
# 基准模式常量
# =============================================================================

BENCHMARK_DEFAULT_K = 15
BENCHMARK_MULTI_SESSION_K = 30
BENCHMARK_AGGREGATION_K = 50
AGGREGATION_CONTEXT_MAX_RESULTS = 20

# =============================================================================
# 单会话用户增强常量
# =============================================================================

SINGLE_SESSION_USER_K_MULTIPLIER = 2.0
SINGLE_SESSION_USER_MAX_CONTEXT = 15000
KEYWORD_MATCH_BOOST = 0.20
KEYWORD_MATCH_MIN_RATIO = 0.25
USER_ROLE_PRIORITY_BOOST = 0.15

# 停用词
QUESTION_STOP_WORDS = {
    'what', 'where', 'when', 'how', 'why', 'who', 'which', 'whose', 'whom',
    'is', 'are', 'was', 'were', 'did', 'do', 'does', 'done', 'been', 'being',
    'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'my', 'your', 'i', 'you', 'me', 'we', 'they', 'it', 'our', 'their', 'its',
    'have', 'has', 'had', 'can', 'could', 'would', 'should', 'will', 'shall',
    'about', 'that', 'this', 'these', 'those', 'there', 'here',
    '什么', '哪里', '哪个', '谁', '怎么', '为什么', '是', '的', '了', '吗', '我', '你',
}

# 问题类型提示映射
QUESTION_TYPE_HINT_MAPPINGS = {
    'single-session-user': 'USER_FACT',
    'single_session_user': 'USER_FACT',
    'singlesessionuser': 'USER_FACT',
    'user': 'USER_FACT',
    'multi-session': 'MULTI_HOP',
    'multi_session': 'MULTI_HOP',
    'multisession': 'MULTI_HOP',
    'temporal-reasoning': 'TEMPORAL',
    'temporal_reasoning': 'TEMPORAL',
    'temporal': 'TEMPORAL',
    'single-session-assistant': 'FACTUAL',
    'single_session_assistant': 'FACTUAL',
}

# =============================================================================
# 事实增强索引常量
# =============================================================================

FACT_KEY_BOOST_MAX = 0.15
FACT_KEY_MATCH_THRESHOLD = 0.7
