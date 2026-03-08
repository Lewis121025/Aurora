"""
知识分类相关配置常量
======================

控制知识类型分类和冲突解决的参数。
"""

# =============================================================================
# 知识分类置信度阈值
# =============================================================================

KNOWLEDGE_CLASSIFICATION_MIN_CONFIDENCE = 0.5

# =============================================================================
# 语义相似性阈值（用于特质互补性检测）
# =============================================================================

COMPLEMENTARY_TRAIT_SIM_MIN = 0.2
COMPLEMENTARY_TRAIT_SIM_MAX = 0.7
CONTRADICTORY_TRAIT_SIM_THRESHOLD = -0.3

# =============================================================================
# 知识类型在摄入决策中的权重
# =============================================================================

KNOWLEDGE_TYPE_WEIGHT_STATE = 0.7
KNOWLEDGE_TYPE_WEIGHT_STATIC = 0.9
KNOWLEDGE_TYPE_WEIGHT_TRAIT = 0.8
KNOWLEDGE_TYPE_WEIGHT_VALUE = 0.95
KNOWLEDGE_TYPE_WEIGHT_PREFERENCE = 0.6
KNOWLEDGE_TYPE_WEIGHT_BEHAVIOR = 0.5

# =============================================================================
# 知识更新检测
# =============================================================================

# 潜在更新检测的时间间隔阈值（秒）
UPDATE_TIME_GAP_THRESHOLD = 3600.0  # 1 小时

# 强化时间窗口（秒）
REINFORCEMENT_TIME_WINDOW = 300.0  # 5 分钟

# 冗余分类的相似性阈值
UPDATE_HIGH_SIMILARITY_THRESHOLD = 0.55
UPDATE_MODERATE_SIMILARITY_THRESHOLD = 0.35

# 更新信号关键词
UPDATE_KEYWORDS_ZH = {
    "现在", "已经", "换了", "搬到", "不再", "改成", "变成", "转为",
    "升级", "降级", "更新", "修改", "改为", "目前", "最新", "新的",
    "之前是", "以前是", "原来是", "曾经是", "过去是",
    "其实", "实际上", "事实上", "纠正", "更正", "应该是",
}

UPDATE_KEYWORDS_EN = {
    "now", "already", "changed", "moved", "no longer", "switched", "updated",
    "upgraded", "downgraded", "modified", "currently", "latest", "new",
    "quit", "joined", "left", "started", "stopped", "became", "turned",
    "relocated", "transferred", "promoted", "demoted", "resigned",
    "used to", "was previously", "before was", "formerly", "previously",
    "but now", "these days", "nowadays", "at present", "as of now",
    "actually", "in fact", "correction", "should be", "is now",
    "not anymore", "no more", "instead", "rather", "different now",
}

UPDATE_KEYWORDS = UPDATE_KEYWORDS_ZH | UPDATE_KEYWORDS_EN

# 数值变化检测模式组件
NUMERIC_CHANGE_INDICATORS = {"->", "→", "=>", "变为", "改为", "from", "to"}
