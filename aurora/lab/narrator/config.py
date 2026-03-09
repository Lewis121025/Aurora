"""
演化和反思相关配置常量
========================

控制记忆演化、故事边界检测和反思触发的参数。
"""

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
CLIMAX_DECLINE_RATIO = 0.3

# 分辨率检测 - 冲突分辨阈值
RESOLUTION_TENSION_DROP_RATIO = 0.5
RESOLUTION_MIN_ARC_LENGTH = 5

# =============================================================================
# 叙事引擎参数
# =============================================================================

# 视角选择权重（先验偏好，从使用中学习）
PERSPECTIVE_PRIOR_CHRONOLOGICAL = 1.0
PERSPECTIVE_PRIOR_RETROSPECTIVE = 0.8
PERSPECTIVE_PRIOR_CONTRASTIVE = 0.6
PERSPECTIVE_PRIOR_FOCUSED = 0.7
PERSPECTIVE_PRIOR_ABSTRACTED = 0.5

# 上下文恢复参数
DEFAULT_CAUSAL_DEPTH = 3
MAX_CAUSAL_CHAIN_LENGTH = 10
TURNING_POINT_TENSION_THRESHOLD_BASE = 0.5

# 叙事一致性权重
TEMPORAL_COHERENCE_WEIGHT = 0.3
SEMANTIC_COHERENCE_WEIGHT = 0.4
CAUSAL_COHERENCE_WEIGHT = 0.3

# 叙事片段长度
NARRATIVE_SEGMENT_MAX_LENGTH = 500
SNIPPET_MAX_LENGTH = 240
