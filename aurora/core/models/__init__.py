"""
AURORA 数据模型
==================

AURORA 记忆系统的核心数据结构。

记忆模型有三个层次：
- 事实层：发生了什么（Plot 核心字段）
- 关系层：我在这段关系中是谁（RelationalContext）
- 身份层：这如何影响我是谁（IdentityImpact）
"""

from aurora.core.models.plot import Plot, RelationalContext, IdentityImpact
from aurora.core.models.story import StoryArc, RelationshipMoment
from aurora.core.models.theme import Theme
from aurora.core.models.config import AlgorithmConfig, MemoryConfig
from aurora.core.models.trace import (
    RetrievalTrace,
    EvolutionSnapshot,
    EvolutionPatch,
    QueryHit,
    KnowledgeTimeline,
    TimelineGroup,
)

__all__ = [
    # Plot 及其层次
    "Plot",
    "RelationalContext",
    "IdentityImpact",
    # Story 及其关系结构
    "StoryArc",
    "RelationshipMoment",
    # 主题
    "Theme",
    # 配置
    "AlgorithmConfig",
    "MemoryConfig",
    # 追踪和查询结果
    "RetrievalTrace",
    "EvolutionSnapshot",
    "EvolutionPatch",
    "QueryHit",
    # 基于时间线的检索（第一性原理：被取代 ≠ 被删除）
    "KnowledgeTimeline",
    "TimelineGroup",
]
