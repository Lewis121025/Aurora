"""
AURORA 叙述者模块
======================

用于内存重构和叙事生成的故事讲述引擎。

核心职责：
1. 故事重构：基于查询上下文重新组织内存片段
2. 视角选择：选择最优的叙事视角（概率性）
3. 上下文恢复：因果链追踪和转折点识别

设计原则：
- 零硬编码阈值：所有决策使用贝叶斯/随机策略
- 确定性可重现性：所有随机操作支持种子
- 完整的类型注解
- 可序列化状态

用法：
    from aurora.algorithms.narrator import NarratorEngine, NarrativePerspective

    narrator = NarratorEngine(metric=metric, seed=42)
    trace = narrator.reconstruct_story(query, plots)
"""

from aurora.algorithms.narrator.context import (
    ContextRecovery,
    TurningPointDetector,
)
from aurora.algorithms.narrator.perspective import (
    NarrativePerspective,
    NarrativeRole,
    PerspectiveOrganizer,
    PerspectiveScore,
    PerspectiveSelector,
)
from aurora.algorithms.narrator.reconstruction import (
    NarrativeElement,
    NarrativeTrace,
    NarratorEngine,
)

__all__ = [
    # Main engine
    "NarratorEngine",
    # Data classes
    "NarrativeElement",
    "NarrativeTrace",
    # Enums
    "NarrativePerspective",
    "NarrativeRole",
    # Perspective components
    "PerspectiveScore",
    "PerspectiveSelector",
    "PerspectiveOrganizer",
    # Context components
    "ContextRecovery",
    "TurningPointDetector",
]
