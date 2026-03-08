"""
AURORA 可学习组件
============================

内存系统的核心可学习构建块。
"""

from aurora.core.components.density import OnlineKDE
from aurora.core.components.metric import LowRankMetric
from aurora.core.components.bandit import ThompsonBernoulliGate
from aurora.core.components.assignment import CRPAssigner, StoryModel, ThemeModel
# HashEmbedding已移至aurora.integrations.embeddings.hash以获得更清晰的架构
# 为了向后兼容性在此重新导出
from aurora.integrations.embeddings.hash import HashEmbedding

__all__ = [
    "OnlineKDE",
    "LowRankMetric",
    "ThompsonBernoulliGate",
    "CRPAssigner",
    "StoryModel",
    "ThemeModel",
    "HashEmbedding",
]
