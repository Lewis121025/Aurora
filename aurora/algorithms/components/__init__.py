"""
AURORA 可学习组件
============================

内存系统的核心可学习构建块。
"""

from aurora.algorithms.components.density import OnlineKDE
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.components.bandit import ThompsonBernoulliGate
from aurora.algorithms.components.assignment import CRPAssigner, StoryModel, ThemeModel
# HashEmbedding已移至aurora.embeddings.hash以获得更清晰的架构
# 为了向后兼容性在此重新导出
from aurora.embeddings.hash import HashEmbedding

__all__ = [
    "OnlineKDE",
    "LowRankMetric",
    "ThompsonBernoulliGate",
    "CRPAssigner",
    "StoryModel",
    "ThemeModel",
    "HashEmbedding",
]
