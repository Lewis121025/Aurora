"""
AURORA Learnable Components
============================

Core learnable building blocks for the memory system.
"""

from aurora.algorithms.components.density import OnlineKDE
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.components.bandit import ThompsonBernoulliGate
from aurora.algorithms.components.assignment import CRPAssigner, StoryModel, ThemeModel
# HashEmbedding moved to aurora.embeddings.hash for cleaner architecture
# Re-export here for backward compatibility
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
