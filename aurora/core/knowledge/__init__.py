"""
AURORA 知识分类模块
==============================

将知识分类器、结果模型和规则模式统一收纳到该子包。
"""

from aurora.core.knowledge.classifier import (
    KnowledgeClassifier,
    classify_knowledge,
    resolve_knowledge_conflict,
)
from aurora.core.knowledge.models import (
    ClassificationResult,
    ConflictAnalysis,
    ConflictResolution,
    KnowledgeType,
)

__all__ = [
    "ClassificationResult",
    "ConflictAnalysis",
    "ConflictResolution",
    "KnowledgeClassifier",
    "KnowledgeType",
    "classify_knowledge",
    "resolve_knowledge_conflict",
]
