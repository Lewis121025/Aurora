"""
AURORA 知识分类模型
========================

知识类型、冲突策略和分析结果数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class KnowledgeType(Enum):
    """知识类型分类。"""

    FACTUAL_STATE = "factual_state"
    FACTUAL_STATIC = "factual_static"
    IDENTITY_TRAIT = "identity_trait"
    IDENTITY_VALUE = "identity_value"
    PREFERENCE = "preference"
    BEHAVIOR_PATTERN = "behavior"
    UNKNOWN = "unknown"


class ConflictResolution(Enum):
    """知识冲突的处理策略。"""

    UPDATE = "update"
    PRESERVE_BOTH = "preserve"
    CORRECT = "correct"
    EVOLVE = "evolve"
    NO_ACTION = "no_action"


@dataclass
class ClassificationResult:
    """知识分类结果。"""

    knowledge_type: KnowledgeType
    confidence: float
    matched_patterns: List[str] = field(default_factory=list)
    subject: Optional[str] = None
    predicate: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_type": self.knowledge_type.value,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "subject": self.subject,
            "predicate": self.predicate,
        }


@dataclass
class ConflictAnalysis:
    """两个知识片段之间的冲突分析。"""

    resolution: ConflictResolution
    rationale: str
    confidence: float
    knowledge_type_a: KnowledgeType
    knowledge_type_b: KnowledgeType
    is_complementary: bool = False
    requires_human_review: bool = False
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution.value,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "knowledge_type_a": self.knowledge_type_a.value,
            "knowledge_type_b": self.knowledge_type_b.value,
            "is_complementary": self.is_complementary,
            "requires_human_review": self.requires_human_review,
            "recommended_actions": self.recommended_actions,
        }
