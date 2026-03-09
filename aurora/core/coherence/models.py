"""
AURORA 连贯性数据模型
=========================

连贯性分析共享的枚举和数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from aurora.utils.time_utils import now_ts


class ConflictType(Enum):
    """连贯性冲突的类型。"""

    FACTUAL = "factual"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    THEMATIC = "thematic"
    SELF_NARRATIVE = "self"


@dataclass
class Resolution:
    """冲突的建议解决方案。"""

    strategy: str
    target_node: str
    action_description: str
    expected_coherence_gain: float
    cost: float
    condition: Optional[str] = None


@dataclass
class Conflict:
    """检测到的连贯性冲突。"""

    type: ConflictType
    node_a: str
    node_b: str
    severity: float
    confidence: float
    description: str
    evidence: List[str] = field(default_factory=list)
    resolutions: List[Resolution] = field(default_factory=list)


@dataclass
class CoherenceReport:
    """完整的连贯性分析报告。"""

    overall_score: float
    conflicts: List[Conflict]
    unfinished_stories: List[str]
    orphan_plots: List[str]
    factual_coherence: float
    temporal_coherence: float
    causal_coherence: float
    thematic_coherence: float
    recommended_actions: List[Resolution]


@dataclass
class BeliefState:
    """信念网络节点状态。"""

    prior: float
    evidence_strength: float
    last_updated: float = field(default_factory=now_ts)
