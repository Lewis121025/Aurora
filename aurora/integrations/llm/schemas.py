from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


SCHEMA_VERSION = "1.0"


class Claim(BaseModel):
    subject: str
    predicate: str
    object: str
    polarity: Literal["positive", "negative"] = "positive"
    certainty: float = Field(ge=0.0, le=1.0, default=0.7)
    # 可选限定符：时间、条件、位置
    qualifiers: Dict[str, str] = Field(default_factory=dict)


class PlotExtraction(BaseModel):
    schema_version: str = SCHEMA_VERSION
    actors: List[str] = Field(default_factory=list)
    action: str
    context: str = ""
    outcome: str = ""
    # 叙述信号
    goal: str = ""
    obstacles: List[str] = Field(default_factory=list)
    decision: str = ""
    emotion_valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    emotion_arousal: float = Field(ge=0.0, le=1.0, default=0.2)
    # 知识层
    claims: List[Claim] = Field(default_factory=list)


class StoryUpdate(BaseModel):
    schema_version: str = SCHEMA_VERSION
    title: str
    protagonist: str = ""
    central_conflict: str = ""
    stage: Literal["setup", "rising", "climax", "falling", "resolution"] = "rising"
    turning_points: List[str] = Field(default_factory=list)
    resolution: Optional[str] = None
    moral: Optional[str] = None
    summary: str = ""


class ThemeCandidate(BaseModel):
    schema_version: str = SCHEMA_VERSION
    name: str
    description: str
    theme_type: Literal["pattern", "lesson", "preference", "causality", "capability", "limitation"]
    falsification_conditions: List[str] = Field(default_factory=list)
    scope: str = "general"  # e.g. "user:123", "agent", "global"
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)


class SelfNarrativeUpdate(BaseModel):
    schema_version: str = SCHEMA_VERSION
    identity_statement: str
    identity_narrative: str
    capability_narrative: str
    relationship_narratives: Dict[str, str] = Field(default_factory=dict)
    core_beliefs: List[str] = Field(default_factory=list)
    unresolved_tensions: List[str] = Field(default_factory=list)


class ContradictionJudgement(BaseModel):
    schema_version: str = SCHEMA_VERSION
    is_contradiction: bool
    explanation: str = ""
    # e.g. "contextual" meaning both can be true under different conditions
    reconciliation_hint: str = ""


class MemoryBriefCompilation(BaseModel):
    schema_version: str = SCHEMA_VERSION
    known_facts: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)
    relationship_state: List[str] = Field(default_factory=list)
    active_narratives: List[str] = Field(default_factory=list)
    temporal_context: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# 因果推理 Schema
# -----------------------------------------------------------------------------

class CausalRelation(BaseModel):
    """事件之间的提取因果关系"""
    schema_version: str = SCHEMA_VERSION
    cause_id: str
    effect_id: str

    # 因果方向置信度 (0-1)
    direction_confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # 因果强度 (0-1)
    strength: float = Field(ge=0.0, le=1.0, default=0.5)

    # 因果关系类型
    relation_type: Literal["direct", "indirect", "enabling", "preventing"] = "direct"

    # 此关系的证据
    evidence: str = ""

    # 此关系成立的条件
    conditions: List[str] = Field(default_factory=list)

    # 潜在的混淆因素
    confounders: List[str] = Field(default_factory=list)


class CausalChainExtraction(BaseModel):
    """从事件序列提取因果链"""
    schema_version: str = SCHEMA_VERSION

    # 链中的因果关系列表
    relations: List[CausalRelation] = Field(default_factory=list)

    # 识别的根本原因
    root_causes: List[str] = Field(default_factory=list)

    # 识别的最终效果
    final_effects: List[str] = Field(default_factory=list)

    # 整体链置信度
    chain_confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class CounterfactualQuery(BaseModel):
    """反事实推理查询"""
    schema_version: str = SCHEMA_VERSION

    # 事实情况
    factual_description: str

    # 假设的变化
    counterfactual_antecedent: str

    # 我们想知道的
    query: str

    # 答案
    counterfactual_consequent: str = ""

    # 答案的置信度
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # 推理链
    reasoning: str = ""


# -----------------------------------------------------------------------------
# 自我叙述 Schema
# -----------------------------------------------------------------------------

class CapabilityAssessment(BaseModel):
    """从交互中评估能力"""
    schema_version: str = SCHEMA_VERSION

    capability_name: str
    description: str = ""

    # 能力证据（正面）
    demonstrated: bool = False
    demonstration_evidence: str = ""

    # 限制证据（负面）
    limitation_found: bool = False
    limitation_evidence: str = ""

    # 此能力适用的上下文
    applicable_contexts: List[str] = Field(default_factory=list)

    # 评估的置信度
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class RelationshipAssessment(BaseModel):
    """从交互中评估关系质量"""
    schema_version: str = SCHEMA_VERSION

    entity_id: str
    entity_type: Literal["user", "system", "concept"] = "user"

    # 交互质量
    interaction_positive: bool = True

    # 信任信号
    trust_signal: float = Field(ge=-1.0, le=1.0, default=0.0)

    # 学到的偏好
    preferences_observed: Dict[str, float] = Field(default_factory=dict)

    # 备注
    notes: str = ""


class IdentityReflection(BaseModel):
    """自我身份反思"""
    schema_version: str = SCHEMA_VERSION

    # 当前身份摘要
    identity_summary: str

    # 关键能力
    strong_capabilities: List[str] = Field(default_factory=list)
    developing_capabilities: List[str] = Field(default_factory=list)

    # 展示的价值观
    values_demonstrated: List[str] = Field(default_factory=list)

    # 成长领域
    growth_areas: List[str] = Field(default_factory=list)

    # 紧张或冲突
    tensions: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# 一致性 Schema
# -----------------------------------------------------------------------------

class CoherenceCheck(BaseModel):
    """两个元素之间的一致性检查结果"""
    schema_version: str = SCHEMA_VERSION

    element_a_id: str
    element_b_id: str

    # 潜在冲突的类型
    conflict_type: Optional[Literal["factual", "temporal", "causal", "thematic"]] = None

    # 是否存在冲突？
    has_conflict: bool = False
    conflict_severity: float = Field(ge=0.0, le=1.0, default=0.0)

    # 解释
    explanation: str = ""

    # 建议的解决方案
    resolution_suggestion: str = ""

    # 在不同条件下两者都能成立吗？
    contextually_compatible: bool = True
    compatibility_conditions: str = ""
