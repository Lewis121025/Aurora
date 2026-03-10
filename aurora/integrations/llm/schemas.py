"""
aurora/integrations/llm/schemas.py
本模块定义了与大语言模型（LLM）交互时使用的各种结构化数据模型（Schema）。
利用 Pydantic 模型确保 LLM 返回的 JSON 数据符合预期的格式和类型约束。
主要涵盖了记忆提取、因果推理、身份反思以及 Generative Soul v4 的核心数据结构。
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


SCHEMA_VERSION = "1.0"


class Claim(BaseModel):
    """原子事实主张模型：描述“主语-谓语-宾语”结构的事实片段"""

    subject: str  # 主语（如：用户）
    predicate: str  # 谓语（如：喜欢）
    object: str  # 宾语（如：蓝色的衬衫）
    polarity: Literal["positive", "negative"] = "positive"  # 极性：肯定或否定
    certainty: float = Field(ge=0.0, le=1.0, default=0.7)  # 置信度 (0-1)
    # 可选限定符：描述时间、条件、位置等额外语境
    qualifiers: Dict[str, str] = Field(default_factory=dict)


class StoryUpdate(BaseModel):
    """故事更新模型：描述一段叙事弧的当前状态"""

    schema_version: str = SCHEMA_VERSION
    title: str  # 故事标题
    protagonist: str = ""  # 主角
    central_conflict: str = ""  # 核心冲突
    stage: Literal["setup", "rising", "climax", "falling", "resolution"] = "rising"  # 叙事阶段
    turning_points: List[str] = Field(default_factory=list)  # 关键转折点
    resolution: Optional[str] = None  # 结局/解决方式
    moral: Optional[str] = None  # 寓意/教训
    summary: str = ""  # 故事摘要


class ThemeCandidate(BaseModel):
    """主题候选模型：描述从多个故事中抽象出的规律或模式"""

    schema_version: str = SCHEMA_VERSION
    name: str  # 主题名称
    description: str  # 详细描述
    # 主题类型：模式、教训、偏好、因果、能力、限制
    theme_type: Literal["pattern", "lesson", "preference", "causality", "capability", "limitation"]
    falsification_conditions: List[str] = Field(default_factory=list)  # 证伪条件
    scope: str = "general"  # 适用范围（如：特定的用户、代理或全局）
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)  # 置信度


class ContradictionJudgement(BaseModel):
    """矛盾判定模型：判断两条信息之间是否存在冲突"""

    schema_version: str = SCHEMA_VERSION
    is_contradiction: bool  # 是否存在矛盾
    explanation: str = ""  # 矛盾的详细解释
    # 调和提示：例如“上下文相关”，意味着两者在不同条件下可能同时成立
    reconciliation_hint: str = ""


# -----------------------------------------------------------------------------
# 因果推理 Schema (Causal Reasoning)
# -----------------------------------------------------------------------------


class CausalRelation(BaseModel):
    """因果关系模型：描述两个事件之间的因果联系"""

    schema_version: str = SCHEMA_VERSION
    cause_id: str  # 原因事件 ID
    effect_id: str  # 结果事件 ID

    # 因果方向置信度 (0-1)
    direction_confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # 因果强度 (0-1)
    strength: float = Field(ge=0.0, le=1.0, default=0.5)

    # 因果关系类型：直接、间接、促成（Enabling）、阻止（Preventing）
    relation_type: Literal["direct", "indirect", "enabling", "preventing"] = "direct"

    # 此关系的证据描述
    evidence: str = ""

    # 此关系成立的条件列表
    conditions: List[str] = Field(default_factory=list)

    # 潜在的混淆因素（Confounders）
    confounders: List[str] = Field(default_factory=list)


class CausalChainExtraction(BaseModel):
    """因果链提取模型：从事件序列中识别出的完整因果路径"""

    schema_version: str = SCHEMA_VERSION

    # 链中的因果关系列表
    relations: List[CausalRelation] = Field(default_factory=list)

    # 识别的根本原因（Root Causes）
    root_causes: List[str] = Field(default_factory=list)

    # 识别的最终结果（Final Effects）
    final_effects: List[str] = Field(default_factory=list)

    # 整体链条的置信度
    chain_confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class CounterfactualQuery(BaseModel):
    """反事实推理查询模型：执行“如果当初...会怎样”的推理"""

    schema_version: str = SCHEMA_VERSION

    # 现实中的事实情况描述
    factual_description: str

    # 假设的改变条件（Antecedent）
    counterfactual_antecedent: str

    # 我们想探究的问题
    query: str

    # 推理出的反事实结果（Consequent）
    counterfactual_consequent: str = ""

    # 推理答案的置信度
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # 详细的推理链条描述
    reasoning: str = ""


# -----------------------------------------------------------------------------
# 自我叙述 Schema (Self-Narrative)
# -----------------------------------------------------------------------------


class CapabilityAssessment(BaseModel):
    """能力评估模型：从交互中评估 Agent 自身能力的表现"""

    schema_version: str = SCHEMA_VERSION

    capability_name: str  # 能力名称
    description: str = ""  # 能力描述

    # 能力展示证据（正面）
    demonstrated: bool = False
    demonstration_evidence: str = ""

    # 发现的局限性证据（负面）
    limitation_found: bool = False
    limitation_evidence: str = ""

    # 此能力适用的上下文环境
    applicable_contexts: List[str] = Field(default_factory=list)

    # 评估结果的置信度
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class RelationshipAssessment(BaseModel):
    """关系评估模型：从交互中评估 Agent 与外部实体（如用户）的关系质量"""

    schema_version: str = SCHEMA_VERSION

    entity_id: str  # 外部实体 ID
    entity_type: Literal["user", "system", "concept"] = "user"  # 实体类型

    # 交互质量是否为正向
    interaction_positive: bool = True

    # 信任信号值 (-1 to 1)
    trust_signal: float = Field(ge=-1.0, le=1.0, default=0.0)

    # 观察到的实体偏好字典
    preferences_observed: Dict[str, float] = Field(default_factory=dict)

    # 额外备注
    notes: str = ""


class IdentityReflection(BaseModel):
    """身份反思模型：Agent 对自身“数字灵魂”状态的深度总结"""

    schema_version: str = SCHEMA_VERSION

    # 当前身份状态的文字摘要
    identity_summary: str

    # 表现强劲的核心能力
    strong_capabilities: List[str] = Field(default_factory=list)
    # 正在发展中的能力
    developing_capabilities: List[str] = Field(default_factory=list)

    # 展示出的核心价值观
    values_demonstrated: List[str] = Field(default_factory=list)

    # 需要成长的领域
    growth_areas: List[str] = Field(default_factory=list)

    # 存在的内在冲突或张力
    tensions: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# 一致性 Schema (Coherence)
# -----------------------------------------------------------------------------


class CoherenceCheck(BaseModel):
    """一致性检查模型：检测两个认知元素之间的一致性结果"""

    schema_version: str = SCHEMA_VERSION

    element_a_id: str  # 元素 A 的 ID
    element_b_id: str  # 元素 B 的 ID

    # 潜在冲突的类型：事实、时间、因果、主题
    conflict_type: Optional[Literal["factual", "temporal", "causal", "thematic"]] = None

    # 是否存在冲突
    has_conflict: bool = False
    # 冲突严重程度 (0-1)
    conflict_severity: float = Field(ge=0.0, le=1.0, default=0.0)

    # 冲突解释
    explanation: str = ""

    # 建议的解决方案
    resolution_suggestion: str = ""

    # 在特定上下文中两者是否可以共存？
    contextually_compatible: bool = True
    # 兼容的条件描述
    compatibility_conditions: str = ""


# -----------------------------------------------------------------------------
# Generative Soul v4 Schema (核心生成式灵魂模型)
# -----------------------------------------------------------------------------


class PersonaAxisSpec(BaseModel):
    """人设维度规格：定义一个心理轴的极性、描述和权重"""

    name: str  # 轴名称
    positive_pole: str  # 正极标签
    negative_pole: str  # 负极标签
    description: str = ""  # 维度详细描述
    positive_examples: List[str] = Field(default_factory=list)  # 正极示例
    negative_examples: List[str] = Field(default_factory=list)  # 负极示例
    weight: float = Field(default=1.0, ge=0.1, le=3.0)  # 轴权重


class PersonaAxisPayload(BaseModel):
    """人设维度载荷：用于从人设文本中批量提取维度"""

    schema_version: str = SCHEMA_VERSION
    axes: List[PersonaAxisSpec] = Field(default_factory=list)


class MeaningFramePayloadV4(BaseModel):
    """意义框架载荷：描述单次交互在 V4 模型下的心理内涵"""

    schema_version: str = SCHEMA_VERSION
    # 命中的心理轴得分贡献字典
    axis_evidence: Dict[str, float] = Field(default_factory=dict)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)  # 效价
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)  # 唤醒度
    care: float = Field(default=0.0, ge=0.0, le=1.0)  # 被照顾感
    threat: float = Field(default=0.0, ge=0.0, le=1.0)  # 受威胁感
    control: float = Field(default=0.0, ge=0.0, le=1.0)  # 受控感
    abandonment: float = Field(default=0.0, ge=0.0, le=1.0)  # 被遗弃感
    agency_signal: float = Field(default=0.0, ge=0.0, le=1.0)  # 自主性信号
    shame: float = Field(default=0.0, ge=0.0, le=1.0)  # 羞耻感
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)  # 新颖度
    self_relevance: float = Field(default=0.5, ge=0.0, le=1.0)  # 自我相关性
    tags: List[str] = Field(default_factory=list)  # 提取的语义标签


class NarrativeSummaryPayloadV4(BaseModel):
    """叙事总结载荷：Agent 周期性生成的自我总结文本"""

    schema_version: str = SCHEMA_VERSION
    text: str = ""  # 总结正文
    current_mode: str = "origin"  # 当前身份模式名称
    salient_axes: List[str] = Field(default_factory=list)  # 最显著的心理轴列表


class RepairNarrationPayloadV4(BaseModel):
    """修复叙事载荷：当发生身份重构时，生成的自我解释文本"""

    schema_version: str = SCHEMA_VERSION
    text: str = ""  # 解释正文
    # 修复模式：坚持、重构、修正、分化、整合
    mode: Literal["preserve", "reframe", "revise", "differentiate", "integrate"] = "integrate"


class DreamNarrationPayloadV4(BaseModel):
    """梦境叙事载荷：Agent 在“离线演化”阶段生成的梦境描述"""

    schema_version: str = SCHEMA_VERSION
    text: str = ""  # 梦境正文
    # 梦境操作类型：反事实、整合、恐惧演练、愿望演练、混合
    operator: Literal[
        "counterfactual", "integration", "fear_rehearsal", "wish_rehearsal", "blend"
    ] = "blend"


class ModeLabelPayloadV4(BaseModel):
    """模式标签载荷：为涌现出的身份吸引子生成的简短名称"""

    schema_version: str = SCHEMA_VERSION
    label: str = "origin"  # 标签文本（如“防卫模式”）


class AxisMergeJudgementPayload(BaseModel):
    """轴合并判定载荷：决定两个相似的心理轴是否应当合并"""

    schema_version: str = SCHEMA_VERSION
    should_merge: bool = False  # 是否合并
    canonical_name: str = ""  # 合并后的规范名称
    alias_name: str = ""  # 被合并的别名
    rationale: str = ""  # 合并理由
