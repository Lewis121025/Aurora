"""
AURORA Plot 模型
==================

原子交互/事件记忆 - AURORA 记忆的基本单位。

增强了关系层和身份层：
- 第1层（事实层）：发生了什么（不可变，但可以被遗忘）
- 第2层（关系层）：关系上下文和含义
- 第3层（身份层）：对自我身份的影响（可随时间演变）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts


# -----------------------------------------------------------------------------
# 关系层和身份层数据结构
# -----------------------------------------------------------------------------


@dataclass
class RelationalContext:
    """
    Plot 的关系层 - 捕获"我在这段关系中是谁"。

    这是核心创新：记忆不仅仅是关于发生了什么，
    而是关于它对关系和我在其中的角色意味着什么。
    """

    with_whom: str  # 关系实体 ID
    my_role_in_relation: str  # "我在这段关系中是谁"
    relationship_quality_delta: float  # 对关系质量的影响 [-1, 1]
    what_this_says_about_us: str  # 关系含义的自然语言描述

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "with_whom": self.with_whom,
            "my_role_in_relation": self.my_role_in_relation,
            "relationship_quality_delta": self.relationship_quality_delta,
            "what_this_says_about_us": self.what_this_says_about_us,
        }

    # 向后兼容别名
    to_dict = to_state_dict

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "RelationalContext":
        """从状态字典重构。"""
        return cls(
            with_whom=d["with_whom"],
            my_role_in_relation=d.get("my_role_in_relation", "assistant"),
            relationship_quality_delta=d.get("relationship_quality_delta", 0.0),
            what_this_says_about_us=d.get("what_this_says_about_us", ""),
        )

    # 向后兼容别名
    from_dict = from_state_dict


@dataclass
class IdentityImpact:
    """
    Plot 的身份层 - 捕获这个经历如何影响"我是谁"。

    关键洞察：一个经历的含义可以随时间演变。
    看起来像失败的东西后来可能被理解为转折点。
    """

    when_formed: float  # 何时形成这个解释
    initial_meaning: str  # 初始理解
    current_meaning: str  # 当前理解（可以更新）
    identity_dimensions_affected: List[str]  # 哪些身份维度受到影响
    evolution_history: List[Tuple[float, str]]  # （时间戳，含义）演变历史

    def update_meaning(self, new_meaning: str) -> None:
        """更新当前含义并记录演变。"""
        if new_meaning != self.current_meaning:
            self.evolution_history.append((now_ts(), self.current_meaning))
            self.current_meaning = new_meaning

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "when_formed": self.when_formed,
            "initial_meaning": self.initial_meaning,
            "current_meaning": self.current_meaning,
            "identity_dimensions_affected": self.identity_dimensions_affected,
            "evolution_history": self.evolution_history,
        }

    # 向后兼容别名
    to_dict = to_state_dict

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "IdentityImpact":
        """从状态字典重构。"""
        return cls(
            when_formed=d["when_formed"],
            initial_meaning=d["initial_meaning"],
            current_meaning=d.get("current_meaning", d["initial_meaning"]),
            identity_dimensions_affected=d.get("identity_dimensions_affected", []),
            evolution_history=d.get("evolution_history", []),
        )

    # 向后兼容别名
    from_dict = from_state_dict


# -----------------------------------------------------------------------------
# Plot 模型
# -----------------------------------------------------------------------------


@dataclass
class Plot:
    """
    原子交互/事件记忆，包含三个层级：

    第1层 - 事实层（不可变）：
        发生了什么 - 客观事件记录。

    第2层 - 关系层（核心）：
        关系上下文 - "我在这段关系中是谁"。
        这是主要的组织维度。

    第3层 - 身份层（可演变）：
        身份影响 - 这如何影响"我是谁"。
        随着理解的深化可以演变。

    属性:
        id: 唯一标识符
        ts: Plot 创建时的时间戳
        text: 完整交互文本
        actors: 涉及的参与者标识符元组
        embedding: 交互的向量嵌入

    信号（在线计算，无固定混合权重）：
        surprise: OnlineKDE 下的 -log p(x)
        pred_error: 与最佳故事预测器的不匹配
        redundancy: 与现有 Plot 的最大相似度
        goal_relevance: 与查询/目标上下文的相似度
        tension: 自由能代理

    关系层:
        relational: 捕获关系含义的 RelationalContext

    身份层:
        identity_impact: 捕获自我身份影响的 IdentityImpact

    分配:
        story_id: 该 Plot 所属故事的 ID

    使用统计:
        access_count: 访问次数
        last_access_ts: 最后访问时间戳
        status: 当前状态（active、absorbed、archived）
    """

    # === 第1层：事实层（不可变，但可以被遗忘）===
    id: str
    ts: float
    text: str
    actors: Tuple[str, ...]
    embedding: np.ndarray

    # 在线计算的信号（无固定混合权重）
    surprise: float = 0.0
    pred_error: float = 0.0
    redundancy: float = 0.0
    goal_relevance: float = 0.0
    tension: float = 0.0

    # === 第2层：关系层（核心 - "我在这段关系中是谁"）===
    relational: Optional[RelationalContext] = None

    # === 第3层：身份层（可演变 - "这如何影响我是谁"）===
    identity_impact: Optional[IdentityImpact] = None

    # 分配
    story_id: Optional[str] = None
    source: Literal["interaction", "seed"] = "interaction"
    exposure: Literal["explicit", "shadow", "repressed"] = "explicit"

    # === 知识更新追踪 ===
    # 用于重新叙述：旧信息不被删除，而是重新定位为"过去的自我"
    supersedes_id: Optional[str] = None  # 该 Plot 替代/更新的 Plot 的 ID
    superseded_by_id: Optional[str] = None  # 替代该 Plot 的 Plot 的 ID
    update_type: Optional[Literal["state_change", "correction", "refinement"]] = None
    redundancy_type: Optional[Literal["novel", "update", "reinforcement", "pure_redundant"]] = None

    # === 知识类型分类 ===
    # 用于智能冲突解决（不是所有矛盾都需要消除）
    knowledge_type: Optional[
        Literal[
            "factual_state",  # 可变事实（地址、工作）- UPDATE 策略
            "factual_static",  # 不可变事实（生日）- CORRECT 策略
            "identity_trait",  # 性格特征 - PRESERVE_BOTH 策略
            "identity_value",  # 核心价值观 - PRESERVE_BOTH 策略
            "preference",  # 喜好/厌恶 - EVOLVE 策略
            "behavior",  # 行为模式 - EVOLVE 策略
            "unknown",  # 无法分类
        ]
    ] = None
    knowledge_confidence: float = 0.0  # 分类置信度 [0, 1]

    # === 第5阶段：事实增强索引 ===
    # 从文本中提取的事实键，用于多会话回忆增强
    fact_keys: List[str] = field(default_factory=list)  # 用于索引的提取事实字符串

    # 使用统计 -> "质量"涌现
    access_count: int = 0
    last_access_ts: float = field(default_factory=now_ts)
    status: Literal["active", "absorbed", "archived", "superseded", "corrected"] = "active"

    def mass(self) -> float:
        """
        涌现惯性：随访问频率增加而增加，随年龄增加而减少。

        返回:
            结合新鲜度和访问计数的质量值
        """
        age = max(1.0, now_ts() - self.ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(self.access_count + 1)

    def get_relationship_entity(self) -> Optional[str]:
        """从关系上下文获取关系实体 ID。"""
        return self.relational.with_whom if self.relational else None

    def get_my_role(self) -> str:
        """获取我在关系中的角色。"""
        return self.relational.my_role_in_relation if self.relational else "assistant"

    def has_identity_impact(self) -> bool:
        """检查该 Plot 是否有身份影响。"""
        return self.identity_impact is not None

    def get_identity_dimensions(self) -> List[str]:
        """获取该 Plot 影响的身份维度。"""
        if self.identity_impact:
            return self.identity_impact.identity_dimensions_affected
        return []

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        result = {
            "id": self.id,
            "ts": self.ts,
            "text": self.text,
            "actors": list(self.actors),
            "embedding": self.embedding.tolist(),
            "surprise": self.surprise,
            "pred_error": self.pred_error,
            "redundancy": self.redundancy,
            "goal_relevance": self.goal_relevance,
            "tension": self.tension,
            "story_id": self.story_id,
            "source": self.source,
            "exposure": self.exposure,
            "supersedes_id": self.supersedes_id,
            "superseded_by_id": self.superseded_by_id,
            "update_type": self.update_type,
            "redundancy_type": self.redundancy_type,
            "knowledge_type": self.knowledge_type,
            "knowledge_confidence": self.knowledge_confidence,
            "fact_keys": self.fact_keys,
            "access_count": self.access_count,
            "last_access_ts": self.last_access_ts,
            "status": self.status,
        }

        # 如果存在关系上下文则添加
        if self.relational is not None:
            result["relational"] = self.relational.to_state_dict()

        # 如果存在身份影响则添加
        if self.identity_impact is not None:
            result["identity_impact"] = self.identity_impact.to_state_dict()

        return result

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "Plot":
        """从状态字典重构。"""
        # 如果存在则解析关系上下文
        relational = None
        if "relational" in d and d["relational"] is not None:
            relational = RelationalContext.from_state_dict(d["relational"])

        # 如果存在则解析身份影响
        identity_impact = None
        if "identity_impact" in d and d["identity_impact"] is not None:
            identity_impact = IdentityImpact.from_state_dict(d["identity_impact"])

        return cls(
            id=d["id"],
            ts=d["ts"],
            text=d["text"],
            actors=tuple(d["actors"]),
            embedding=np.array(d["embedding"], dtype=np.float32),
            surprise=d.get("surprise", 0.0),
            pred_error=d.get("pred_error", 0.0),
            redundancy=d.get("redundancy", 0.0),
            goal_relevance=d.get("goal_relevance", 0.0),
            tension=d.get("tension", 0.0),
            relational=relational,
            identity_impact=identity_impact,
            story_id=d.get("story_id"),
            source=d.get("source", "interaction"),
            exposure=d.get("exposure", "explicit"),
            supersedes_id=d.get("supersedes_id"),
            superseded_by_id=d.get("superseded_by_id"),
            update_type=d.get("update_type"),
            redundancy_type=d.get("redundancy_type"),
            knowledge_type=d.get("knowledge_type"),
            knowledge_confidence=d.get("knowledge_confidence", 0.0),
            fact_keys=d.get("fact_keys", []),
            access_count=d.get("access_count", 0),
            last_access_ts=d.get("last_access_ts", now_ts()),
            status=d.get("status", "active"),
        )
