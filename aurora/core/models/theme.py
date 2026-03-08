"""
AURORA 主题模型
===================

从故事中涌现的宏观稳定模式（吸引子）。

现在：主题作为身份维度 - 直接回答"我是谁"。

关键哲学转变：
- 旧：主题 = 从故事中提取的抽象模式
- 新：主题 = 身份维度，是"我是谁"的部分答案
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from aurora.utils.time_utils import now_ts


@dataclass
class Theme:
    """
    宏观稳定模式（吸引子）- 现在是身份维度。

    主题代表"我是谁"的答案，而不仅仅是抽象模式。
    每个主题是身份的一个维度，由关系证据支持。

    关键范式转变：
    - 旧："这是我观察到的一个模式"
    - 新："这是我是谁的一部分"

    属性：
        id: 唯一标识符
        created_ts: 创建时间戳
        updated_ts: 最后更新时间戳
        story_ids: 支持故事 ID 列表
        prototype: 代表此主题的平均嵌入

    身份维度：
        identity_dimension: 身份维度名称（例如，"作为解释者的我"）
        supporting_relationships: 为此维度提供证据的关系
        strength: 支持此维度的证据强度 [0, 1]

    功能矛盾管理：
        tensions_with: 与此维度有张力的其他身份维度
        harmonizes_with: 与此维度互补的其他身份维度

    认识论置信度（Beta 后验）：
        a: Alpha 参数（成功 + 1）
        b: Beta 参数（失败 + 1）

    元数据：
        name: 人类可读的名称
        description: 详细描述
        theme_type: 主题类型的分类
    """

    id: str
    created_ts: float
    updated_ts: float
    story_ids: List[str] = field(default_factory=list)
    prototype: Optional[np.ndarray] = None

    # === 身份维度（新 - 核心范式转变） ===
    identity_dimension: str = ""                                    # 例如，"作为解释者的我"
    supporting_relationships: List[str] = field(default_factory=list)  # 关系实体 ID
    strength: float = 0.5                                           # 证据有多强 [0, 1]

    # === 功能矛盾管理 ===
    tensions_with: List[str] = field(default_factory=list)          # 有张力的主题 ID
    harmonizes_with: List[str] = field(default_factory=list)        # 互补的主题 ID

    # 作为 Beta 后验的认识论置信度（来自应用的证据）
    a: float = 1.0
    b: float = 1.0

    name: str = ""
    description: str = ""
    theme_type: Literal[
        "pattern", "lesson", "preference", "causality", "capability", "limitation",
        "identity"  # 身份维度类型
    ] = "pattern"

    def confidence(self) -> float:
        """
        获取作为 Beta 后验平均值的置信度。

        返回：
            (0, 1) 中的置信度，其中更高意味着更多成功证据
        """
        return self.a / (self.a + self.b)

    def update_evidence(self, success: bool) -> None:
        """
        根据结果更新认识论置信度。

        参数：
            success: 主题应用是否成功
        """
        if success:
            self.a += 1.0
        else:
            self.b += 1.0
        self.updated_ts = now_ts()

    def mass(self) -> float:
        """
        主题级别的涌现重要性。

        返回：
            结合新鲜度、支持故事和置信度的质量值
        """
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(len(self.story_ids) + 1) * self.confidence()

    # ─────────────────────────────────────────────────────────────────────────
    # 身份维度方法
    # ─────────────────────────────────────────────────────────────────────────

    def is_identity_dimension(self) -> bool:
        """检查此主题是否代表身份维度。"""
        return bool(self.identity_dimension) or self.theme_type == "identity"

    def identity_strength(self) -> float:
        """
        获取此身份维度的强度。

        结合多个信号：
        - 证据强度 (a / (a + b))
        - 关系支持
        - 显式强度设置
        """
        evidence_strength = self.confidence()
        relationship_support = min(1.0, len(self.supporting_relationships) * 0.2)

        return 0.5 * evidence_strength + 0.3 * relationship_support + 0.2 * self.strength

    def add_supporting_relationship(self, relationship_entity: str) -> None:
        """添加支持此身份维度的关系。"""
        if relationship_entity not in self.supporting_relationships:
            self.supporting_relationships.append(relationship_entity)
            self._update_strength()

    def _update_strength(self) -> None:
        """根据支持证据更新强度。"""
        # 更多关系 = 更强的身份维度
        relationship_factor = min(1.0, len(self.supporting_relationships) * 0.15)
        story_factor = min(1.0, len(self.story_ids) * 0.1)

        self.strength = 0.5 * self.confidence() + 0.3 * relationship_factor + 0.2 * story_factor

    def add_tension(self, other_theme_id: str) -> None:
        """记录与另一个身份维度的张力。"""
        if other_theme_id not in self.tensions_with:
            self.tensions_with.append(other_theme_id)

    def add_harmony(self, other_theme_id: str) -> None:
        """记录与另一个身份维度的和谐。"""
        if other_theme_id not in self.harmonizes_with:
            self.harmonizes_with.append(other_theme_id)

    def has_significant_tensions(self) -> bool:
        """检查此维度是否有显著的未解决张力。"""
        return len(self.tensions_with) > 2

    def to_identity_narrative(self) -> str:
        """
        为此身份维度生成叙事。

        这直接回答"我是谁"的一部分。
        """
        if not self.identity_dimension:
            return f"我有一个特点：{self.name or self.description}"

        strength_desc = ""
        if self.identity_strength() > 0.7:
            strength_desc = "我坚定地"
        elif self.identity_strength() > 0.4:
            strength_desc = "我"
        else:
            strength_desc = "我正在成为"

        narrative = f"{strength_desc}是{self.identity_dimension}。"

        # 添加关系背景
        if self.supporting_relationships:
            rel_str = "、".join(self.supporting_relationships[:2])
            narrative += f"这在与{rel_str}的互动中得到了体现。"

        # 添加张力确认
        if self.tensions_with:
            narrative += "这与我其他的一些特质有时会产生张力，但这是健康的。"

        return narrative

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "story_ids": self.story_ids,
            "prototype": self.prototype.tolist() if self.prototype is not None else None,
            # 身份维度字段
            "identity_dimension": self.identity_dimension,
            "supporting_relationships": self.supporting_relationships,
            "strength": self.strength,
            "tensions_with": self.tensions_with,
            "harmonizes_with": self.harmonizes_with,
            # 原始字段
            "a": self.a,
            "b": self.b,
            "name": self.name,
            "description": self.description,
            "theme_type": self.theme_type,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "Theme":
        """从状态字典重建。"""
        prototype = d.get("prototype")
        return cls(
            id=d["id"],
            created_ts=d["created_ts"],
            updated_ts=d["updated_ts"],
            story_ids=d.get("story_ids", []),
            prototype=np.array(prototype, dtype=np.float32) if prototype is not None else None,
            # 身份维度字段
            identity_dimension=d.get("identity_dimension", ""),
            supporting_relationships=d.get("supporting_relationships", []),
            strength=d.get("strength", 0.5),
            tensions_with=d.get("tensions_with", []),
            harmonizes_with=d.get("harmonizes_with", []),
            # 原始字段
            a=d.get("a", 1.0),
            b=d.get("b", 1.0),
            name=d.get("name", ""),
            description=d.get("description", ""),
            theme_type=d.get("theme_type", "pattern"),
        )
