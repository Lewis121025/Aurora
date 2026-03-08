"""
AURORA 关系模块
==========================

关系识别、身份评估和关系背景提取。

主要职责：
- 从交互中识别关系实体
- 评估交互对身份的相关性
- 提取关系背景（"我在这段关系中的身份"）
- 提取身份影响（"这如何影响我的身份"）

哲学：记忆应该围绕关系组织，而不仅仅是语义相似性。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

from aurora.core.models.plot import IdentityImpact, Plot, RelationalContext
from aurora.core.models.story import StoryArc
from aurora.utils.id_utils import det_id
from aurora.utils.time_utils import now_ts
from aurora.core.config.identity import (
    CHALLENGE_WEIGHT,
    IDENTITY_DIMENSION_GROWTH_RATE,
    IDENTITY_RELEVANCE_THRESHOLD,
    INTERACTION_COUNT_LOG_NORMALIZER,
    MAX_IDENTITY_DIMENSIONS,
    MAX_QUALITY_DELTA,
    MODERATE_SIMILARITY_MAX,
    MODERATE_SIMILARITY_MIN,
    NOVELTY_WEIGHT,
    QUALITY_DELTA_COEFFICIENT,
    REINFORCEMENT_WEIGHT,
    ROLE_CONSISTENCY_THRESHOLD,
)

if TYPE_CHECKING:
    import numpy as np

class RelationshipMixin:
    """提供关系识别和身份评估功能的 Mixin。"""

    # -------------------------------------------------------------------------
    # 关系识别
    # -------------------------------------------------------------------------

    def _identify_relationship_entity(self, actors: Tuple[str, ...], text: str) -> str:
        """
        从此交互中识别主要关系实体。

        关键洞察：记忆应该围绕关系组织，而不仅仅是语义相似性。
        此方法识别此交互中的"另一方"是谁。

        参数：
            actors: 参与交互的演员元组
            text: 交互文本（用于上下文，目前未使用）

        返回：
            识别的关系实体名称
        """
        # 过滤掉代理/助手本身
        others = [a for a in actors if a.lower() not in ("agent", "assistant", "ai", "system")]

        if others:
            # 返回第一个非自身演员作为关系实体
            return others[0]

        # 如果没有明确的另一方，使用"user"作为默认值
        return "user"

    # -------------------------------------------------------------------------
    # 身份相关性评估
    # -------------------------------------------------------------------------

    def _assess_identity_relevance(
        self, text: str, relationship_entity: str, emb: "np.ndarray"
    ) -> float:
        """
        评估此交互与"我是谁"的相关性。

        这用身份相关性替代了纯信息论的 VOI。
        问题不是"这是否令人惊讶？"而是"这是否影响我的身份？"

        优化：单次遍历计算强化、挑战和新颖性。

        参数：
            text: 交互文本
            relationship_entity: 识别的关系实体
            emb: 交互的嵌入

        返回：
            0 到 1 之间的身份相关性分数
        """
        # 在一次遍历中计算所有身份信号
        reinforcement, challenge, novelty = self._compute_identity_signals(text, emb)

        relevance = (
            reinforcement * REINFORCEMENT_WEIGHT +
            challenge * CHALLENGE_WEIGHT +  # 挑战更重要，需要记住
            novelty * NOVELTY_WEIGHT
        )

        # 关系因素：重要关系中的交互更重要
        relationship_importance = self._get_relationship_importance(relationship_entity)
        relevance *= (0.5 + 0.5 * relationship_importance)

        return min(1.0, relevance)

    def _compute_identity_signals(
        self, text: str, emb: "np.ndarray"
    ) -> Tuple[float, float, float]:
        """
        在单次遍历主题中计算身份信号。

        此优化方法在单次遍历所有主题时计算强化、挑战和新颖性，
        避免冗余计算。

        参数：
            text: 交互文本
            emb: 交互的嵌入

        返回：
            (强化、挑战、新颖性) 分数的元组
        """
        # 边界情况的早期返回
        if not self.themes:
            return 0.0, 0.0, 1.0  # 没有主题 = 一切都是新颖的

        # 单次遍历所有主题
        max_sim = 0.0
        min_sim = 1.0
        max_moderate_sim = 0.0

        for theme in self.themes.values():
            if theme.prototype is None:
                continue
            sim = self.metric.sim(emb, theme.prototype)

            # 对于强化：最大相似性
            max_sim = max(max_sim, sim)

            # 对于新颖性：最小相似性（稍后反转）
            min_sim = min(min_sim, sim)

            # 对于挑战：中等范围内的最大相似性
            if MODERATE_SIMILARITY_MIN < sim < MODERATE_SIMILARITY_MAX:
                max_moderate_sim = max(max_moderate_sim, sim)

        # 强化：此交互强化现有身份的程度
        reinforcement = max_sim if self._identity_dimensions else 0.0

        # 新颖性：与最大相似性成反比
        novelty = max(0.0, 1.0 - max_sim)

        # 挑战：高惊讶度 + 中等相似性
        challenge = 0.0
        if max_moderate_sim > 0:
            surprise = float(self.kde.surprise(emb))
            if surprise > 0:
                challenge = min(1.0, surprise * 0.3 * max_moderate_sim)

        return reinforcement, challenge, novelty

    def _get_relationship_importance(self, relationship_entity: str) -> float:
        """
        根据历史获取关系的重要性。

        参数：
            relationship_entity: 要检查重要性的实体

        返回：
            0 到 1 之间的重要性分数
        """
        story_id = self._relationship_story_index.get(relationship_entity)
        if story_id is None:
            return 0.5  # 新关系的中立值

        story = self.stories.get(story_id)
        if story is None:
            return 0.5

        # 重要性基于：交互数量 + 健康度
        interaction_count = len(story.plot_ids)
        health = story.relationship_health

        # 更多交互 = 更重要的关系
        count_factor = min(1.0, math.log1p(interaction_count) / INTERACTION_COUNT_LOG_NORMALIZER)

        return 0.3 * count_factor + 0.7 * health

    # -------------------------------------------------------------------------
    # 关系背景提取
    # -------------------------------------------------------------------------

    def _extract_relational_context(
        self,
        text: str,
        relationship_entity: str,
        actors: Tuple[str, ...],
        identity_relevance: float
    ) -> RelationalContext:
        """
        从此交互中提取关系意义。

        关键问题："我在这段关系中是谁？"

        参数：
            text: 交互文本
            relationship_entity: 识别的关系实体
            actors: 交互中的所有演员
            identity_relevance: 之前计算的身份相关性分数

        返回：
            描述关系意义的 RelationalContext
        """
        # 根据交互内容和关系历史确定我的角色
        my_role = self._infer_my_role(text, relationship_entity)

        # 估计关系质量影响
        quality_delta = self._compute_quality_delta(text, identity_relevance)

        # 生成简短的关系意义描述（生产环境中可使用 LLM）
        meaning = self._generate_relational_meaning(text, my_role, quality_delta)

        return RelationalContext(
            with_whom=relationship_entity,
            my_role_in_relation=my_role,
            relationship_quality_delta=quality_delta,
            what_this_says_about_us=meaning,
        )

    def _infer_my_role(self, text: str, relationship_entity: Optional[str] = None) -> str:
        """
        推断我在此交互中的角色。

        增强功能使用关系历史（如果可用）：
        - 如果有既定历史，优先使用该关系中的一致角色
        - 否则，使用基于关键词的推断

        参数：
            text: 交互文本
            relationship_entity: 用于上下文的可选关系实体

        返回：
            推断的角色字符串
        """
        # 检查是否有关系历史
        if relationship_entity:
            story = self.get_relationship_story(relationship_entity)
            if story and story.my_identity_in_this_relationship:
                # 如果角色一致性高，使用既定身份
                if story.get_role_consistency(window=5) > ROLE_CONSISTENCY_THRESHOLD:
                    return story.my_identity_in_this_relationship

        # 回退到基于关键词的推断
        return self._keyword_based_role(text)

    def _keyword_based_role(self, text: str) -> str:
        """
        基于关键词的角色推断。

        参数：
            text: 交互文本

        返回：
            基于关键词推断的角色字符串
        """
        text_lower = text.lower()

        # 简单的基于启发式的角色推断
        if any(kw in text_lower for kw in ["解释", "explain", "说明", "clarify"]):
            return "解释者"
        elif any(kw in text_lower for kw in ["帮助", "help", "协助", "assist"]):
            return "帮助者"
        elif any(kw in text_lower for kw in ["分析", "analyze", "评估", "evaluate"]):
            return "分析者"
        elif any(kw in text_lower for kw in ["代码", "code", "编程", "program"]):
            return "编程助手"
        elif any(kw in text_lower for kw in ["计划", "plan", "规划", "design"]):
            return "规划者"
        elif any(kw in text_lower for kw in ["学习", "learn", "理解", "understand"]):
            return "学习伙伴"
        elif any(kw in text_lower for kw in ["创作", "create", "写", "write"]):
            return "创作伙伴"
        else:
            return "助手"

    def _compute_quality_delta(self, text: str, identity_relevance: float) -> float:
        """
        估计此交互如何影响关系质量。

        参数：
            text: 交互文本
            identity_relevance: 身份相关性分数

        返回：
            -MAX_QUALITY_DELTA 到 MAX_QUALITY_DELTA 之间的质量增量
        """
        text_lower = text.lower()

        positive_indicators = ["谢谢", "感谢", "太好了", "perfect", "great", "thanks", "helpful", "好的"]
        negative_indicators = ["不对", "错误", "不行", "wrong", "error", "fail", "不满意"]

        positive_count = sum(1 for kw in positive_indicators if kw in text_lower)
        negative_count = sum(1 for kw in negative_indicators if kw in text_lower)

        # 基础增量按身份相关性缩放
        base_delta = (positive_count - negative_count) * QUALITY_DELTA_COEFFICIENT
        return max(-MAX_QUALITY_DELTA, min(MAX_QUALITY_DELTA, base_delta * (0.5 + 0.5 * identity_relevance)))

    def _generate_relational_meaning(self, text: str, my_role: str, quality_delta: float) -> str:
        """
        生成关系意义的简短描述。

        参数：
            text: 交互文本
            my_role: 此交互中推断的角色
            quality_delta: 对关系质量的影响

        返回：
            关系意义的描述
        """
        if quality_delta > 0.1:
            return f"作为{my_role}，我们的关系更进一步了"
        elif quality_delta < -0.1:
            return f"作为{my_role}，这次互动有些挑战"
        else:
            return f"作为{my_role}，这是一次常规互动"

    # -------------------------------------------------------------------------
    # 身份影响提取
    # -------------------------------------------------------------------------

    def _extract_identity_impact(
        self,
        text: str,
        relational: RelationalContext,
        identity_relevance: float
    ) -> Optional[IdentityImpact]:
        """
        提取此交互如何影响我的身份。

        关键洞察：体验的意义可以随时间演变。

        参数：
            text: 交互文本
            relational: 关系背景
            identity_relevance: 身份相关性分数

        返回：
            如果显著则返回 IdentityImpact，否则返回 None
        """
        if identity_relevance < IDENTITY_RELEVANCE_THRESHOLD:
            return None  # 对身份影响不够显著

        # 识别受影响的维度
        affected_dimensions = self._identify_affected_dimensions(text, relational)

        if not affected_dimensions:
            return None

        # 生成初始意义
        initial_meaning = self._generate_identity_meaning(text, relational, affected_dimensions)

        return IdentityImpact(
            when_formed=now_ts(),
            initial_meaning=initial_meaning,
            current_meaning=initial_meaning,  # 创建时相同
            identity_dimensions_affected=affected_dimensions,
            evolution_history=[],
        )

    def _identify_affected_dimensions(self, text: str, relational: RelationalContext) -> List[str]:
        """
        识别此交互影响的身份维度。

        参数：
            text: 交互文本
            relational: 关系背景

        返回：
            受影响维度名称的列表
        """
        dimensions = []
        text_lower = text.lower()

        # 将关键词映射到身份维度
        dimension_keywords = {
            "作为解释者的我": ["解释", "说明", "clarify", "explain"],
            "作为帮助者的我": ["帮助", "协助", "help", "assist"],
            "作为学习者的我": ["学习", "理解", "learn", "understand"],
            "作为创造者的我": ["创作", "创建", "create", "build", "写"],
            "作为分析者的我": ["分析", "评估", "analyze", "evaluate"],
            "作为编程者的我": ["代码", "编程", "code", "program"],
        }

        for dimension, keywords in dimension_keywords.items():
            if any(kw in text_lower for kw in keywords):
                dimensions.append(dimension)

        # 始终包括基于角色的维度
        role_dimension = f"作为{relational.my_role_in_relation}的我"
        if role_dimension not in dimensions:
            dimensions.append(role_dimension)

        return dimensions[:MAX_IDENTITY_DIMENSIONS]

    def _generate_identity_meaning(
        self,
        text: str,
        relational: RelationalContext,
        affected_dimensions: List[str]
    ) -> str:
        """
        生成此交互的身份意义。

        参数：
            text: 交互文本
            relational: 关系背景
            affected_dimensions: 受影响身份维度的列表

        返回：
            身份意义的描述
        """
        dims_str = "、".join(affected_dimensions[:2]) if affected_dimensions else "我的身份"

        if relational.relationship_quality_delta > 0.1:
            return f"这次互动强化了{dims_str}"
        elif relational.relationship_quality_delta < -0.1:
            return f"这次互动挑战了{dims_str}"
        else:
            return f"这是{dims_str}的一次体现"

    # -------------------------------------------------------------------------
    # 关系故事管理
    # -------------------------------------------------------------------------

    def _get_or_create_relationship_story(self, relationship_entity: str) -> StoryArc:
        """
        获取或创建关系的故事。

        参数：
            relationship_entity: 关系所涉及的实体

        返回：
            此关系的现有或新创建的 StoryArc
        """
        # 检查是否已有此关系的故事
        story_id = self._relationship_story_index.get(relationship_entity)

        if story_id and story_id in self.stories:
            return self.stories[story_id]

        # 为此关系创建新故事
        story = StoryArc(
            id=det_id("story", f"rel_{relationship_entity}"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with=relationship_entity,
            relationship_type="user" if "user" in relationship_entity.lower() else "other",
        )

        self.stories[story.id] = story
        self.graph.add_node(story.id, "story", story)
        self._relationship_story_index[relationship_entity] = story.id

        return story

    def _update_identity_dimensions(self, plot: Plot) -> None:
        """
        根据情节的身份影响更新身份维度。

        参数：
            plot: 要从中提取身份影响的情节
        """
        if plot.identity_impact:
            for dim in plot.identity_impact.identity_dimensions_affected:
                current = self._identity_dimensions.get(dim, 0.0)
                # 身份维度的逐步强化
                self._identity_dimensions[dim] = current + IDENTITY_DIMENSION_GROWTH_RATE * (1.0 - current)
