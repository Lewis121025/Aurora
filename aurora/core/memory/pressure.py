"""
AURORA 内存压力管理模块 (Pressure Management Module)
=================================

以成长为导向的遗忘与压力管理。

核心职责：
- 基于成长而非仅基于容量来决定遗忘什么
- 计算记忆对身份认同、人际关系和成长的贡献
- 通过选择性遗忘来管理记忆压力

理念："遗忘不是丢失信息，而是选择成为什么样的人" (Forgetting is not losing information, it's choosing what to become.)
"""

from __future__ import annotations

import logging
import math

import numpy as np

from aurora.core.models.plot import Plot
from aurora.utils.math_utils import softmax
from aurora.utils.time_utils import now_ts
from aurora.core.config.evolution import GROWTH_HINDRANCE_AGE_SECONDS

logger = logging.getLogger(__name__)

class PressureMixin:
    """提供以成长为导向的压力管理功能的 Mixin 类。"""

    # -------------------------------------------------------------------------
    # 以成长为导向的遗忘 (Growth-Oriented Forgetting)
    # -------------------------------------------------------------------------

    def _pressure_manage(self) -> None:
        """
        以成长为导向的遗忘：基于成长而非仅基于容量来决定遗忘什么。
        
        核心理念："遗忘就是选择成为什么样的人。"
        
        在 benchmark_mode（基准测试模式）下，禁用此方法以确保没有信息丢失，
        因为用来评估的每一轮对话都可能包含关键信息。
        """
        # 基准测试模式：跳过压力管理以保留所有情节(plots)
        if getattr(self, 'benchmark_mode', False):
            return
        
        max_plots = self.cfg.max_plots
        explicit_count = sum(
            1 for plot in self.plots.values()
            if plot.exposure == "explicit" and plot.source != "seed"
        )
        if explicit_count <= max_plots:
            return

        # 获取要移除的候选对象
        candidates = [
            plot
            for plot in self.plots.values()
            if plot.status == "active"
            and plot.story_id is not None
            and plot.exposure == "explicit"
            and plot.source != "seed"
        ]
        if not candidates:
            return

        # 对候选对象进行评分并选择要移除的对象
        self._score_candidates_for_removal(candidates)
        remove_ids = self._select_plots_to_forget(candidates, explicit_count - max_plots)
        
        # 遗忘选定的情节(plots)
        for pid in remove_ids:
            if pid in self.plots:
                self._forget_plot(pid)
        
        logger.debug(
            f"Pressure managed: removed {len(remove_ids)} plots, "
            f"remaining={len([plot for plot in self.plots.values() if plot.status == 'active'])}"
        )

    def _score_candidates_for_removal(self, candidates: list) -> None:
        """基于成长贡献对每个候选情节(plot)进行评分。"""
        for plot in candidates:
            identity_contribution = self._compute_identity_contribution(plot)
            relationship_contribution = self._compute_relationship_contribution(plot)
            growth_contribution = self._compute_growth_contribution(plot)
            
            plot._keep_score = (
                0.4 * identity_contribution +
                0.3 * relationship_contribution +
                0.3 * growth_contribution
            )

    def _select_plots_to_forget(self, candidates: list, excess: int) -> set:
        """基于保留分数(keep scores)选择要遗忘的情节。
        
        对负的保留分数使用 softmax 来进行概率性选择，决定遗忘哪些情节。
        保留分数越低 = 遗忘的概率越高。
        """
        if not candidates or excess <= 0:
            return set()
        
        # 将多余数量限制在可用候选范围内，以避免采样错误
        actual_excess = min(excess, len(candidates))
        
        keep_scores = np.array([getattr(plot, '_keep_score', plot.mass()) for plot in candidates], dtype=np.float32)
        logits = (-keep_scores).tolist()
        probs = np.array(softmax(logits), dtype=np.float64)
        
        return set(self.rng.choice([plot.id for plot in candidates], size=actual_excess, replace=False, p=probs))

    # -------------------------------------------------------------------------
    # 贡献计算 (Contribution Computations)
    # -------------------------------------------------------------------------

    def _compute_identity_contribution(self, plot: Plot) -> float:
        """
        计算该情节对当前身份认同的贡献程度。
        
        如果符合以下条件，则表示情节对身份有贡献：
        - 它是身份维度（主题，Theme）的证据
        - 它强化了“我是谁”
        
        参数:
            plot: 要评估的情节
            
        返回:
            0 到 1 之间的身份贡献分数
        """
        if not plot.identity_impact:
            return 0.3  # 没有明确身份影响的情节的基线分数
        
        contribution = 0.0
        
        # 检查此情节是否为活跃身份维度的证据
        for dim in plot.identity_impact.identity_dimensions_affected:
            dim_strength = self._identity_dimensions.get(dim, 0.0)
            contribution += dim_strength * 0.3
        
        # 检查情节的故事是否与重要主题相连
        if plot.story_id and plot.story_id in self.stories:
            story = self.stories[plot.story_id]
            # 找到此故事支持的主题
            for theme in self.themes.values():
                if plot.story_id in theme.story_ids:
                    contribution += theme.confidence() * 0.2
        
        return min(1.0, contribution + 0.2)  # 基线分数为 0.2
    
    def _compute_relationship_contribution(self, plot: Plot) -> float:
        """
        计算该情节在多大程度上维持了一段重要的人际关系。
        
        如果符合以下条件，情节对关系有贡献：
        - 它是一段重要关系中的锚点
        - 这段关系是健康且持续进行的
        
        参数:
            plot: 要评估的情节
            
        返回:
            0 到 1 之间的关系贡献分数
        """
        if not plot.relational:
            return 0.3  # 非关系型情节的基线分数
        
        relationship_entity = plot.relational.with_whom
        story_id = self._relationship_story_index.get(relationship_entity)
        
        if not story_id or story_id not in self.stories:
            return 0.3
        
        story = self.stories[story_id]
        
        # 因素 1：关系健康度 (Relationship health)
        health_factor = story.relationship_health
        
        # 因素 2：关系近期的新鲜度（正在进行的关系更重要）
        recency = 1.0 / (1.0 + math.log1p(now_ts() - story.updated_ts) / 10)
        
        # 因素 3：该情节在关系中是否是最近发生的？
        if story.plot_ids:
            plot_position = story.plot_ids.index(plot.id) if plot.id in story.plot_ids else -1
            if plot_position >= 0:
                # 越近期的情节越重要
                recency_in_story = plot_position / len(story.plot_ids)
            else:
                recency_in_story = 0.5
        else:
            recency_in_story = 0.5
        
        return 0.3 * health_factor + 0.4 * recency + 0.3 * recency_in_story
    
    def _compute_growth_contribution(self, plot: Plot) -> float:
        """
        计算保留此情节对个人成长有多大帮助。
        
        核心问题：
        - 这是否提供了持续的学习价值？
        - 这是否会影响未来的行为？
        - 这是否会阻碍成长（例如：强化消极的自我认知）？
        
        参数:
            plot: 要评估的情节
            
        返回:
            0 到 1 之间的成长贡献分数
        """
        # 因素 1：学习价值 - 具有高惊喜/张力的情节通常具有学习价值
        learning_value = min(1.0, plot.tension * 0.5) if plot.tension > 0 else 0.3
        
        # 因素 2：未来影响 - 最近被访问过的情节将影响未来
        age_factor = 1.0 / math.log1p(max(1.0, now_ts() - plot.ts))
        access_factor = math.log1p(plot.access_count + 1) / 5.0
        future_influence = 0.5 * age_factor + 0.5 * min(1.0, access_factor)
        
        # 因素 3：成长障碍检查
        # （在了完整的实现中，这里会使用情感/内容分析）
        # 目前使用一个简单的启发式规则：非常古老、从未被访问过的情节可能会阻碍成长
        if plot.access_count == 0 and (now_ts() - plot.ts) > GROWTH_HINDRANCE_AGE_SECONDS:
            growth_hindrance = 0.3  # 一定程度的惩罚
        else:
            growth_hindrance = 0.0
        
        return 0.4 * learning_value + 0.4 * future_influence - growth_hindrance

    # -------------------------------------------------------------------------
    # 遗忘 (Forgetting)
    # -------------------------------------------------------------------------

    def _forget_plot(self, plot_id: str) -> None:
        """
        遗忘一个情节 - 不是物理删除，而是放手(letting go)。
        
        理念："遗忘不是丢失信息，而是选择成为什么样的人。"
        
        情节的精髓将以以下方式保留在：
        - 故事的质心 (总体意义) 中
        - 关系发展的轨迹 (如果具有关系属性) 中
        - 其所影响的身份维度中
        
        参数:
            plot_id: 要被遗忘的情节的 ID
        """
        p = self.plots.get(plot_id)
        if p is None or p.story_id is None:
            return
        
        p.status = "absorbed"

        # 从向量索引中移除以减少检索噪音
        self.vindex.remove(plot_id)
        
        # 如果此情节对身份认同有影响，该影响已保存在各个维度中
        # （_identity_dimensions 字典保留了累积的效应）
        
        # 如果此情节包含人际关系背景，关系的轨迹会将其保留下来
        # （故事的 relationship_arc 保存了该模式）
