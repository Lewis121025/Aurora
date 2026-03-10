"""
叙事视角模块
============================

定义叙事视角和视角选择逻辑。

职责：
1. 定义叙事视角类型（时间序、回顾式等）
2. 定义故事结构的叙事角色
3. 使用贝叶斯/随机策略进行视角选择
4. 特定视角的内存元素组织

设计原则：
- 零硬编码阈值：所有决策使用贝叶斯/随机策略
- 确定性可重现性：所有随机操作支持种子
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from aurora.lab.primitives.metric import LowRankMetric
from aurora.lab.narrator.config import (
    PERSPECTIVE_PRIOR_CHRONOLOGICAL,
    PERSPECTIVE_PRIOR_RETROSPECTIVE,
    PERSPECTIVE_PRIOR_CONTRASTIVE,
    PERSPECTIVE_PRIOR_FOCUSED,
    PERSPECTIVE_PRIOR_ABSTRACTED,
    SNIPPET_MAX_LENGTH,
)

from aurora.lab.models.plot import Plot
from aurora.lab.models.story import StoryArc
from aurora.lab.models.theme import Theme
from aurora.utils.math_utils import softmax
from aurora.utils.time_utils import now_ts

# =============================================================================
# Enums
# =============================================================================


class NarrativePerspective(Enum):
    """用于故事重构的叙事视角。

    每个视角为组织记忆提供不同的视角：

    CHRONOLOGICAL: 时间序叙事，最适合理解序列
    RETROSPECTIVE: 从现在回望，最适合反思
    CONTRASTIVE: 突出对比和变化，最适合成长分析
    FOCUSED: 围绕特定方面，最适合深入探索
    ABSTRACTED: 高级模式和主题，最适合身份综合
    """

    CHRONOLOGICAL = "chronological"  # 时间序：按时间顺序叙述
    RETROSPECTIVE = "retrospective"  # 回顾式：从现在回望过去
    CONTRASTIVE = "contrastive"  # 对比式：突出变化和对比
    FOCUSED = "focused"  # 聚焦式：围绕特定主题深入
    ABSTRACTED = "abstracted"  # 抽象式：提炼模式和主题


class NarrativeRole(Enum):
    """内存元素在叙事中可以扮演的角色。"""

    EXPOSITION = "exposition"  # 背景介绍
    RISING_ACTION = "rising_action"  # 情节发展
    CLIMAX = "climax"  # 高潮/转折点
    FALLING_ACTION = "falling_action"  # 情节回落
    RESOLUTION = "resolution"  # 结局/解决


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PerspectiveScore:
    """具有不确定性的视角选择分数。"""

    perspective: NarrativePerspective
    score: float  # 主要分数
    uncertainty: float = 0.5  # 认识论不确定性

    # 证据信号
    temporal_signal: float = 0.0  # 时间模式的强度
    contrast_signal: float = 0.0  # 对比模式的强度
    focus_signal: float = 0.0  # 查询特异性
    abstraction_signal: float = 0.0  # 主题相关性


# =============================================================================
# Perspective Selector
# =============================================================================


class PerspectiveSelector:
    """使用贝叶斯方法选择叙事视角。

    使用多个信号概率组合：
    - 查询特征
    - 内存结构（时间跨度、对比、主题）
    - 历史有效性（从反馈中学习）

    属性：
        metric: 用于相似度计算的学习度量
        rng: 用于可重现性的随机数生成器
        perspective_beliefs: 视角有效性的Beta后验信念
    """

    def __init__(
        self,
        metric: LowRankMetric,
        rng: np.random.Generator,
        perspective_beliefs: Optional[Dict[NarrativePerspective, Tuple[float, float]]] = None,
    ):
        """初始化视角选择器。

        参数：
            metric: 用于相似度计算的学习低秩度量
            rng: 随机数生成器
            perspective_beliefs: 可选的预存在信念（Beta后验）
        """
        self.metric: LowRankMetric = metric
        self.rng: np.random.Generator = rng

        # Default beliefs if not provided
        self.perspective_beliefs = perspective_beliefs or {
            NarrativePerspective.CHRONOLOGICAL: (2.0, 1.0),
            NarrativePerspective.RETROSPECTIVE: (1.5, 1.0),
            NarrativePerspective.CONTRASTIVE: (1.2, 1.0),
            NarrativePerspective.FOCUSED: (1.5, 1.0),
            NarrativePerspective.ABSTRACTED: (1.0, 1.0),
        }

    def select_perspective(
        self,
        query: str,
        plots: Optional[List[Plot]] = None,
        stories: Optional[Dict[str, StoryArc]] = None,
        themes: Optional[Dict[str, Theme]] = None,
        context: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[NarrativePerspective, Dict[str, float]]:
        """选择最合适的叙事视角。

        参数：
            query: 查询文本
            plots: 要考虑的可选情节列表
            stories: 用于上下文的可选故事
            themes: 用于抽象信号的可选主题
            context: 可选的附加上下文
            query_embedding: 可选的预计算查询嵌入

        返回：
            (选定的视角, 概率字典) 元组
        """
        scores: Dict[NarrativePerspective, PerspectiveScore] = {}

        for perspective in NarrativePerspective:
            score = self._compute_perspective_score(
                perspective=perspective,
                query=query,
                plots=plots,
                stories=stories,
                themes=themes,
                query_embedding=query_embedding,
            )
            scores[perspective] = score

        # 使用Thompson采样将分数转换为概率
        log_odds = []
        perspectives = list(NarrativePerspective)

        for p in perspectives:
            # 基础分数
            base = scores[p].score

            # 从有效性信念中Thompson采样
            a, b = self.perspective_beliefs[p]
            sampled_effectiveness = self.rng.beta(a, b)

            # 组合：基础分数加权有效性采样
            combined = base * (0.5 + sampled_effectiveness)
            log_odds.append(combined)

        # Softmax获取概率
        probs = softmax(log_odds)

        # 从分布中采样（随机选择）
        choice_idx = self.rng.choice(len(perspectives), p=probs)
        selected = perspectives[choice_idx]

        prob_dict = {p.value: float(prob) for p, prob in zip(perspectives, probs)}

        return selected, prob_dict

    def _compute_perspective_score(
        self,
        perspective: NarrativePerspective,
        query: str,
        plots: Optional[List[Plot]],
        stories: Optional[Dict[str, StoryArc]],
        themes: Optional[Dict[str, Theme]],
        query_embedding: Optional[np.ndarray],
    ) -> PerspectiveScore:
        """计算给定上下文的视角分数。"""
        score = PerspectiveScore(perspective=perspective, score=0.0)

        # 获取先验偏好
        priors = {
            NarrativePerspective.CHRONOLOGICAL: PERSPECTIVE_PRIOR_CHRONOLOGICAL,
            NarrativePerspective.RETROSPECTIVE: PERSPECTIVE_PRIOR_RETROSPECTIVE,
            NarrativePerspective.CONTRASTIVE: PERSPECTIVE_PRIOR_CONTRASTIVE,
            NarrativePerspective.FOCUSED: PERSPECTIVE_PRIOR_FOCUSED,
            NarrativePerspective.ABSTRACTED: PERSPECTIVE_PRIOR_ABSTRACTED,
        }
        score.score = priors.get(perspective, 0.5)

        if not plots:
            return score

        # 计算信号

        # 1. 时间信号：时间戳的跨度
        timestamps = [p.ts for p in plots]
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        score.temporal_signal = min(1.0, time_span / (7 * 24 * 3600))  # 归一化到一周

        # 2. 对比信号：张力的方差
        tensions = [p.tension for p in plots]
        if tensions:
            score.contrast_signal = float(np.std(tensions))

        # 3. 焦点信号：查询特异性
        if query_embedding is not None and plots:
            relevances = [self.metric.sim(p.embedding, query_embedding) for p in plots]
            score.focus_signal = float(np.max(relevances)) if relevances else 0.0

        # 4. 抽象信号：主题覆盖
        if themes:
            score.abstraction_signal = min(1.0, len(themes) * 0.2)

        # 基于视角调整分数
        if perspective == NarrativePerspective.CHRONOLOGICAL:
            score.score += 0.3 * score.temporal_signal
        elif perspective == NarrativePerspective.RETROSPECTIVE:
            score.score += 0.2 * score.temporal_signal + 0.1 * score.focus_signal
        elif perspective == NarrativePerspective.CONTRASTIVE:
            score.score += 0.4 * score.contrast_signal
        elif perspective == NarrativePerspective.FOCUSED:
            score.score += 0.5 * score.focus_signal
        elif perspective == NarrativePerspective.ABSTRACTED:
            score.score += 0.4 * score.abstraction_signal

        return score

    def feedback(self, perspective: NarrativePerspective, success: bool) -> None:
        """基于反馈更新信念。

        参数：
            perspective: 被使用的视角
            success: 是否有帮助/成功
        """
        a, b = self.perspective_beliefs[perspective]
        if success:
            self.perspective_beliefs[perspective] = (a + 1.0, b)
        else:
            self.perspective_beliefs[perspective] = (a, b + 1.0)


# =============================================================================
# Perspective-specific Organization
# =============================================================================


class PerspectiveOrganizer:
    """根据不同的叙事视角组织情节。

    提供以适合不同故事讲述方法的方式排列内存元素的方法。
    """

    def __init__(self, metric: LowRankMetric, rng: np.random.Generator):
        """初始化组织器。

        参数：
            metric: 用于相似度计算的学习度量
            rng: 随机数生成器
        """
        self.metric: LowRankMetric = metric
        self.rng: np.random.Generator = rng

    def organize_chronological(
        self,
        plots: List[Plot],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """按时间顺序组织情节。

        参数：
            plots: 要组织的情节
            compute_significance: 计算情节意义的函数

        返回：
            包含情节信息和角色的元素字典列表
        """
        sorted_plots = sorted(plots, key=lambda p: p.ts)

        elements = []
        for plot in sorted_plots:
            element = {
                "plot_id": plot.id,
                "role": NarrativeRole.RISING_ACTION,
                "content": self._truncate_content(plot.text),
                "timestamp": plot.ts,
                "significance": compute_significance(plot),
                "tension_level": plot.tension,
                "annotation": "",
            }
            elements.append(element)

        return elements

    def organize_retrospective(
        self,
        plots: List[Plot],
        query: str,
        query_embedding: Optional[np.ndarray],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """从现在回望组织情节。

        最近和最相关的事件优先，然后追踪回去。
        """
        now = now_ts()

        def retrospective_score(plot: Plot) -> float:
            recency = 1.0 / (1.0 + math.log1p(now - plot.ts))
            relevance = 0.5
            if query_embedding is not None:
                relevance = self.metric.sim(plot.embedding, query_embedding)
            return 0.6 * relevance + 0.4 * recency

        sorted_plots = sorted(plots, key=retrospective_score, reverse=True)

        elements = []
        for i, plot in enumerate(sorted_plots):
            element = {
                "plot_id": plot.id,
                "role": NarrativeRole.RISING_ACTION,
                "content": self._truncate_content(plot.text),
                "timestamp": plot.ts,
                "significance": retrospective_score(plot),
                "tension_level": plot.tension,
                "annotation": "回顾" if i == 0 else "",
            }
            elements.append(element)

        return elements

    def organize_contrastive(
        self,
        plots: List[Plot],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """组织情节以突出对比和变化。

        配对具有不同结果或情感的相似情节。
        """
        elements = []
        used = set()

        plots_list = list(plots)
        for i, plot_a in enumerate(plots_list):
            if plot_a.id in used:
                continue

            best_contrast = None
            best_contrast_score = 0.0

            for j, plot_b in enumerate(plots_list):
                if i == j or plot_b.id in used:
                    continue

                # 对比分数：相似的嵌入但不同的张力/结果
                semantic_sim = self.metric.sim(plot_a.embedding, plot_b.embedding)
                tension_diff = abs(plot_a.tension - plot_b.tension)

                # 好的对比：语义相关但张力不同
                if semantic_sim > 0.4:
                    contrast_score = semantic_sim * tension_diff
                    if contrast_score > best_contrast_score:
                        best_contrast_score = contrast_score
                        best_contrast = plot_b

            # 添加配对
            element_a = {
                "plot_id": plot_a.id,
                "role": NarrativeRole.RISING_ACTION,
                "content": self._truncate_content(plot_a.text),
                "timestamp": plot_a.ts,
                "significance": compute_significance(plot_a),
                "tension_level": plot_a.tension,
                "annotation": "对比起点",
            }
            elements.append(element_a)
            used.add(plot_a.id)

            if best_contrast:
                element_b = {
                    "plot_id": best_contrast.id,
                    "role": NarrativeRole.RISING_ACTION,
                    "content": self._truncate_content(best_contrast.text),
                    "timestamp": best_contrast.ts,
                    "significance": compute_significance(best_contrast),
                    "tension_level": best_contrast.tension,
                    "annotation": "对比终点",
                }
                elements.append(element_b)
                used.add(best_contrast.id)

        return elements

    def organize_focused(
        self,
        plots: List[Plot],
        query: str,
        query_embedding: Optional[np.ndarray],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """围绕特定焦点组织情节。

        最相关的查询优先，相关性递减。
        """
        if query_embedding is None:
            return self.organize_chronological(plots, compute_significance)

        def focus_score(plot: Plot) -> float:
            return self.metric.sim(plot.embedding, query_embedding)

        sorted_plots = sorted(plots, key=focus_score, reverse=True)

        elements = []
        for i, plot in enumerate(sorted_plots):
            relevance = focus_score(plot)
            element = {
                "plot_id": plot.id,
                "role": NarrativeRole.RISING_ACTION,
                "content": self._truncate_content(plot.text),
                "timestamp": plot.ts,
                "significance": relevance,
                "tension_level": plot.tension,
                "annotation": "核心" if i == 0 else ("相关" if relevance > 0.5 else ""),
            }
            elements.append(element)

        return elements

    def organize_abstracted(
        self,
        plots: List[Plot],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """通过提取模式和主题组织情节。

        将相似的情节分组以形成主题集群。
        """
        # 通过嵌入相似性进行简单聚类
        clusters: List[List[Plot]] = []
        unclustered = list(plots)

        while unclustered:
            seed = unclustered.pop(0)
            cluster = [seed]

            remaining = []
            for plot in unclustered:
                sim = self.metric.sim(seed.embedding, plot.embedding)
                if sim > 0.6:
                    cluster.append(plot)
                else:
                    remaining.append(plot)

            clusters.append(cluster)
            unclustered = remaining

        # 创建带有集群注解的元素
        elements = []
        for cluster_idx, cluster in enumerate(clusters):
            cluster.sort(key=lambda p: p.ts)

            for i, plot in enumerate(cluster):
                annotation = f"主题{cluster_idx + 1}" if i == 0 else ""
                element = {
                    "plot_id": plot.id,
                    "role": NarrativeRole.RISING_ACTION,
                    "content": self._truncate_content(plot.text),
                    "timestamp": plot.ts,
                    "significance": compute_significance(plot),
                    "tension_level": plot.tension,
                    "annotation": annotation,
                }
                elements.append(element)

        return elements

    def _truncate_content(self, content: str) -> str:
        """将内容截断到最大长度。"""
        if len(content) <= SNIPPET_MAX_LENGTH:
            return content
        return content[: SNIPPET_MAX_LENGTH - 3] + "..."
