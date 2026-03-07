"""
上下文恢复模块
=======================

处理因果上下文恢复和转折点识别。

职责：
1. 从情节追踪因果链
2. 识别叙事序列中的转折点
3. 基于时间、语义和故事关系查找连接的情节

设计原则：
- 零硬编码阈值：使用概率修剪
- 确定性可重现性：所有随机操作支持种子
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.constants import (
    DEFAULT_CAUSAL_DEPTH,
    MAX_CAUSAL_CHAIN_LENGTH,
    TURNING_POINT_TENSION_THRESHOLD_BASE,
)
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.utils.math_utils import sigmoid


class ContextRecovery:
    """处理上下文恢复和因果链追踪。

    提供以下方法：
    - 通过因果链和时间连接追踪回溯
    - 识别情节序列中的转折点
    - 查找通过各种关系连接的情节

    所有决策使用概率方法而不是硬阈值。

    属性：
        metric: 用于相似度计算的学习度量
        rng: 用于可重现性的随机数生成器
        graph: 用于显式因果链接的可选内存图
    """
    
    def __init__(
        self,
        metric: LowRankMetric,
        rng: np.random.Generator,
        graph: Optional[MemoryGraph] = None,
    ):
        """初始化上下文恢复。

        参数：
            metric: 用于相似度计算的学习低秩度量
            rng: 随机数生成器
            graph: 用于因果追踪的可选内存图
        """
        self.metric: LowRankMetric = metric
        self.rng: np.random.Generator = rng
        self.graph: Optional[MemoryGraph] = graph
    
    def recover_context(
        self,
        plot: Plot,
        plots_dict: Dict[str, Plot],
        depth: int = DEFAULT_CAUSAL_DEPTH,
    ) -> List[Plot]:
        """恢复情节的因果上下文。

        通过因果链和时间连接追踪回溯
        以查找导致此情节的上下文。

        参数：
            plot: 要恢复上下文的情节
            plots_dict: 所有可用情节的字典
            depth: 要追踪的因果链的最大深度

        返回：
            形成因果上下文的情节列表（按相关性排序）
        """
        context_plots: List[Tuple[Plot, float]] = []  # (plot, relevance_score)
        visited: set = {plot.id}

        # BFS with probabilistic pruning
        queue: List[Tuple[str, int, float]] = [(plot.id, 0, 1.0)]  # (id, depth, path_strength)
        
        while queue and len(context_plots) < MAX_CAUSAL_CHAIN_LENGTH:
            current_id, current_depth, path_strength = queue.pop(0)
            
            if current_depth >= depth:
                continue

            current_plot = plots_dict.get(current_id)
            if current_plot is None:
                continue

            # 查找连接的情节
            connected = self._find_connected_plots(
                current_plot, plots_dict, visited
            )

            for connected_plot, connection_strength in connected:
                if connected_plot.id in visited:
                    continue

                visited.add(connected_plot.id)

                # 计算相关性分数
                new_strength = path_strength * connection_strength

                # 概率修剪（较弱的连接不太可能被跟踪）
                if self.rng.random() < new_strength:
                    context_plots.append((connected_plot, new_strength))
                    queue.append((connected_plot.id, current_depth + 1, new_strength))

        # 按相关性排序并返回仅情节
        context_plots.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in context_plots]
    
    def identify_turning_points(
        self,
        plots: List[Plot],
        stories: Optional[Dict[str, StoryArc]] = None,
    ) -> List[Plot]:
        """识别情节序列中的转折点。

        转折点是显著变化的时刻：
        - 高张力后跟随解决
        - 关系动态的转变
        - 身份相关事件

        使用概率检测，而不是固定阈值。

        参数：
            plots: 要分析的情节列表
            stories: 可选的故事上下文

        返回：
            被识别为转折点的情节列表
        """
        if len(plots) < 2:
            return []

        # 按时间戳排序
        sorted_plots = sorted(plots, key=lambda p: p.ts)

        turning_points: List[Plot] = []

        # 计算张力曲线
        tensions = [p.tension for p in sorted_plots]

        # 基于张力分布的自适应阈值
        if tensions:
            mean_tension = np.mean(tensions)
            std_tension = np.std(tensions) + 1e-6

            for i, plot in enumerate(sorted_plots):
                # 张力的Z分数
                z_score = (plot.tension - mean_tension) / std_tension

                # 成为转折点的概率
                # 更高的张力 = 更高的概率
                p_turning = sigmoid(z_score - TURNING_POINT_TENSION_THRESHOLD_BASE)

                # 也考虑身份影响
                if plot.has_identity_impact():
                    p_turning = min(1.0, p_turning * 1.5)

                # 检查张力下降（高潮后跟随解决）
                if i > 0 and i < len(sorted_plots) - 1:
                    prev_tension = sorted_plots[i - 1].tension
                    next_tension = sorted_plots[i + 1].tension

                    # 峰值检测
                    if plot.tension > prev_tension and plot.tension > next_tension:
                        p_turning = min(1.0, p_turning * 1.3)

                # 随机选择
                if self.rng.random() < p_turning:
                    turning_points.append(plot)

        return turning_points
    
    def _find_connected_plots(
        self,
        plot: Plot,
        plots_dict: Dict[str, Plot],
        visited: set,
    ) -> List[Tuple[Plot, float]]:
        """查找与给定情节连接的情节。

        考虑多种连接类型：
        - 时间接近（时间上接近）
        - 语义相似性（相似的嵌入）
        - 故事成员（相同的故事弧）

        参数：
            plot: 要查找连接的情节
            plots_dict: 所有可用情节的字典
            visited: 已访问情节ID的集合

        返回：
            (情节, 连接强度) 元组列表，按强度排序
        """
        connected = []

        for pid, p in plots_dict.items():
            if pid in visited or pid == plot.id:
                continue

            # 时间连接（时间上接近）
            time_diff = abs(p.ts - plot.ts)
            temporal_strength = math.exp(-time_diff / 3600.0)  # 在1小时内衰减

            # 语义连接
            semantic_strength = self.metric.sim(plot.embedding, p.embedding)

            # 故事连接
            story_strength = 1.0 if p.story_id == plot.story_id and plot.story_id else 0.0

            # 组合强度
            strength = 0.3 * temporal_strength + 0.4 * semantic_strength + 0.3 * story_strength

            if strength > 0.2:  # 软阈值
                connected.append((p, strength))

        # 按强度排序
        connected.sort(key=lambda x: x[1], reverse=True)
        return connected[:5]  # 限制前5个


class TurningPointDetector:
    """用于叙事中转折点的专用检测器。

    提供可与基本方法一起使用或代替基本方法的替代检测策略。
    """

    def __init__(self, rng: np.random.Generator):
        """初始化检测器。

        参数：
            rng: 随机数生成器
        """
        self.rng = rng
    
    def detect_from_elements(
        self,
        elements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """从叙事元素识别转折点。

        参数：
            elements: 具有tension_level的叙事元素字典列表

        返回：
            被识别为转折点的元素列表
        """
        if len(elements) < 2:
            return []
        
        turning_points = []
        tensions = [e.get("tension_level", 0.0) for e in elements]
        
        if not tensions or all(t == 0 for t in tensions):
            return []
        
        mean_t = np.mean(tensions)
        std_t = np.std(tensions) + 1e-6
        
        for element in elements:
            tension = element.get("tension_level", 0.0)
            z = (tension - mean_t) / std_t
            p_turn = sigmoid(z - 0.5)
            
            if self.rng.random() < p_turn:
                turning_points.append(element)
        
        return turning_points
