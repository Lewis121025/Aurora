"""
AURORA 分配模型
=========================

使用中餐厅过程（CRP）和概率生成模型的非参数分层分配。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from aurora.utils.math_utils import softmax
from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.core.components.metric import LowRankMetric


class CRPAssigner:
    """用于（项目 -> 集群）的通用CRP类分配器，具有概率采样。

    为非参数聚类实现中餐厅过程，其中集群数量随数据增长但以浓度参数控制的速率增长。

    属性:
        alpha: CRP浓度参数（越高 = 越多新集群）
    """

    def __init__(self, alpha: float = 1.0, seed: int = 0):
        """初始化CRP分配器。

        参数:
            alpha: 浓度参数
            seed: 随机种子
        """
        self.alpha = alpha
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, logps: Dict[str, float]) -> Tuple[Optional[str], Dict[str, float]]:
        """使用CRP采样集群分配。

        参数:
            logps: 现有集群的对数概率

        返回:
            元组（选择的集群ID或None表示新集群，后验概率）
        """
        # 添加新集群选项
        logs = dict(logps)
        logs["__new__"] = math.log(self.alpha)
        keys = list(logs.keys())
        probs = softmax([logs[k] for k in keys])
        choice = self.rng.choice(keys, p=np.array(probs, dtype=np.float64))
        post = {k: p for k, p in zip(keys, probs)}
        if choice == "__new__":
            return None, post
        return choice, post

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为JSON兼容的字典。"""
        return {
            "alpha": self.alpha,
            "seed": self._seed,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "CRPAssigner":
        """从状态字典重构。"""
        return cls(alpha=d["alpha"], seed=d.get("seed", 0))


class StoryModel:
    """似然模型 p(plot | story)，具有可解释的因子。

    无固定权重：所有组件都是生成对数似然项。
    该模型基于语义、时间和演员特征计算给定故事观察到情节的概率。

    组件:
        - 语义似然：使用故事的离散度在度量空间中的高斯分布
        - 时间似然：围绕学习的典型间隙的指数分布
        - 演员似然：Dirichlet-多项式预测

    属性:
        metric: 用于语义相似性的学习度量
    """

    def __init__(self, metric: LowRankMetric):
        """初始化故事模型。

        参数:
            metric: 用于语义相似性的学习低秩度量
        """
        self.metric = metric

    def loglik(self, plot: Plot, story: StoryArc) -> float:
        """计算给定故事的情节的对数似然。

        参数:
            plot: 要评估的情节
            story: 要条件化的故事

        返回:
            对数似然值
        """
        # 语义似然：使用故事的离散度在度量空间中的高斯分布
        ll_sem = 0.0
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            var = max(story.dist_var(), 1e-3)
            ll_sem = -0.5 * d2 / var

        # 时间似然：围绕学习的典型间隙的指数分布
        ll_time = 0.0
        if story.plot_ids:
            gap = max(0.0, plot.ts - story.updated_ts)
            tau = story.gap_mean_safe()
            lam = 1.0 / max(tau, 1e-6)
            ll_time = math.log(lam + 1e-12) - lam * gap

        # 演员似然：Dirichlet-多项式预测
        ll_actor = 0.0
        beta = 1.0
        total = sum(story.actor_counts.values())
        denom = total + beta * max(len(story.actor_counts), 1)
        for a in plot.actors:
            ll_actor += math.log(story.actor_counts.get(a, 0) + beta) - math.log(denom + 1e-12)

        return ll_sem + ll_time + ll_actor


class ThemeModel:
    """似然模型 p(story | theme)。

    基于学习的度量空间中的语义相似性计算故事属于主题的概率。

    属性:
        metric: 用于语义相似性的学习度量
    """

    def __init__(self, metric: LowRankMetric):
        """初始化主题模型。

        参数:
            metric: 用于语义相似性的学习低秩度量
        """
        self.metric = metric

    def loglik(self, story: StoryArc, theme: Theme) -> float:
        """计算给定主题的故事的对数似然。

        参数:
            story: 要评估的故事
            theme: 要条件化的主题

        返回:
            对数似然值
        """
        if theme.prototype is None or story.centroid is None:
            return 0.0
        d2 = self.metric.d2(story.centroid, theme.prototype)
        # 主题离散度未存储；我们使用稳健的默认尺度 = 1
        return -0.5 * d2
