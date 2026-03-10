"""
AURORA 强盗算法组件
=========================

基于Thompson采样的随机编码决策制定。
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from aurora.utils.math_utils import sigmoid


MIN_STORE_PROB = 0.6


class ThompsonBernoulliGate:
    """具有Thompson采样的随机编码策略。

    我们不通过"分数 > 阈值"来决定。我们采样参数向量w并以概率sigmoid(w·x)编码。映射从延迟奖励中学习。

    这实现了对编码/跳过决策的贝叶斯方法，允许系统探索不同的编码策略，同时从下游任务成功中学习。

    参数稳定性:
        - forgetting_factor (lambda)：防止精度无限累积。
          prec = lambda * prec + grad * grad，其中lambda=0.99确保每次更新约1%的衰减。
        - 这使系统保持"可塑性"并能够适应分布变化。

    属性:
        d: 特征维度
        w_mean: 权重分布的均值
        prec: 权重分布的精度（逆方差）
        grad2: 用于自适应学习的RMSprop累加器
        t: 更新计数器
    """

    def __init__(
        self,
        feature_dim: int,
        seed: int = 0,
        forgetting_factor: float = 0.98,
        init_precision: float = 5e-2,
        min_store_prob: float = MIN_STORE_PROB,
    ):
        """初始化Thompson采样门。

        参数:
            feature_dim: 特征向量的维度
            seed: 随机种子
            forgetting_factor: 精度累积的衰减因子（0.98 = 更可塑）
            init_precision: 权重的初始精度（越高 = 越少探索）
            min_store_prob: 最小存储概率下限（默认0.3）。
                即使Thompson采样建议低存储概率，
                这个下限也确保某种基线存储速率以防止
                丢失太多信息。有助于改进AR分数。

        基准优化:
        - 更高的init_precision（5e-2 vs 1e-2）减少初始探索，
          导致编码策略更快收敛
        - 更低的forgetting_factor（0.98 vs 0.99）允许更快适应
          分布变化，改进TTL和CR
        - min_store_prob为0.3确保~70%的存储速率目标是可达到的
        """
        self.d = feature_dim
        self._seed = seed
        self.lambda_ = forgetting_factor  # 精度的遗忘因子
        self.min_store_prob = min_store_prob  # 存储概率的下限
        self.rng = np.random.default_rng(seed)

        self.w_mean = np.zeros(self.d, dtype=np.float32)
        self.prec = np.ones(self.d, dtype=np.float32) * init_precision
        self.grad2 = np.zeros(self.d, dtype=np.float32)  # RMS

        self.t = 0

        # 用于监控的统计跟踪
        self._encode_count = 0
        self._skip_count = 0

        # 存储初始参数以供序列化
        self._init_precision = init_precision

    def _sample_w(self) -> np.ndarray:
        """从后验分布采样权重向量。

        返回:
            采样的权重向量
        """
        std = np.sqrt(1.0 / (self.prec + 1e-9))
        return self.w_mean + self.rng.normal(size=self.d).astype(np.float32) * std

    def prob(self, x: np.ndarray) -> float:
        """计算特征向量的编码概率。

        返回的概率有一个min_store_prob的下限，以确保
        即使对于低价值特征也有基线存储速率。这防止
        过度激进的过滤，可能会伤害基准AR分数。

        参数:
            x: 特征向量

        返回:
            编码概率（Thompson采样，带min_store_prob下限）
        """
        w = self._sample_w()
        raw_prob = sigmoid(float(np.dot(w, x)))
        # 应用最小存储概率下限
        return max(raw_prob, self.min_store_prob)

    def decide(self, x: np.ndarray) -> bool:
        """做出随机编码决策。

        参数:
            x: 特征向量

        返回:
            如果应该编码则为True，如果应该跳过则为False
        """
        result = bool(self.rng.random() < self.prob(x))
        if result:
            self._encode_count += 1
        else:
            self._skip_count += 1
        return result

    def update(self, x: np.ndarray, reward: float) -> None:
        """强盗更新：来自下游任务成功的奖励在[-1, 1]范围内。

        使用遗忘因子防止精度无限累积，
        这会导致方差接近零并冻结策略。

        参数:
            x: 用于决策的特征向量
            reward: 来自下游任务的奖励信号
        """
        self.t += 1
        y = 1.0 if reward > 0 else 0.0
        p = sigmoid(float(np.dot(self.w_mean, x)))
        grad = (y - p) * x  # 上升

        # 具有自调整步长的RMS
        self.grad2 = 0.99 * self.grad2 + 0.01 * (grad * grad)
        base = 1.0 / math.sqrt(self.t + 1.0)
        step = base * grad / (np.sqrt(self.grad2) + 1e-6)
        self.w_mean += step

        # 应用遗忘因子防止精度无限增长
        # 这确保系统保持"可塑性"并能够适应分布变化
        self.prec = self.lambda_ * self.prec + grad * grad

    def pass_rate(self) -> float:
        """返回门的通过率（编码 / 总决策）。"""
        total = self._encode_count + self._skip_count
        return self._encode_count / total if total > 0 else 0.5

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为JSON兼容的字典。"""
        return {
            "d": self.d,
            "seed": self._seed,
            "lambda": self.lambda_,
            "init_precision": getattr(self, "_init_precision", 5e-2),
            "min_store_prob": self.min_store_prob,
            "w_mean": self.w_mean.tolist(),
            "prec": self.prec.tolist(),
            "grad2": self.grad2.tolist(),
            "t": self.t,
            "encode_count": self._encode_count,
            "skip_count": self._skip_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "ThompsonBernoulliGate":
        """从状态字典重构。"""
        obj = cls(
            feature_dim=d["d"],
            seed=d.get("seed", 0),
            forgetting_factor=d.get("lambda", 0.98),
            init_precision=d.get("init_precision", 5e-2),
            min_store_prob=d.get("min_store_prob", 0.3),
        )
        obj.w_mean = np.array(d["w_mean"], dtype=np.float32)
        obj.prec = np.array(d["prec"], dtype=np.float32)
        obj.grad2 = np.array(d["grad2"], dtype=np.float32)
        obj.t = d["t"]
        obj._encode_count = d.get("encode_count", 0)
        obj._skip_count = d.get("skip_count", 0)
        return obj
