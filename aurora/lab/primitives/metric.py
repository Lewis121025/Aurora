"""
AURORA 度量学习
=======================

用于自适应相似性学习的低秩Mahalanobis度量。
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


class LowRankMetric:
    """低秩Mahalanobis度量：d(x,y)^2 = ||L(x-y)||^2。

    可解释为从检索反馈中学习任务和用户适应的"信息几何"度量。

    该度量学习映射嵌入空间，使相关项接近，无关项远离。

    参数稳定性:
        - window_size：Adagrad累加器重置的滑动窗口。
          每`window_size`次更新，累加器G被重新缩放以防止
          学习速率随时间衰减到接近零。
        - decay_factor：在周期性重新计算期间应用于G（默认0.5）。

    属性:
        dim: 输入嵌入维度
        rank: 度量的秩（L的输出维度）
        L: 低秩投影矩阵（rank x dim）
        G: 用于自适应学习速率的Adagrad累加器
        t: 更新计数器
    """

    def __init__(
        self,
        dim: int,
        rank: int = 64,
        seed: int = 0,
        window_size: int = 10000,
        decay_factor: float = 0.5,
    ):
        """初始化低秩度量。

        参数:
            dim: 嵌入维度
            rank: 学习度量的秩
            seed: 初始化的随机种子
            window_size: Adagrad累加器重置窗口
            decay_factor: 累加器重新缩放的衰减因子
        """
        self.dim = dim
        self.rank = min(rank, dim)
        self._seed = seed
        self.window_size = window_size
        self.decay_factor = decay_factor

        rng = np.random.default_rng(seed)
        self.L = np.eye(dim, dtype=np.float32)[: self.rank].copy()
        self.L += (0.01 * rng.normal(size=self.L.shape)).astype(np.float32)

        self.G = np.zeros_like(self.L)  # Adagrad累加器
        self.t = 0

        # 用于监控的统计
        self._total_loss = 0.0
        self._update_count = 0

    def d2(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算学习度量中的平方距离。

        参数:
            x: 第一个嵌入
            y: 第二个嵌入

        返回:
            平方距离 d(x,y)^2 = ||L(x-y)||^2
        """
        z = (x - y).astype(np.float32)
        p = self.L @ z
        return float(np.dot(p, p))

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算学习度量中的相似性。

        参数:
            x: 第一个嵌入
            y: 第二个嵌入

        返回:
            相似性在(0, 1]范围内，其中1表示相同
        """
        return 1.0 / (1.0 + self.d2(x, y))

    def update_triplet(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        margin: float = 1.0,
    ) -> float:
        """在线OASIS类更新，使用Adagrad。

        更新度量以使锚点更接近正样本并
        远离负样本。

        margin不是相似性的阈值；它是几何分离单位。

        包括滑动窗口机制：定期重新缩放Adagrad
        累加器以防止学习速率消失。

        参数:
            anchor: 锚点嵌入
            positive: 正（相似）嵌入
            negative: 负（不相似）嵌入
            margin: 三元组损失的边界

        返回:
            损失值（如果约束已满足则为0）
        """
        self.t += 1
        ap = (anchor - positive).astype(np.float32)
        an = (anchor - negative).astype(np.float32)
        Lap = self.L @ ap
        Lan = self.L @ an
        dap = float(np.dot(Lap, Lap))
        dan = float(np.dot(Lan, Lan))
        loss = max(0.0, margin + dap - dan)
        if loss <= 0:
            return 0.0

        grad = 2.0 * (np.outer(Lap, ap) - np.outer(Lan, an)).astype(np.float32)
        self.G += grad * grad
        # 自调整学习速率：随t自动衰减
        base = 1.0 / math.sqrt(self.t + 1.0)
        step = base * grad / (np.sqrt(self.G) + 1e-8)
        self.L -= step

        # 跟踪统计
        self._total_loss += loss
        self._update_count += 1

        # 滑动窗口：定期重新缩放G以保持可塑性
        # 这防止Adagrad累加器无限增长
        # 这会导致学习速率接近零
        if self.t > 0 and self.t % self.window_size == 0:
            self._rescale_accumulator()

        return float(loss)

    def _rescale_accumulator(self) -> None:
        """重新缩放Adagrad累加器以保持学习能力。

        这实现了一个"软重置"，保留学习的结构
        同时防止累加器变得太大。
        """
        self.G *= self.decay_factor

    def average_loss(self) -> float:
        """返回所有更新的平均三元组损失。"""
        return self._total_loss / self._update_count if self._update_count > 0 else 0.0

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为JSON兼容的字典。"""
        return {
            "dim": self.dim,
            "rank": self.rank,
            "seed": self._seed,
            "window_size": self.window_size,
            "decay_factor": self.decay_factor,
            "L": self.L.tolist(),
            "G": self.G.tolist(),
            "t": self.t,
            "total_loss": self._total_loss,
            "update_count": self._update_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "LowRankMetric":
        """从状态字典重构。"""
        obj = cls(
            dim=d["dim"],
            rank=d["rank"],
            seed=d.get("seed", 0),
            window_size=d.get("window_size", 10000),
            decay_factor=d.get("decay_factor", 0.5),
        )
        obj.L = np.array(d["L"], dtype=np.float32)
        obj.G = np.array(d["G"], dtype=np.float32)
        obj.t = d["t"]
        obj._total_loss = d.get("total_loss", 0.0)
        obj._update_count = d.get("update_count", 0)
        return obj
