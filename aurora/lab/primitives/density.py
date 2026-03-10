"""
AURORA 密度估计
==========================

用于惊喜度计算的在线核密度估计器。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_COLD_START_SURPRISE = 8.0
DENSITY_MIN_SAMPLES = 5


class OnlineKDE:
    """嵌入空间中的核密度估计器。

    用于计算惊喜度 = -log p(x)，无需任何相似度阈值。

    这实现了一个水库采样的KDE，在保持有限内存占用的同时为传入的嵌入提供密度估计。

    属性:
        dim: 嵌入维度
        reservoir: 保留的最大向量数
        k_sigma: 用于带宽估计的最近邻数
    """

    def __init__(self, dim: int, reservoir: int = 4096, k_sigma: int = 20, seed: int = 0):
        """初始化在线KDE。

        参数:
            dim: 嵌入维度
            reservoir: 采样的最大水库大小
            k_sigma: 用于自适应带宽的k最近邻（默认值：20）
            seed: 水库采样的随机种子

        基准优化:
        - 较低的k_sigma（20 vs 25）产生更尖锐的惊喜峰值
        - 这通过使新颖/重要信息更易区分来帮助AR
        - 也通过使矛盾信息更突出来帮助CR
        """
        self.dim = dim
        self.reservoir = reservoir
        self.k_sigma = k_sigma
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._vecs: List[np.ndarray] = []
        self._matrix_cache: Optional[np.ndarray] = None

    def add(self, x: np.ndarray) -> None:
        """将向量添加到密度估计中。

        使用水库采样来保持有限的内存。

        参数:
            x: 要添加的嵌入向量
        """
        x = x.astype(np.float32)
        if len(self._vecs) < self.reservoir:
            self._vecs.append(x)
        else:
            # 水库采样（容量限制的内存）
            j = int(self.rng.integers(0, len(self._vecs) + 1))
            if j < len(self._vecs):
                self._vecs[j] = x
        self._matrix_cache = None

    def _matrix(self) -> Optional[np.ndarray]:
        """返回当前水库的矩阵视图。"""
        if not self._vecs:
            return None
        if self._matrix_cache is None:
            self._matrix_cache = np.asarray(self._vecs, dtype=np.float32)
        return self._matrix_cache

    def _sigma(self, x: np.ndarray) -> float:
        """使用k最近邻计算自适应带宽。

        参数:
            x: 查询向量

        返回:
            带宽估计（到k个最近邻的中位数距离）
        """
        matrix = self._matrix()
        if matrix is None:
            return 1.0

        deltas = matrix - x
        distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
        k = min(self.k_sigma, len(distances))
        if k <= 0:
            med = float(np.median(distances))
        else:
            nearest = np.partition(distances, k - 1)[:k]
            med = float(np.median(nearest))
        return med + 1e-6

    def log_density(self, x: np.ndarray) -> float:
        """计算一个点处的对数密度。

        参数:
            x: 查询向量

        返回:
            对数密度估计
        """
        matrix = self._matrix()
        if matrix is None:
            # Weak prior: very low density
            return -10.0
        sigma = self._sigma(x)
        inv2 = 1.0 / (2.0 * sigma * sigma)
        deltas = matrix - x
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        vals = np.exp(-squared_distances * inv2)
        p = float(np.mean(vals))
        return math.log(p + 1e-12)

    def surprise(self, x: np.ndarray) -> float:
        """计算一个点处的惊喜度（负对数密度）。

        冷启动保护:
        - 当样本数 < DENSITY_MIN_SAMPLES 时，KDE估计不可靠
        - 返回 DEFAULT_COLD_START_SURPRISE 以鼓励早期存储
        - 这防止丢失关键的早期信息（名称、偏好等）

        参数:
            x: 查询向量

        返回:
            惊喜度值（越高 = 越令人惊讶）
        """
        # 冷启动保护：样本不足时使用默认惊喜度
        if len(self._vecs) < DENSITY_MIN_SAMPLES:
            return DEFAULT_COLD_START_SURPRISE
        return -self.log_density(x)

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为JSON兼容的字典。"""
        return {
            "dim": self.dim,
            "reservoir": self.reservoir,
            "k_sigma": self.k_sigma,
            "seed": self._seed,
            "vecs": [v.tolist() for v in self._vecs],
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "OnlineKDE":
        """从状态字典重构。"""
        obj = cls(
            dim=d["dim"],
            reservoir=d["reservoir"],
            k_sigma=d["k_sigma"],
            seed=d["seed"],
        )
        obj._vecs = [np.array(v, dtype=np.float32) for v in d.get("vecs", [])]
        obj._matrix_cache = None
        return obj
