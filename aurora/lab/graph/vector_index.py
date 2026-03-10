"""AURORA 向量索引 - 本地精确检索实现。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.utils.math_utils import cosine_sim


class VectorIndex:
    """支持类型过滤的本地精确向量索引。"""

    def __init__(self, dim: int):
        """初始化向量索引。

        参数:
            dim: 向量维度
        """
        self.dim = dim
        self.ids: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.kinds: List[str] = []

    def add(self, _id: str, vec: np.ndarray, kind: str) -> None:
        """添加向量到索引。

        参数:
            _id: 向量 ID
            vec: 向量数据
            kind: 向量类型
        """
        vec = vec.astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"vector dim mismatch: {vec.shape} vs {(self.dim,)}")
        self.ids.append(_id)
        self.vecs.append(vec)
        self.kinds.append(kind)

    def remove(self, _id: str) -> None:
        """从索引中移除向量。

        参数:
            _id: 要移除的向量 ID
        """
        if _id not in self.ids:
            return
        i = self.ids.index(_id)
        self.ids.pop(i)
        self.vecs.pop(i)
        self.kinds.pop(i)

    def search(
        self, q: np.ndarray, k: int = 10, kind: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """搜索最相似的向量。

        参数:
            q: 查询向量
            k: 返回的结果数量
            kind: 可选的类型过滤

        返回:
            (向量ID, 相似度) 的列表，按相似度降序排列
        """
        if not self.vecs:
            return []
        q = q.astype(np.float32)
        hits = [
            (_id, cosine_sim(q, v))
            for _id, v, kd in zip(self.ids, self.vecs, self.kinds)
            if kind is None or kd == kind
        ]
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]

    def to_state_dict(self) -> Dict[str, Any]:
        """将索引转换为状态字典。

        返回:
            包含索引状态的字典
        """
        return {
            "dim": self.dim,
            "ids": self.ids,
            "vecs": [v.tolist() for v in self.vecs],
            "kinds": self.kinds,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "VectorIndex":
        """从状态字典恢复索引。

        参数:
            d: 状态字典

        返回:
            恢复的 VectorIndex 实例
        """
        obj = cls(dim=d["dim"])
        obj.ids = d["ids"]
        obj.vecs = [np.array(v, dtype=np.float32) for v in d["vecs"]]
        obj.kinds = d["kinds"]
        return obj
