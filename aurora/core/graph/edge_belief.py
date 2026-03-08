"""
AURORA 边信念
===================

使用 Beta 后验的概率边强度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from aurora.utils.time_utils import now_ts


@dataclass
class EdgeBelief:
    """边有用性后验：Beta(a,b)。

    使用共轭 Beta-Bernoulli 模型
    对遍历此边是否有帮助的概率进行建模。

    属性：
        edge_type: 此边代表的关系类型
        a: Alpha 参数（成功 + 1）
        b: Beta 参数（失败 + 1）
        use_count: 此边被使用的次数
        last_used_ts: 最后使用的时间戳
    """

    edge_type: str
    a: float = 1.0
    b: float = 1.0
    use_count: int = 0
    last_used_ts: float = field(default_factory=now_ts)

    def mean(self) -> float:
        """获取预期有用性（Beta 后验平均值）。

        返回：
            (0, 1) 中的有用性预期概率
        """
        return self.a / (self.a + self.b)

    def update(self, success: bool) -> None:
        """根据结果更新信念。

        参数：
            success: 遍历此边是否有帮助
        """
        self.use_count += 1
        self.last_used_ts = now_ts()
        if success:
            self.a += 1.0
        else:
            self.b += 1.0

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "edge_type": self.edge_type,
            "a": self.a,
            "b": self.b,
            "use_count": self.use_count,
            "last_used_ts": self.last_used_ts,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "EdgeBelief":
        """从状态字典重建。"""
        return cls(
            edge_type=d["edge_type"],
            a=d["a"],
            b=d["b"],
            use_count=d["use_count"],
            last_used_ts=d["last_used_ts"],
        )
