"""关系存储模块。

简化版关系存储，仅保留必要接口。
"""

from __future__ import annotations

from collections import defaultdict


class RelationStore:
    """简化关系存储。

    Attributes:
        moments: 关系时刻字典。
    """

    def __init__(self) -> None:
        self.moments: dict[str, list[dict]] = defaultdict(list)

    def record_exchange(
        self,
        relation_id: str,
        user_text: str,
        aurora_text: str,
        aurora_move: str,
        now_ts: float,
    ) -> None:
        """记录交互交换。"""
        self.moments[relation_id].append(
            {
                "user_text": user_text,
                "aurora_text": aurora_text,
                "aurora_move": aurora_move,
                "created_at": now_ts,
            }
        )

    def moments_for_relation(self, relation_id: str) -> list[dict]:
        """获取关系时刻。"""
        return self.moments.get(relation_id, [])
