"""关系存储模块。

实现关系存储（RelationStore），管理：
- 关系形成记录（RelationFormation）
- 关系时刻列表（RelationMoment）

提供关系级别的操作：记录交互、吸收 sleep 结果、查询统计。
"""
from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from aurora.relation.formation import RelationFormation
from aurora.relation.moment import RelationMoment
from aurora.runtime.contracts import AuroraMove, TraceChannel


MOMENT_CAP = 128  # 每个关系最多保留的关系时刻数


class RelationStore:
    """关系存储。

    管理所有关系的形成记录和时刻历史，支持：
    - 记录交互交换
    - 吸收 sleep 阶段结果
    - 脏标记追踪（用于持久化）

    Attributes:
        formations: 关系形成记录字典。
        moments: 关系时刻列表字典（按关系 ID 分组）。
        _dirty_formations: 脏标记的形成记录关系 ID。
        _dirty_moment_relations: 脏标记的时刻关系 ID。
    """

    def __init__(self) -> None:
        self.formations: dict[str, RelationFormation] = {}
        self.moments: dict[str, list[RelationMoment]] = defaultdict(list)
        self._dirty_formations: set[str] = set()
        self._dirty_moment_relations: set[str] = set()

    def formation_for(self, relation_id: str) -> RelationFormation:
        """获取或创建关系形成记录。

        Args:
            relation_id: 关系 ID。

        Returns:
            RelationFormation: 关系形成记录。
        """
        if relation_id not in self.formations:
            self.formations[relation_id] = RelationFormation(relation_id=relation_id)
            self._dirty_formations.add(relation_id)
        return self.formations[relation_id]

    def record_exchange(
        self,
        relation_id: str,
        user_turn_id: str,
        aurora_turn_id: str | None,
        user_channels: tuple[TraceChannel, ...],
        aurora_move: AuroraMove,
        summary: str,
        now_ts: float,
    ) -> RelationMoment:
        """记录交互交换。

        创建关系时刻记录，更新形成记录，维护时刻列表大小。

        Args:
            relation_id: 关系 ID。
            user_turn_id: 用户转换 ID。
            aurora_turn_id: Aurora 转换 ID。
            user_channels: 用户触发的通道列表。
            aurora_move: Aurora 行为选择。
            summary: 交互摘要。
            now_ts: 当前时间戳。

        Returns:
            RelationMoment: 创建的关系时刻。
        """
        # 规范化通道（去重排序）
        normalized_channels = tuple(sorted(set(user_channels), key=lambda c: c.value))

        # 自动推导边界/修复事件
        boundary_event = TraceChannel.BOUNDARY in normalized_channels or aurora_move == "boundary"
        repair_event = TraceChannel.REPAIR in normalized_channels or aurora_move == "repair"

        moment = RelationMoment(
            moment_id=f"moment_{uuid4().hex[:12]}",
            relation_id=relation_id,
            user_turn_id=user_turn_id,
            aurora_turn_id=aurora_turn_id,
            user_channels=normalized_channels,
            aurora_move=aurora_move,
            boundary_event=boundary_event,
            repair_event=repair_event,
            summary=summary,
            created_at=now_ts,
        )

        # 添加到时刻列表，维护容量上限
        bucket = self.moments[relation_id]
        bucket.append(moment)
        if len(bucket) > MOMENT_CAP:
            del bucket[: len(bucket) - MOMENT_CAP]

        # 更新形成记录
        self.formation_for(relation_id).register_moment(moment)
        self._dirty_formations.add(relation_id)
        self._dirty_moment_relations.add(relation_id)

        return moment

    def absorb_sleep(
        self,
        relation_id: str,
        thread_ids: tuple[str, ...],
        knot_ids: tuple[str, ...],
        now_ts: float,
    ) -> None:
        """吸收 sleep 阶段结果。

        Args:
            relation_id: 关系 ID。
            thread_ids: 线程 ID 列表。
            knot_ids: 记忆结 ID 列表。
            now_ts: 当前时间戳。
        """
        self.formation_for(relation_id).absorb_sleep(
            thread_ids=thread_ids,
            knot_ids=knot_ids,
            now_ts=now_ts,
        )
        self._dirty_formations.add(relation_id)

    def relation_count(self) -> int:
        """获取关系数量。

        Returns:
            关系形成记录数。
        """
        return len(self.formations)

    def moment_count(self) -> int:
        """获取关系时刻总数。

        Returns:
            所有关系的时刻数之和。
        """
        return sum(len(items) for items in self.moments.values())

    def clear_dirty(self) -> None:
        """清空脏标记。

        在持久化完成后调用。
        """
        self._dirty_formations.clear()
        self._dirty_moment_relations.clear()
