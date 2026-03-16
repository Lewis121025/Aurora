"""关系形成模块。

定义关系形成记录（RelationFormation），追踪关系的长期结构：
- 关联的线程和记忆结
- 边界事件和修复事件计数
- 共鸣事件计数
- 最后接触时间
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aurora.relation.moment import RelationMoment


@dataclass(slots=True)
class RelationFormation:
    """关系形成记录。

    记录关系的长期演化历史，包括：
    - 整合的记忆结构（线程、记忆结）
    - 关系动态事件（边界、修复、共鸣）
    - 接触时间戳

    Attributes:
        relation_id: 关系唯一 ID。
        thread_ids: 关联的线程 ID 集合。
        knot_ids: 关联的记忆结 ID 集合。
        boundary_events: 边界事件计数。
        repair_events: 修复事件计数。
        resonance_events: 共鸣事件计数。
        last_contact_at: 最后接触时间戳。
    """

    relation_id: str
    thread_ids: set[str] = field(default_factory=set)
    knot_ids: set[str] = field(default_factory=set)
    boundary_events: int = 0
    repair_events: int = 0
    resonance_events: int = 0
    last_contact_at: float = 0.0

    def register_moment(self, moment: RelationMoment) -> None:
        """注册关系时刻。

        根据时刻内容更新事件计数和最后接触时间。

        Args:
            moment: 关系时刻对象。
        """
        if moment.boundary_event:
            self.boundary_events += 1
        if moment.repair_event:
            self.repair_events += 1
        # approach/repair/witness 视为共鸣行为
        if moment.aurora_move in {"approach", "repair", "witness"}:
            self.resonance_events += 1
        self.last_contact_at = max(self.last_contact_at, moment.created_at)

    def absorb_sleep(
        self,
        thread_ids: tuple[str, ...],
        knot_ids: tuple[str, ...],
        now_ts: float,
    ) -> None:
        """吸收 sleep 阶段结果。

        将 sleep 创建的线程和记忆结纳入关系形成记录。

        Args:
            thread_ids: 线程 ID 列表。
            knot_ids: 记忆结 ID 列表。
            now_ts: 当前时间戳。
        """
        self.thread_ids.update(thread_ids)
        self.knot_ids.update(knot_ids)
        self.last_contact_at = max(self.last_contact_at, now_ts)
