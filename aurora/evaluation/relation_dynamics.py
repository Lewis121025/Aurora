"""关系动力学评估模块。

评估关系的状态和演化，包括：
- 关系时刻数量
- 边界事件和修复事件计数
- 关联的线程和记忆结数量
- 关系是否有记忆碎片
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.store import MemoryStore
from aurora.relation.store import RelationStore


@dataclass(frozen=True, slots=True)
class RelationDynamicsCheck:
    """关系动力学检查结果。

    Attributes:
        moment_count: 关系时刻数量。
        boundary_events: 边界事件计数。
        repair_events: 修复事件计数。
        linked_threads: 关联的线程数量。
        linked_knots: 关联的记忆结数量。
        relation_has_memory: 关系是否有记忆碎片。
        ok: 检查是否通过（至少一个时刻且有记忆）。
    """

    moment_count: int
    boundary_events: int
    repair_events: int
    linked_threads: int
    linked_knots: int
    relation_has_memory: bool

    @property
    def ok(self) -> bool:
        """检查通过条件：至少一个关系时刻且关系有记忆碎片。"""
        return self.moment_count >= 1 and self.relation_has_memory


def evaluate_relation_dynamics(
    relation_store: RelationStore,
    memory_store: MemoryStore,
    relation_id: str,
) -> RelationDynamicsCheck:
    """评估指定关系的动力学状态。

    统计关系的时刻数量、事件计数、关联的记忆结构，
    并检查关系是否有对应的记忆碎片。

    Args:
        relation_store: 关系存储。
        memory_store: 记忆存储。
        relation_id: 关系 ID。

    Returns:
        RelationDynamicsCheck: 关系动力学检查结果。
    """
    formation = relation_store.formation_for(relation_id)
    return RelationDynamicsCheck(
        moment_count=len(relation_store.moments.get(relation_id, ())),
        boundary_events=formation.boundary_events,
        repair_events=formation.repair_events,
        linked_threads=len(formation.thread_ids),
        linked_knots=len(formation.knot_ids),
        relation_has_memory=bool(memory_store.fragments_for_relation(relation_id)),
    )
