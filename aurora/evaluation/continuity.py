"""连续性评估模块。

评估 Aurora 运行时状态的连续性，确保：
- 活跃关系和记忆结在存储中可追溯
- 锚定线程存在
- 状态转换时间戳单调递增
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.store import MemoryStore
from aurora.relation.store import RelationStore
from aurora.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class ContinuityCheck:
    """连续性检查结果。

    Attributes:
        active_relations_known: 所有活跃关系是否已知。
        active_knots_known: 所有活跃记忆结是否已知。
        anchor_threads_known: 所有锚定线程是否已知。
        transitions_monotonic: 状态转换时间戳是否单调递增。
        ok: 四项检查是否全部通过。
    """

    active_relations_known: bool
    active_knots_known: bool
    anchor_threads_known: bool
    transitions_monotonic: bool

    @property
    def ok(self) -> bool:
        """四项检查全部通过时返回 True。"""
        return (
            self.active_relations_known
            and self.active_knots_known
            and self.anchor_threads_known
            and self.transitions_monotonic
        )


def evaluate_continuity(
    state: RuntimeState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
) -> ContinuityCheck:
    """评估运行时状态的连续性。

    检查活跃关系、记忆结、锚定线程是否在存储中存在，
    并验证状态转换时间戳的单调性。

    Args:
        state: 运行时状态。
        memory_store: 记忆存储。
        relation_store: 关系存储。

    Returns:
        ContinuityCheck: 连续性检查结果。
    """
    # 检查活跃关系是否在关系存储或记忆碎片中已知
    known_relations = set(relation_store.formations) | set(memory_store.relation_fragments)
    active_relations_known = all(
        relation_id in known_relations for relation_id in state.metabolic.active_relation_ids
    )

    # 检查活跃记忆结是否在记忆存储中存在
    active_knots_known = all(
        knot_id in memory_store.knots for knot_id in state.metabolic.active_knot_ids
    )

    # 检查锚定线程是否在记忆存储中存在
    anchor_threads_known = all(
        thread_id in memory_store.threads for thread_id in state.orientation.anchor_thread_ids
    )

    # 检查状态转换时间戳是否单调递增
    transition_times = [item.created_at for item in state.transitions]
    transitions_monotonic = transition_times == sorted(transition_times)

    return ContinuityCheck(
        active_relations_known=active_relations_known,
        active_knots_known=active_knots_known,
        anchor_threads_known=anchor_threads_known,
        transitions_monotonic=transitions_monotonic,
    )
