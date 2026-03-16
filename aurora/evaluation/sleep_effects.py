"""睡眠效应评估模块。

评估 sleep 阶段的效果，通过比较 sleep 前后的状态快照，
验证：
- sleep 周期是否增加
- 待处理关系队列是否被清空
- 记忆结构（线程/记忆结）的变化
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.store import MemoryStore
from aurora.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class SleepSnapshot:
    """睡眠前/后的状态快照。

    Attributes:
        sleep_cycles: 睡眠周期计数。
        thread_count: 线程总数。
        knot_count: 记忆结总数。
        anchor_thread_count: 锚定线程数。
        active_knot_count: 活跃记忆结数。
        pending_sleep_relations: 待处理关系数。
    """

    sleep_cycles: int
    thread_count: int
    knot_count: int
    anchor_thread_count: int
    active_knot_count: int
    pending_sleep_relations: int


@dataclass(frozen=True, slots=True)
class SleepEffectsCheck:
    """睡眠效应检查结果。

    通过比较 sleep 前后快照，评估 sleep 阶段的效果。

    Attributes:
        sleep_cycles_delta: 睡眠周期增量。
        thread_delta: 线程数量变化。
        knot_delta: 记忆结数量变化。
        anchor_thread_delta: 锚定线程变化。
        active_knot_delta: 活跃记忆结变化。
        pending_sleep_cleared: 待处理队列是否被清空。
        ok: 检查是否通过（周期增加且队列清空）。
    """

    sleep_cycles_delta: int
    thread_delta: int
    knot_delta: int
    anchor_thread_delta: int
    active_knot_delta: int
    pending_sleep_cleared: bool

    @property
    def ok(self) -> bool:
        """检查通过条件：至少一个 sleep 周期且待处理队列已清空。"""
        return self.sleep_cycles_delta >= 1 and self.pending_sleep_cleared


def snapshot_sleep_state(state: RuntimeState, memory_store: MemoryStore) -> SleepSnapshot:
    """生成睡眠状态快照。

    捕获 sleep 阶段前后的关键状态指标。

    Args:
        state: 运行时状态。
        memory_store: 记忆存储。

    Returns:
        SleepSnapshot: 状态快照。
    """
    return SleepSnapshot(
        sleep_cycles=memory_store.sleep_cycles,
        thread_count=len(memory_store.threads),
        knot_count=len(memory_store.knots),
        anchor_thread_count=len(state.orientation.anchor_thread_ids),
        active_knot_count=len(state.metabolic.active_knot_ids),
        pending_sleep_relations=len(state.metabolic.pending_sleep_relation_ids),
    )


def evaluate_sleep_effects(before: SleepSnapshot, after: SleepSnapshot) -> SleepEffectsCheck:
    """评估睡眠效应。

    比较 sleep 前后快照，计算各项指标的变化量。

    Args:
        before: sleep 前的状态快照。
        after: sleep 后的状态快照。

    Returns:
        SleepEffectsCheck: 睡眠效应检查结果。
    """
    return SleepEffectsCheck(
        sleep_cycles_delta=after.sleep_cycles - before.sleep_cycles,
        thread_delta=after.thread_count - before.thread_count,
        knot_delta=after.knot_count - before.knot_count,
        anchor_thread_delta=after.anchor_thread_count - before.anchor_thread_count,
        active_knot_delta=after.active_knot_count - before.active_knot_count,
        pending_sleep_cleared=before.pending_sleep_relations >= after.pending_sleep_relations
        and after.pending_sleep_relations == 0,
    )
