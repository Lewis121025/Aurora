"""运行时投影模块。

定义状态投影函数，将运行时状态转换为可序列化的摘要字典：
- HealthSummary: 健康检查摘要
- StateSummary: 完整状态摘要
"""
from __future__ import annotations

from typing import TypedDict

from aurora.memory.store import MemoryStore
from aurora.relation.store import RelationStore
from aurora.runtime.state import RuntimeState


class HealthSummary(TypedDict):
    """健康摘要类型。

    Attributes:
        status: 状态（固定为 "ok"）。
        phase: 当前相位。
        turns: 转换次数。
        transitions: 相位转换次数。
    """

    status: str
    phase: str
    turns: int
    transitions: int


class StateSummary(TypedDict):
    """状态摘要类型。

    Attributes:
        phase: 当前相位。
        sleep_need: 睡眠需求（0.0–1.0）。
        active_relation_ids: 活跃关系 ID 列表。
        pending_sleep_relation_ids: 待处理关系 ID 列表。
        active_knot_ids: 活跃记忆结 ID 列表。
        anchor_thread_ids: 锚定线程 ID 列表。
        turns: 转换次数。
        memory_fragments: 记忆片段数。
        memory_traces: 记忆轨迹数。
        memory_associations: 关联边数。
        memory_threads: 记忆线程数。
        memory_knots: 记忆结数。
        relation_formations: 关系形成记录数。
        relation_moments: 关系时刻数。
        sleep_cycles: sleep 周期数。
        transitions: 相位转换次数。
    """

    phase: str
    sleep_need: float
    active_relation_ids: tuple[str, ...]
    pending_sleep_relation_ids: tuple[str, ...]
    active_knot_ids: tuple[str, ...]
    anchor_thread_ids: tuple[str, ...]
    turns: int
    memory_fragments: int
    memory_traces: int
    memory_associations: int
    memory_threads: int
    memory_knots: int
    relation_formations: int
    relation_moments: int
    sleep_cycles: int
    transitions: int


def project_health_summary(
    state: RuntimeState,
    turns: int,
    transitions: int,
) -> HealthSummary:
    """投影健康摘要。

    Args:
        state: 运行时状态。
        turns: 转换次数。
        transitions: 相位转换次数。

    Returns:
        HealthSummary: 健康摘要。
    """
    return {
        "status": "ok",
        "phase": state.metabolic.phase.value,
        "turns": turns,
        "transitions": transitions,
    }


def project_state_summary(
    state: RuntimeState,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    turns: int,
    transitions: int,
) -> StateSummary:
    """投影状态摘要。

    Args:
        state: 运行时状态。
        memory_store: 记忆存储。
        relation_store: 关系存储。
        turns: 转换次数。
        transitions: 相位转换次数。

    Returns:
        StateSummary: 状态摘要。
    """
    metabolic = state.metabolic
    orientation = state.orientation
    return {
        "phase": metabolic.phase.value,
        "sleep_need": round(metabolic.sleep_need, 4),
        "active_relation_ids": metabolic.active_relation_ids,
        "pending_sleep_relation_ids": metabolic.pending_sleep_relation_ids,
        "active_knot_ids": metabolic.active_knot_ids,
        "anchor_thread_ids": orientation.anchor_thread_ids,
        "turns": turns,
        "memory_fragments": len(memory_store.fragments),
        "memory_traces": len(memory_store.traces),
        "memory_associations": len(memory_store.associations),
        "memory_threads": len(memory_store.threads),
        "memory_knots": len(memory_store.knots),
        "relation_formations": len(relation_store.formations),
        "relation_moments": relation_store.moment_count(),
        "sleep_cycles": memory_store.sleep_cycles,
        "transitions": transitions,
    }
