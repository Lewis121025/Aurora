"""记忆沉积模块。

实现记忆沉积（sediment）清理机制，移除长期未被激活的低显著性记忆：
- 片段：显著性和未解决度低于下限，或超过 168 小时（7 天）未触碰
- 轨迹：随片段一起清理
- 关联边：两端片段都被清理时移除
- 线程/记忆结：所有成员片段都被清理时移除
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aurora.memory.recall import SALIENCE_FLOOR

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore
    from aurora.memory.fragment import Fragment


UNRESOLVED_FLOOR = 0.06           # 未解决度下限
STALENESS_HOURS = 168.0           # 陈旧阈值（小时，7 天）
ASSOCIATION_WEIGHT_FLOOR = 0.10   # 关联边权重下限


@dataclass(frozen=True, slots=True)
class SedimentResult:
    """沉积清理结果。

    Attributes:
        removed_fragment_ids: 被移除的片段 ID 列表。
        removed_trace_ids: 被移除的轨迹 ID 列表。
        removed_association_ids: 被移除的关联边 ID 列表。
        removed_thread_ids: 被移除的线程 ID 列表。
        removed_knot_ids: 被移除的记忆结 ID 列表。
    """

    removed_fragment_ids: tuple[str, ...]
    removed_trace_ids: tuple[str, ...]
    removed_association_ids: tuple[str, ...]
    removed_thread_ids: tuple[str, ...]
    removed_knot_ids: tuple[str, ...]


def _is_sediment_candidate(fragment: "Fragment", now_ts: float) -> bool:
    """判断片段是否为沉积清理候选。

    条件：
    - 显著性和未解决度均低于下限，或
    - 超过 168 小时（7 天）未触碰

    Args:
        fragment: 候选片段。
        now_ts: 当前时间戳。

    Returns:
        是否为清理候选。
    """
    if fragment.durability >= 0.5:
        return False
    hours_stale = (now_ts - fragment.last_touched_at) / 3600.0
    staleness_threshold = STALENESS_HOURS * (1.0 + fragment.durability * 3.0)
    below_floor = (
        fragment.salience < SALIENCE_FLOOR
        and fragment.unresolvedness < UNRESOLVED_FLOOR
    )
    return below_floor or hours_stale >= staleness_threshold


def sediment(store: MemoryStore, now_ts: float) -> SedimentResult:
    """执行记忆沉积清理。

    清理顺序：
    1. 识别候选片段
    2. 移除所有成员均为候选的线程和记忆结
    3. 移除不再被线程/记忆结引用的片段
    4. 移除随片段清理而孤立的轨迹和关联边

    Args:
        store: 记忆存储。
        now_ts: 当前时间戳。

    Returns:
        SedimentResult: 清理结果。
    """
    # 识别候选片段
    candidate_fids = {
        fid for fid, f in store.fragments.items() if _is_sediment_candidate(f, now_ts)
    }

    # 移除所有成员均为候选的线程
    remove_thread_ids: list[str] = []
    for thread in list(store.threads.values()):
        if set(thread.fragment_ids) <= candidate_fids:
            remove_thread_ids.append(thread.thread_id)

    # 移除所有成员均为候选的记忆结
    remove_knot_ids: list[str] = []
    for knot in list(store.knots.values()):
        if set(knot.fragment_ids) <= candidate_fids:
            remove_knot_ids.append(knot.knot_id)

    # 执行线程/记忆结移除
    for thread_id in remove_thread_ids:
        store.remove_thread(thread_id)
    for knot_id in remove_knot_ids:
        store.remove_knot(knot_id)

    # 计算仍被引用的片段
    active_fids: set[str] = set()
    for thread in store.threads.values():
        active_fids.update(thread.fragment_ids)
    for knot in store.knots.values():
        active_fids.update(knot.fragment_ids)

    # 移除未被引用的候选片段
    remove_fids = [fid for fid in candidate_fids if fid not in active_fids]

    # 收集待移除的轨迹
    remove_tids: list[str] = []
    for fid in remove_fids:
        for tid in list(store.fragment_traces.get(fid, ())):
            remove_tids.append(tid)

    # 收集待移除的关联边
    removed_fid_set = set(remove_fids)
    remove_eids: list[str] = []
    for eid, edge in list(store.associations.items()):
        both_gone = (
            edge.src_fragment_id in removed_fid_set
            and edge.dst_fragment_id in removed_fid_set
        )
        one_gone = (
            edge.src_fragment_id in removed_fid_set
            or edge.dst_fragment_id in removed_fid_set
        )
        # 两端都被清理，或一端被清理且权重低于下限
        if both_gone or (one_gone and edge.weight < ASSOCIATION_WEIGHT_FLOOR):
            remove_eids.append(eid)

    # 执行清理
    for tid in remove_tids:
        store.remove_trace(tid)
    for eid in remove_eids:
        store.remove_association(eid)
    for fid in remove_fids:
        store.remove_fragment(fid)

    return SedimentResult(
        removed_fragment_ids=tuple(remove_fids),
        removed_trace_ids=tuple(remove_tids),
        removed_association_ids=tuple(remove_eids),
        removed_thread_ids=tuple(remove_thread_ids),
        removed_knot_ids=tuple(remove_knot_ids),
    )
