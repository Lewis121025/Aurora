"""doze 状态记忆操作模块。

定义 doze 阶段的记忆维护操作：
- hover：提升活跃片段及其邻居的显著性
- decay：随时间衰减片段的显著性和未解决度
- trace 衰减：轨迹强度随时间衰减
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from aurora.memory.affinity import neighbor_fragment_ids
from aurora.memory.recall import recent_recall
from aurora.runtime.contracts import clamp

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore


# doze 阶段的显著性/未解决度调整量
HOVER_SALIENCE_DELTA = 0.02           # 活跃片段显著性提升
HOVER_UNRESOLVED_DELTA = -0.01        # 活跃片段未解决度降低
HOVER_NEIGHBOR_SALIENCE_DELTA = 0.01  # 邻居片段显著性提升
HOVER_NEIGHBOR_UNRESOLVED_DELTA = -0.005  # 邻居片段未解决度降低
HOVER_THREAD_SALIENCE_DELTA = 0.015   # 线程内片段显著性提升
HOVER_THREAD_UNRESOLVED_DELTA = -0.01  # 线程内片段未解决度降低

# 衰减参数
DECAY_SALIENCE_RATE = 0.012           # 显著性衰减率（每小时）
DECAY_SALIENCE_CAP = 0.12             # 显著性最大衰减量
DECAY_UNRESOLVED_RATE = 0.01          # 未解决度衰减率（每小时）
DECAY_UNRESOLVED_CAP = 0.09           # 未解决度最大衰减量
TRACE_DECAY_HOUR_DIVISOR = 48.0       # 轨迹衰减时间除数
TRACE_DECAY_CAP = 0.2                 # 轨迹最大衰减量
VIVIDNESS_RESISTANCE = 0.5            # 生动度对衰减的阻力系数


def hover_for_doze(store: MemoryStore, relation_ids: tuple[str, ...], now_ts: float) -> None:
    """doze 状态的悬停操作。

    对指定关系的最近记忆片段及其邻居进行显著性提升，
    模拟 doze 状态下的记忆维护过程。

    Args:
        store: 记忆存储。
        relation_ids: 关系 ID 列表。
        now_ts: 当前时间戳。
    """
    for relation_id in relation_ids:
        # 获取最近回忆的片段
        recalled = recent_recall(store, relation_id=relation_id, limit=4, now_ts=now_ts)

        # 提升前 2 个片段及其邻居
        for fragment in recalled[:2]:
            store.set_fragment(
                fragment.fragment_id,
                store.fragments[fragment.fragment_id].touched(
                    at=now_ts,
                    delta_salience=HOVER_SALIENCE_DELTA,
                    delta_unresolved=HOVER_UNRESOLVED_DELTA,
                )
            )
            # 提升邻居片段
            for nid in neighbor_fragment_ids(store, fragment.fragment_id)[:2]:
                store.set_fragment(
                    nid,
                    store.fragments[nid].touched(
                        at=now_ts,
                        delta_salience=HOVER_NEIGHBOR_SALIENCE_DELTA,
                        delta_unresolved=HOVER_NEIGHBOR_UNRESOLVED_DELTA,
                    )
                )

        # 提升最新线程内的片段
        threads = store.threads_for_relation(relation_id)
        if threads:
            recent_thread = max(threads, key=lambda item: item.last_rewoven_at)
            for fragment_id in recent_thread.fragment_ids[:2]:
                if fragment_id in store.fragments:
                    store.set_fragment(
                        fragment_id,
                        store.fragments[fragment_id].touched(
                            at=now_ts,
                            delta_salience=HOVER_THREAD_SALIENCE_DELTA,
                            delta_unresolved=HOVER_THREAD_UNRESOLVED_DELTA,
                        )
                    )


def decay_for_doze(
    store: MemoryStore,
    now_ts: float,
    relation_ids: tuple[str, ...] = (),
) -> None:
    """doze 状态的衰减操作。

    对指定关系的记忆片段和轨迹进行时间衰减，
    模拟记忆随时间自然淡化的过程。

    生动度（vividness）高的片段衰减更慢（阻力更大）。

    Args:
        store: 记忆存储。
        now_ts: 当前时间戳。
        relation_ids: 关系 ID 列表（为空时不执行衰减）。
    """
    # 收集目标片段 ID
    target_fids: set[str] = set()
    for rel_id in relation_ids:
        target_fids.update(store.relation_fragments.get(rel_id, ()))
    if not target_fids:
        return

    # 衰减片段
    for fragment_id in target_fids:
        fragment = store.fragments.get(fragment_id)
        if fragment is None:
            continue

        hours = max(0.0, (now_ts - fragment.last_touched_at) / 3600.0)
        # 生动度提供衰减阻力
        resistance = 1.0 - fragment.vividness * VIVIDNESS_RESISTANCE
        salience_drop = min(DECAY_SALIENCE_CAP, hours * DECAY_SALIENCE_RATE * resistance)
        unresolved_drop = min(DECAY_UNRESOLVED_CAP, hours * DECAY_UNRESOLVED_RATE * resistance)

        store.set_fragment(
            fragment_id,
            replace(
                fragment,
                salience=clamp(fragment.salience - salience_drop),
                unresolvedness=clamp(fragment.unresolvedness - unresolved_drop),
                last_touched_at=now_ts,
            )
        )

    # 衰减轨迹
    target_tids: set[str] = set()
    for fid in target_fids:
        target_tids.update(store.fragment_traces.get(fid, ()))

    for trace_id in target_tids:
        trace = store.traces.get(trace_id)
        if trace is None:
            continue
        hours = max(0.0, (now_ts - trace.last_touched_at) / 3600.0)
        next_intensity = clamp(
            trace.intensity - trace.carry * min(TRACE_DECAY_CAP, hours / TRACE_DECAY_HOUR_DIVISOR)
        )
        store.set_trace(trace_id, replace(trace, intensity=next_intensity, last_touched_at=now_ts))
