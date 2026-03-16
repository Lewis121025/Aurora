"""记忆检索模块。

定义记忆检索（recall）机制，基于多维度评分选择最相关的片段：
- 显著性（salience）
- 未解决度（unresolvedness）
- 激活次数（activation）
- 结构性压力（structural pressure）
- 线程/记忆结关联
- 时间邻近度（recency）
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from aurora.memory.affinity import structural_pressure
from aurora.memory.fragment import Fragment
from aurora.runtime.contracts import TraceChannel

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore


# 检索评分权重配置
SALIENCE_WEIGHT = 0.30              # 显著性权重
UNRESOLVEDNESS_WEIGHT = 0.22        # 未解决度权重
ACTIVATION_WEIGHT = 0.12            # 激活次数权重
STRUCTURAL_WEIGHT = 0.10            # 结构性压力权重
THREAD_KNOT_WEIGHT = 0.10           # 线程/记忆结关联权重
RECENCY_WEIGHT = 0.16               # 时间邻近度权重
ACTIVATION_CAP = 4.0                # 激活次数上限（用于对数缩放）
RECENCY_HALF_LIFE_HOURS = 24.0      # 时间衰减半衰期（小时）
SALIENCE_FLOOR = 0.06               # 显著性下限（低于此值的片段不参与检索）


def _recall_score(
    store: MemoryStore,
    item: Fragment,
    now_ts: float,
) -> float:
    """计算片段的检索评分。

    综合显著性、未解决度、激活次数、结构性压力、线程/记忆结关联、时间邻近度。

    Args:
        store: 记忆存储。
        item: 候选片段。
        now_ts: 当前时间戳。

    Returns:
        检索评分（0.0–1.0）。
    """
    # 时间衰减（小时）
    hours_since_touch = max(0.0, (now_ts - item.last_touched_at) / 3600.0)
    recency = math.exp(-hours_since_touch / RECENCY_HALF_LIFE_HOURS)

    # 激活次数（对数缩放）
    activation = min(math.log1p(item.activation_count) / math.log1p(ACTIVATION_CAP), 1.0)

    # 线程/记忆结关联（对数缩放）
    thread_knot = min(
        1.0,
        math.log1p(len(item.thread_ids)) * 0.5 + math.log1p(len(item.knot_ids)) * 0.6,
    )

    return (
        SALIENCE_WEIGHT * item.salience
        + UNRESOLVEDNESS_WEIGHT * item.unresolvedness
        + ACTIVATION_WEIGHT * activation
        + STRUCTURAL_WEIGHT * structural_pressure(store, item)
        + THREAD_KNOT_WEIGHT * thread_knot
        + RECENCY_WEIGHT * recency
    )


def recent_recall(
    store: MemoryStore,
    relation_id: str,
    limit: int = 8,
    now_ts: float = 0.0,
) -> tuple[Fragment, ...]:
    """执行近期记忆检索。

    从指定关系中选择显著性高于下限的片段，按检索评分排序，
    返回前 limit 个最相关的片段。

    Args:
        store: 记忆存储。
        relation_id: 关系 ID。
        limit: 返回数量上限，默认 8。
        now_ts: 当前时间戳，默认 0.0。

    Returns:
        片段元组，按检索评分降序排列。
    """
    candidates = (
        f for f in store.fragments_for_relation(relation_id)
        if f.salience >= SALIENCE_FLOOR
    )
    ranked = sorted(
        candidates,
        key=lambda item: _recall_score(store, item, now_ts),
        reverse=True,
    )
    return tuple(ranked[:limit])


def build_activation_channels(
    store: MemoryStore,
    fragments: tuple[Fragment, ...],
) -> tuple[TraceChannel, ...]:
    """基于检索片段构建激活通道。

    累加片段关联轨迹的强度 × 携带度，返回前 4 个主导通道。

    Args:
        store: 记忆存储。
        fragments: 片段列表。

    Returns:
        主导轨迹通道元组（最多 4 个）。
    """
    scores: dict[TraceChannel, float] = {}
    for fragment in fragments:
        for trace in store.traces_for_fragment(fragment.fragment_id):
            scores[trace.channel] = (
                scores.get(trace.channel, 0.0) + trace.intensity * trace.carry
            )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return tuple(channel for channel, _ in ranked[:4])
