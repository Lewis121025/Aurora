from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING, Iterable

from aurora.memory.fragment import Fragment
from aurora.runtime.contracts import TraceChannel

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore


# 亲和度权重配置
AFFINITY_RELATION_WEIGHT = 0.30   # 关系一致性权重
AFFINITY_KEYWORD_WEIGHT = 0.24    # 关键词重叠权重
AFFINITY_TRACE_WEIGHT = 0.26      # 轨迹通道重叠权重
AFFINITY_TEMPORAL_WEIGHT = 0.20   # 时间邻近度权重
TEMPORAL_HALF_LIFE_HOURS = 16.0   # 时间衰减半衰期（小时）


def fragment_affinity(store: MemoryStore, left: Fragment, right: Fragment) -> float:
    """计算两个记忆片段之间的亲和度。
    
    综合四个维度：关系一致性、关键词重叠、轨迹通道重叠、时间邻近度。
    返回值为 0.0–1.0，越高表示两个片段越相关。
    """
    # 时间距离（小时），用于计算时间衰减
    temporal_distance = abs(left.created_at - right.created_at) / 3600.0
    temporal = math.exp(-temporal_distance / TEMPORAL_HALF_LIFE_HOURS)
    return (
        AFFINITY_RELATION_WEIGHT * float(left.relation_id == right.relation_id)
        + AFFINITY_KEYWORD_WEIGHT * keyword_overlap(left.tags, right.tags)
        + AFFINITY_TRACE_WEIGHT * trace_overlap(store, left.fragment_id, right.fragment_id)
        + AFFINITY_TEMPORAL_WEIGHT * temporal
    )


def keyword_overlap(left_tags: Iterable[str], right_tags: Iterable[str]) -> float:
    """计算两个标签集合的 Jaccard 相似度。
    
    返回值为 0.0–1.0，空集合返回 0.0。
    """
    left = set(left_tags)
    right = set(right_tags)
    return len(left & right) / max(1, len(left | right)) if (left or right) else 0.0


def trace_overlap(store: MemoryStore, left_fragment_id: str, right_fragment_id: str) -> float:
    """计算两个片段的轨迹通道 Jaccard 相似度。
    
    轨迹通道表示认知过程中触发的通道类型（如 touch/move/response）。
    返回值为 0.0–1.0，空集合返回 0.0。
    """
    left = {trace.channel for trace in store.traces_for_fragment(left_fragment_id)}
    right = {trace.channel for trace in store.traces_for_fragment(right_fragment_id)}
    return len(left & right) / max(1, len(left | right)) if (left or right) else 0.0


def structural_pressure(store: MemoryStore, fragment: Fragment) -> float:
    """计算片段的结构性压力。
    
    基于片段已连接的边数，反映该片段在记忆网络中的连接饱和度。
    6 条边时达到饱和（返回 1.0），用于抑制过度连接的片段继续建立关联。
    """
    return min(1.0, len(store.fragment_edges.get(fragment.fragment_id, ())) / 6.0)


def cluster_keyword_overlap(cluster: list[Fragment]) -> float:
    """计算簇内所有片段对的平均关键词重叠度。
    
    用于评估簇的语义一致性。簇大小 < 2 时返回 0.0。
    """
    if len(cluster) < 2:
        return 0.0
    overlaps = [
        keyword_overlap(left.tags, right.tags) for left, right in combinations(cluster, 2)
    ]
    return sum(overlaps) / len(overlaps)


def cluster_trace_overlap(store: MemoryStore, cluster: list[Fragment]) -> float:
    """计算簇内所有片段对的平均轨迹通道重叠度。
    
    用于评估簇的认知模式一致性。簇大小 < 2 时返回 0.0。
    """
    if len(cluster) < 2:
        return 0.0
    overlaps: list[float] = []
    for left, right in combinations(cluster, 2):
        left_channels = {trace.channel for trace in store.traces_for_fragment(left.fragment_id)}
        right_channels = {trace.channel for trace in store.traces_for_fragment(right.fragment_id)}
        overlaps.append(
            len(left_channels & right_channels) / max(1, len(left_channels | right_channels))
        )
    return sum(overlaps) / len(overlaps)


def cluster_dominant_channels(
    store: MemoryStore, cluster: list[Fragment]
) -> tuple[TraceChannel, ...]:
    """提取簇的主导轨迹通道。
    
    累加簇内所有片段的轨迹强度，返回强度最高的前 2 个通道。
    用于标识簇的认知特征。
    """
    totals: dict[TraceChannel, float] = {}
    for fragment in cluster:
        for trace in store.traces_for_fragment(fragment.fragment_id):
            totals[trace.channel] = totals.get(trace.channel, 0.0) + trace.intensity
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return tuple(channel for channel, _ in ranked[:2])


def neighbor_fragment_ids(store: MemoryStore, fragment_id: str) -> tuple[str, ...]:
    """获取指定片段的直接邻居片段 ID。
    
    通过关联边（Association）查找相邻片段，保持顺序且去重。
    """
    neighbors: list[str] = []
    for edge_id in store.fragment_edges.get(fragment_id, ()):
        edge = store.associations[edge_id]
        neighbor_id = (
            edge.dst_fragment_id if edge.src_fragment_id == fragment_id else edge.src_fragment_id
        )
        if neighbor_id not in neighbors:
            neighbors.append(neighbor_id)
    return tuple(neighbors)


def region_edge_density(store: MemoryStore, fragment_ids: tuple[str, ...]) -> float:
    """计算区域的边密度。
    
    密度 = 实际存在的边数 / 可能的最大边数。
    用于评估区域的结构紧密程度，值为 0.0–1.0。
    """
    if len(fragment_ids) < 2:
        return 0.0
    linked = 0
    total = 0
    for left_id, right_id in combinations(fragment_ids, 2):
        total += 1
        if store._existing_edge_id(left_id, right_id) is not None:
            linked += 1
    return linked / total if total else 0.0
