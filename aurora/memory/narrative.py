"""叙事重组模块。

sleep 阶段的 post-reweave 处理，将记忆从"高分片段分组"提升为"叙事整合"：
- 冲突→修复弧检测：boundary 后接 repair 的 knot 标记为 resolved
- 持久锚点提升：跨线程出现的高持久度片段获得结构加固
- 主题收敛：共享主导通道且片段高度重叠的线程合并为长期主题
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from aurora.memory.knot import Knot
from aurora.runtime.contracts import TraceChannel, clamp

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore
    from aurora.relation.store import RelationStore

THEME_CHANNEL_OVERLAP = 0.5
THEME_FRAGMENT_OVERLAP = 0.4
DURABILITY_ELEVATION_THRESHOLD = 0.3
CROSS_THREAD_DURABILITY_BOOST = 0.15


@dataclass(frozen=True, slots=True)
class NarrativeResult:
    """叙事重组结果。

    Attributes:
        resolved_knot_ids: 标记为已修复的 knot ID。
        elevated_fragment_ids: 持久度被提升的片段 ID。
        merged_thread_pairs: 被合并的线程对（保留 ID, 移除 ID）。
        thread_remap: 被移除线程 ID → 存活线程 ID 映射。
    """

    resolved_knot_ids: tuple[str, ...]
    elevated_fragment_ids: tuple[str, ...]
    merged_thread_pairs: tuple[tuple[str, str], ...]
    thread_remap: dict[str, str] = field(default_factory=dict)


def restructure(
    memory_store: MemoryStore,
    relation_store: RelationStore,
    relation_id: str,
    now_ts: float,
) -> NarrativeResult:
    """对指定关系执行叙事重组。

    Args:
        memory_store: 记忆存储。
        relation_store: 关系存储。
        relation_id: 关系 ID。
        now_ts: 当前时间戳。

    Returns:
        NarrativeResult: 重组结果。
    """
    resolved = _detect_repair_arcs(memory_store, relation_store, relation_id, now_ts)
    elevated = _elevate_durable_anchors(memory_store, relation_id, now_ts)
    merged, remap = _consolidate_themes(memory_store, relation_id, now_ts)
    return NarrativeResult(
        resolved_knot_ids=resolved,
        elevated_fragment_ids=elevated,
        merged_thread_pairs=merged,
        thread_remap=remap,
    )


def _detect_repair_arcs(
    memory_store: MemoryStore,
    relation_store: RelationStore,
    relation_id: str,
    now_ts: float,
) -> tuple[str, ...]:
    """检测冲突→修复弧并标记 knot 为 resolved。

    条件：knot 未 resolved，主导通道含 BOUNDARY 或 HURT，
    且关系的 repair_events >= boundary_events。
    """
    formation = relation_store.formations.get(relation_id)
    if formation is None:
        return ()
    if formation.repair_events < formation.boundary_events:
        return ()

    resolved_ids: list[str] = []
    for knot_id in list(memory_store.relation_knots.get(relation_id, ())):
        knot = memory_store.knots.get(knot_id)
        if knot is None or knot.resolved:
            continue
        channels = set(knot.dominant_channels)
        if TraceChannel.BOUNDARY in channels or TraceChannel.HURT in channels:
            updated = Knot(
                knot_id=knot.knot_id,
                relation_id=knot.relation_id,
                fragment_ids=knot.fragment_ids,
                dominant_channels=knot.dominant_channels,
                intensity=round(clamp(knot.intensity * 0.7), 4),
                resolved=True,
                created_at=knot.created_at,
                last_rewoven_at=now_ts,
            )
            memory_store.update_knot(updated)
            resolved_ids.append(knot.knot_id)
    return tuple(resolved_ids)


def _elevate_durable_anchors(
    memory_store: MemoryStore,
    relation_id: str,
    now_ts: float,
) -> tuple[str, ...]:
    """提升跨线程高持久度片段的结构重要性。

    跨 ≥2 条线程出现且 durability > 阈值的片段获得持久度加固。
    """
    thread_ids = memory_store.relation_threads.get(relation_id, [])
    if len(thread_ids) < 2:
        return ()

    frag_thread_count: dict[str, int] = {}
    for tid in thread_ids:
        thread = memory_store.threads.get(tid)
        if thread is None:
            continue
        for fid in thread.fragment_ids:
            frag_thread_count[fid] = frag_thread_count.get(fid, 0) + 1

    elevated: list[str] = []
    for fid, count in frag_thread_count.items():
        if count < 2:
            continue
        frag = memory_store.fragments.get(fid)
        if frag is None or frag.durability < DURABILITY_ELEVATION_THRESHOLD:
            continue
        new_durability = clamp(frag.durability + CROSS_THREAD_DURABILITY_BOOST)
        if new_durability > frag.durability:
            memory_store.set_fragment(fid, replace(frag, durability=new_durability))
            elevated.append(fid)
    return tuple(elevated)


def _consolidate_themes(
    memory_store: MemoryStore,
    relation_id: str,
    now_ts: float,
) -> tuple[tuple[tuple[str, str], ...], dict[str, str]]:
    """合并共享主题的线程。

    两条线程若主导通道重叠 ≥ THEME_CHANNEL_OVERLAP
    且片段重叠 ≥ THEME_FRAGMENT_OVERLAP，则合并为一条。

    Returns:
        (merged_pairs, remap) — remap 将被移除线程 ID 映射到存活线程 ID。
    """
    thread_ids = list(memory_store.relation_threads.get(relation_id, []))
    threads = [
        memory_store.threads[tid]
        for tid in thread_ids
        if tid in memory_store.threads
    ]
    if len(threads) < 2:
        return (), {}

    merged: list[tuple[str, str]] = []
    remap: dict[str, str] = {}
    consumed: set[str] = set()

    for i in range(len(threads)):
        if threads[i].thread_id in consumed:
            continue
        for j in range(i + 1, len(threads)):
            if threads[j].thread_id in consumed:
                continue
            ch_a = set(threads[i].dominant_channels)
            ch_b = set(threads[j].dominant_channels)
            ch_union = ch_a | ch_b
            ch_overlap = len(ch_a & ch_b) / max(1, len(ch_union))
            if ch_overlap < THEME_CHANNEL_OVERLAP:
                continue

            fids_a = set(threads[i].fragment_ids)
            fids_b = set(threads[j].fragment_ids)
            fid_union = fids_a | fids_b
            fid_overlap = len(fids_a & fids_b) / max(1, len(fid_union))
            if fid_overlap < THEME_FRAGMENT_OVERLAP:
                continue

            combined_fids = tuple(sorted(fid_union))
            combined_channels = tuple(sorted(ch_a | ch_b, key=lambda c: c.value))
            from aurora.memory.thread import Thread
            updated = Thread(
                thread_id=threads[i].thread_id,
                relation_id=relation_id,
                fragment_ids=combined_fids,
                dominant_channels=combined_channels,
                tension=round(clamp(max(threads[i].tension, threads[j].tension)), 4),
                coherence=round(clamp((threads[i].coherence + threads[j].coherence) / 2), 4),
                created_at=min(threads[i].created_at, threads[j].created_at),
                last_rewoven_at=now_ts,
            )
            removed_id = threads[j].thread_id
            survivor_id = threads[i].thread_id
            # 更新被移除线程的成员片段 backlinks
            for fid in threads[j].fragment_ids:
                frag = memory_store.fragments.get(fid)
                if frag is None:
                    continue
                new_tids = tuple(
                    survivor_id if t == removed_id else t
                    for t in frag.thread_ids
                    if t != removed_id or survivor_id not in frag.thread_ids
                )
                if new_tids != frag.thread_ids:
                    memory_store.set_fragment(fid, replace(frag, thread_ids=new_tids))
            memory_store.update_thread(updated)
            memory_store.remove_thread(removed_id)
            consumed.add(removed_id)
            remap[removed_id] = survivor_id
            merged.append((survivor_id, removed_id))
            break

    return tuple(merged), remap
