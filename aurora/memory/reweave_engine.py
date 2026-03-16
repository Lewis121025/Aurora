"""记忆重织引擎模块。

实现 sleep 阶段的记忆重织（reweave）过程：
1. 选择候选片段（基于显著性、未解决度等）
2. 构建叙事区域（以高优先级片段为种子，扩展关联片段）
3. 为每个区域创建/更新线程和记忆结
4. 强化关联边，软化片段
5. 更新关系形成记录
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING, Iterable
from uuid import uuid4

from aurora.memory.affinity import (
    cluster_dominant_channels,
    cluster_keyword_overlap,
    cluster_trace_overlap,
    fragment_affinity,
    neighbor_fragment_ids,
    region_edge_density,
    structural_pressure,
)
from aurora.memory.fragment import Fragment
from aurora.memory.knot import Knot
from aurora.memory.reweave import NarrativeRegion, SleepMutation
from aurora.memory.semantic import SemanticScorer
from aurora.memory.thread import Thread
from aurora.runtime.contracts import AssocKind, TraceChannel, clamp

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore
    from aurora.relation.formation import RelationFormation


# ========== 候选片段选择权重 ==========
CANDIDATE_SALIENCE_WEIGHT = 0.18        # 显著性权重
CANDIDATE_UNRESOLVEDNESS_WEIGHT = 0.34  # 未解决度权重
CANDIDATE_ACTIVATION_WEIGHT = 0.10      # 激活次数权重
CANDIDATE_STRUCTURAL_WEIGHT = 0.22      # 结构性压力权重
CANDIDATE_THREAD_WEIGHT = 0.06          # 线程关联权重
CANDIDATE_KNOT_WEIGHT = 0.10            # 记忆结关联权重
CANDIDATE_ACTIVATION_CAP = 5.0          # 激活次数上限

# ========== 区域种子优先级权重 ==========
SEED_SALIENCE_WEIGHT = 0.32             # 显著性权重
SEED_UNRESOLVEDNESS_WEIGHT = 0.30       # 未解决度权重
SEED_ACTIVATION_WEIGHT = 0.16           # 激活次数权重
SEED_STRUCTURAL_WEIGHT = 0.12           # 结构性压力权重
SEED_FORMATION_THREAD_BONUS = 0.08      # 形成线程奖励
SEED_FORMATION_KNOT_BONUS = 0.12        # 形成记忆结奖励

# ========== 区域亲和度阈值 ==========
REGION_AFFINITY_THRESHOLD = 0.34        # 亲和度阈值
REGION_OVERLAP_THRESHOLD = 0.75         # 重叠度阈值（超过则跳过）

# ========== 区域支持度权重 ==========
SUPPORT_EDGE_DENSITY_WEIGHT = 0.34      # 边密度权重
SUPPORT_THREAD_PRESENCE_WEIGHT = 0.24   # 线程存在权重
SUPPORT_KNOT_PRESENCE_WEIGHT = 0.24     # 记忆结存在权重
SUPPORT_FORMATION_WEIGHT = 0.18         # 形成记录权重

# ========== 区域连贯性 ==========
COHERENCE_BASE = 0.22                   # 基础连贯性
COHERENCE_EDGE_DENSITY_WEIGHT = 0.18    # 边密度权重
COHERENCE_FORMATION_THREAD_BONUS = 0.08 # 形成线程奖励

# ========== 记忆结形成阈值 ==========
KNOT_BASE_THRESHOLD = 0.56              # 基础阈值
KNOT_BOUNDARY_DISCOUNT = 0.03           # 边界事件折扣
KNOT_CHANNEL_DISCOUNT = 0.02            # 通道折扣

# ========== 关联强化权重 ==========
ASSOC_BASE_WEIGHT = 0.46                # 基础权重
ASSOC_AFFINITY_WEIGHT = 0.24            # 亲和度权重
ASSOC_COHERENCE_WEIGHT = 0.16           # 连贯性权重
ASSOC_SUPPORT_WEIGHT = 0.12             # 支持度权重

SEMANTIC_AFFINITY_WEIGHT = 0.18         # 语义亲和度权重（LLM 可用时生效）

# ========== 片段软化参数 ==========
SOFTEN_SALIENCE_BASE = 0.05             # 显著性基础增量
SOFTEN_SALIENCE_SUPPORT_FACTOR = 0.03   # 支持度因子
SOFTEN_UNRESOLVED_BASE = 0.05           # 未解决度基础减量
SOFTEN_UNRESOLVED_COHERENCE_FACTOR = 0.05  # 连贯性因子


def reweave(
    store: MemoryStore,
    relation_formations: dict[str, RelationFormation],
    now_ts: float,
    pending_relations: tuple[str, ...] | None = None,
    semantic_scorer: SemanticScorer | None = None,
) -> SleepMutation:
    """执行记忆重织过程。

    对指定关系进行 sleep 阶段的整合：
    1. 选择候选片段
    2. 构建叙事区域
    3. 创建/更新线程和记忆结
    4. 强化关联边，软化片段

    Args:
        store: 记忆存储。
        relation_formations: 关系形成记录字典。
        now_ts: 当前时间戳。
        pending_relations: 待处理关系列表（为空时处理所有关系）。

    Returns:
        SleepMutation: 重织结果。
    """
    relation_ids = pending_relations or tuple(sorted(store.relation_fragments.keys()))

    created_thread_ids: list[str] = []
    updated_thread_ids: list[str] = []
    created_knot_ids: list[str] = []
    updated_knot_ids: list[str] = []
    strengthened_edge_ids: list[str] = []
    softened_fragment_ids: set[str] = set()
    affected_relation_ids: list[str] = []
    recall_bias: dict[str, tuple[str, ...]] = {}

    for relation_id in relation_ids:
        formation = relation_formations.get(relation_id)
        candidates = _select_candidates(store, relation_id)
        if semantic_scorer is not None:
            semantic_scorer.score_pairs(list(candidates))
        regions = _build_regions(store, relation_id, candidates, formation, semantic_scorer)
        if not regions:
            continue

        affected_relation_ids.append(relation_id)
        relation_thread_ids: list[str] = []
        relation_knot_ids: list[str] = []

        # 处理前 3 个区域
        for region in regions[:3]:
            cluster = [store.fragments[fid] for fid in region.fragment_ids]
            frag_ids = tuple(f.fragment_id for f in cluster)

            # 创建/更新线程
            existing_thread = store.find_matching_thread(relation_id, frag_ids)
            if existing_thread is not None:
                thread = _update_thread(
                    existing_thread, cluster, region.dominant_channels,
                    region.tension, region.coherence, now_ts,
                )
                store.update_thread(thread)
                updated_thread_ids.append(thread.thread_id)
            else:
                thread = _build_thread(
                    relation_id, cluster, region.dominant_channels,
                    region.tension, region.coherence, now_ts,
                )
                store.add_thread(thread)
                created_thread_ids.append(thread.thread_id)
            relation_thread_ids.append(thread.thread_id)

            # 创建/更新记忆结
            knot: Knot | None = None
            if _should_form_knot(region, formation):
                existing_knot = store.find_matching_knot(relation_id, frag_ids)
                if existing_knot is not None:
                    knot = _update_knot(
                        existing_knot, cluster, region.dominant_channels,
                        region.tension, now_ts,
                    )
                    store.update_knot(knot)
                    updated_knot_ids.append(knot.knot_id)
                else:
                    knot = _build_knot(
                        relation_id, cluster, region.dominant_channels, region.tension, now_ts,
                    )
                    store.add_knot(knot)
                    created_knot_ids.append(knot.knot_id)
                relation_knot_ids.append(knot.knot_id)

            # 软化片段
            for fragment in cluster:
                softened_fragment_ids.add(fragment.fragment_id)
                store.set_fragment(
                    fragment.fragment_id,
                    store.fragments[fragment.fragment_id].touched(
                        at=now_ts,
                        delta_salience=SOFTEN_SALIENCE_BASE + SOFTEN_SALIENCE_SUPPORT_FACTOR * region.support,
                        delta_unresolved=-(SOFTEN_UNRESOLVED_BASE + SOFTEN_UNRESOLVED_COHERENCE_FACTOR * region.coherence),
                    )
                )

            # 强化关联边
            edge_kind = AssocKind.KNOT if knot is not None else AssocKind.THREAD
            evidence_token = knot.knot_id if knot is not None else thread.thread_id
            for left, right in combinations(cluster, 2):
                edge = store.strengthen_association(
                    src_fragment_id=left.fragment_id,
                    dst_fragment_id=right.fragment_id,
                    kind=edge_kind,
                    weight=(
                        ASSOC_BASE_WEIGHT
                        + ASSOC_AFFINITY_WEIGHT * fragment_affinity(store, left, right)
                        + ASSOC_COHERENCE_WEIGHT * region.coherence
                        + ASSOC_SUPPORT_WEIGHT * region.support
                    ),
                    evidence=(evidence_token,),
                    now_ts=now_ts,
                )
                strengthened_edge_ids.append(edge.edge_id)

        # 设置检索偏置
        recall_bias[relation_id] = tuple(relation_thread_ids[-4:])

        # 更新关系形成记录
        if formation is not None:
            formation.absorb_sleep(
                thread_ids=tuple(relation_thread_ids),
                knot_ids=tuple(relation_knot_ids),
                now_ts=now_ts,
            )

    # 更新 sleep 周期计数
    all_thread_ids = created_thread_ids + updated_thread_ids
    all_knot_ids = created_knot_ids + updated_knot_ids
    if all_thread_ids or all_knot_ids:
        store.sleep_cycles += 1
        store.last_sleep_at = now_ts

    return SleepMutation(
        created_thread_ids=tuple(all_thread_ids),
        created_knot_ids=tuple(all_knot_ids),
        strengthened_edge_ids=tuple(strengthened_edge_ids),
        softened_fragment_ids=tuple(sorted(softened_fragment_ids)),
        affected_relation_ids=tuple(affected_relation_ids),
        recall_bias=recall_bias,
    )


def _select_candidates(
    store: MemoryStore,
    relation_id: str,
    top_k: int = 24,
) -> tuple[Fragment, ...]:
    """选择候选片段。

    按综合评分排序，返回前 top_k 个片段。

    Args:
        store: 记忆存储。
        relation_id: 关系 ID。
        top_k: 返回数量上限。

    Returns:
        候选片段元组。
    """
    ranked = sorted(
        store.fragments_for_relation(relation_id),
        key=lambda item: (
            CANDIDATE_SALIENCE_WEIGHT * item.salience
            + CANDIDATE_UNRESOLVEDNESS_WEIGHT * item.unresolvedness
            + CANDIDATE_ACTIVATION_WEIGHT * min(math.log1p(item.activation_count) / math.log1p(CANDIDATE_ACTIVATION_CAP), 1.0)
            + CANDIDATE_STRUCTURAL_WEIGHT * structural_pressure(store, item)
            + CANDIDATE_THREAD_WEIGHT * min(math.log1p(len(item.thread_ids)), 1.0)
            + CANDIDATE_KNOT_WEIGHT * min(math.log1p(len(item.knot_ids)), 1.0)
        ),
        reverse=True,
    )
    return tuple(ranked[:top_k])


def _build_regions(
    store: MemoryStore,
    relation_id: str,
    candidates: Iterable[Fragment],
    formation: RelationFormation | None,
    semantic_scorer: SemanticScorer | None = None,
) -> tuple[NarrativeRegion, ...]:
    """构建叙事区域。

    按种子优先级排序，依次尝试构建区域，跳过重叠度过高的区域。

    Args:
        store: 记忆存储。
        relation_id: 关系 ID。
        candidates: 候选片段迭代器。
        formation: 关系形成记录。

    Returns:
        叙事区域元组，按支持度和张力排序。
    """
    ordered = sorted(
        candidates,
        key=lambda item: _seed_priority(store, item, formation),
        reverse=True,
    )
    regions: list[NarrativeRegion] = []

    # 尝试前 8 个种子
    for seed in ordered[:8]:
        fragment_ids = _expand_region(store, seed, ordered, formation, semantic_scorer)
        if len(fragment_ids) < 2:
            continue
        region = _materialize_region(store, relation_id, seed.fragment_id, fragment_ids, formation)

        # 跳过重叠度过高的区域
        if any(
            _region_overlap(region.fragment_ids, existing.fragment_ids) >= REGION_OVERLAP_THRESHOLD
            for existing in regions
        ):
            continue
        regions.append(region)

    return tuple(sorted(regions, key=lambda item: (item.support, item.tension), reverse=True))


def _seed_priority(
    store: MemoryStore,
    fragment: Fragment,
    formation: RelationFormation | None,
) -> float:
    """计算种子优先级评分。

    Args:
        store: 记忆存储。
        fragment: 候选片段。
        formation: 关系形成记录。

    Returns:
        种子优先级评分。
    """
    formation_thread_bonus = 0.0
    formation_knot_bonus = 0.0
    if formation is not None:
        formation_thread_bonus = (
            SEED_FORMATION_THREAD_BONUS if set(fragment.thread_ids) & formation.thread_ids else 0.0
        )
        formation_knot_bonus = (
            SEED_FORMATION_KNOT_BONUS if set(fragment.knot_ids) & formation.knot_ids else 0.0
        )
    return (
        SEED_SALIENCE_WEIGHT * fragment.salience
        + SEED_UNRESOLVEDNESS_WEIGHT * fragment.unresolvedness
        + SEED_ACTIVATION_WEIGHT * min(fragment.activation_count / CANDIDATE_ACTIVATION_CAP, 1.0)
        + SEED_STRUCTURAL_WEIGHT * structural_pressure(store, fragment)
        + formation_thread_bonus
        + formation_knot_bonus
    )


def _expand_region(
    store: MemoryStore,
    seed: Fragment,
    candidates: Iterable[Fragment],
    formation: RelationFormation | None,
    semantic_scorer: SemanticScorer | None = None,
) -> tuple[str, ...]:
    """扩展区域。

    从种子片段开始，添加邻居、关联片段、亲和片段。

    Args:
        store: 记忆存储。
        seed: 种子片段。
        candidates: 候选片段迭代器。
        formation: 关系形成记录。

    Returns:
        区域片段 ID 元组，按创建时间排序。
    """
    region_ids: list[str] = [seed.fragment_id]

    # 添加邻居片段
    for nid in neighbor_fragment_ids(store, seed.fragment_id):
        if nid in store.fragments and nid not in region_ids:
            region_ids.append(nid)

    # 添加关联片段（通过线程/记忆结）
    for linked_id in _linked_fragment_ids(store, seed):
        if linked_id in store.fragments and linked_id not in region_ids:
            region_ids.append(linked_id)

    # 添加亲和片段
    scored: list[tuple[float, str]] = []
    for fragment in candidates:
        if fragment.fragment_id in region_ids:
            continue
        support = _region_affinity(store, seed, fragment, formation, semantic_scorer)
        if support >= REGION_AFFINITY_THRESHOLD:
            scored.append((support, fragment.fragment_id))
    scored.sort(reverse=True)
    for _, fragment_id in scored[:3]:
        if fragment_id not in region_ids:
            region_ids.append(fragment_id)

    # 按创建时间排序
    ordered = sorted(
        (store.fragments[fid] for fid in region_ids),
        key=lambda item: item.created_at,
    )
    return tuple(f.fragment_id for f in ordered)


def _linked_fragment_ids(store: MemoryStore, fragment: Fragment) -> tuple[str, ...]:
    """获取通过线程/记忆结关联的片段 ID。

    Args:
        store: 记忆存储。
        fragment: 源片段。

    Returns:
        关联片段 ID 元组。
    """
    linked: list[str] = []
    for thread_id in fragment.thread_ids:
        thread = store.threads.get(thread_id)
        if thread is None:
            continue
        for fid in thread.fragment_ids:
            if fid != fragment.fragment_id and fid not in linked:
                linked.append(fid)
    for knot_id in fragment.knot_ids:
        knot = store.knots.get(knot_id)
        if knot is None:
            continue
        for fid in knot.fragment_ids:
            if fid != fragment.fragment_id and fid not in linked:
                linked.append(fid)
    return tuple(linked)


def _region_affinity(
    store: MemoryStore,
    seed: Fragment,
    fragment: Fragment,
    formation: RelationFormation | None,
    semantic_scorer: SemanticScorer | None = None,
) -> float:
    """计算区域亲和度。

    综合片段亲和度、结构性、线程/记忆结重叠、形成记录重叠。

    Args:
        store: 记忆存储。
        seed: 种子片段。
        fragment: 候选片段。
        formation: 关系形成记录。

    Returns:
        区域亲和度评分。
    """
    structural = (
        0.12 if fragment.fragment_id in neighbor_fragment_ids(store, seed.fragment_id) else 0.0
    )
    thread_overlap = 0.14 if set(seed.thread_ids) & set(fragment.thread_ids) else 0.0
    knot_overlap = 0.18 if set(seed.knot_ids) & set(fragment.knot_ids) else 0.0
    formation_overlap = 0.0
    if formation is not None and (
        set(fragment.thread_ids) & formation.thread_ids
        or set(fragment.knot_ids) & formation.knot_ids
    ):
        formation_overlap = 0.08
    semantic_bonus = 0.0
    if semantic_scorer is not None:
        semantic_bonus = SEMANTIC_AFFINITY_WEIGHT * semantic_scorer.get(
            seed.fragment_id, fragment.fragment_id
        )
    return (
        fragment_affinity(store, seed, fragment)
        + structural
        + thread_overlap
        + knot_overlap
        + formation_overlap
        + semantic_bonus
    )


def _materialize_region(
    store: MemoryStore,
    relation_id: str,
    anchor_fragment_id: str,
    fragment_ids: tuple[str, ...],
    formation: RelationFormation | None,
) -> NarrativeRegion:
    """实例化叙事区域。

    计算主导通道、支持度、张力、连贯性。

    Args:
        store: 记忆存储。
        relation_id: 关系 ID。
        anchor_fragment_id: 锚定片段 ID。
        fragment_ids: 成员片段 ID 列表。
        formation: 关系形成记录。

    Returns:
        NarrativeRegion: 叙事区域。
    """
    fragments = [store.fragments[fid] for fid in fragment_ids]
    dominant = cluster_dominant_channels(store, fragments)
    support = _region_support(store, fragments, formation)
    tension = _region_tension(store, fragments, formation)
    coherence = _region_coherence(store, fragments, formation)
    return NarrativeRegion(
        relation_id=relation_id,
        anchor_fragment_id=anchor_fragment_id,
        fragment_ids=fragment_ids,
        dominant_channels=dominant,
        tension=round(clamp(tension), 4),
        coherence=round(clamp(coherence), 4),
        support=round(clamp(support), 4),
    )


def _region_support(
    store: MemoryStore,
    fragments: list[Fragment],
    formation: RelationFormation | None,
) -> float:
    """计算区域支持度。

    基于边密度、线程存在、记忆结存在、形成记录命中。

    Args:
        store: 记忆存储。
        fragments: 片段列表。
        formation: 关系形成记录。

    Returns:
        支持度评分（0.0–1.0）。
    """
    edge_density = region_edge_density(store, tuple(f.fragment_id for f in fragments))
    thread_presence = sum(1 for f in fragments if f.thread_ids) / max(1, len(fragments))
    knot_presence = sum(1 for f in fragments if f.knot_ids) / max(1, len(fragments))
    formation_bonus = 0.0
    if formation is not None:
        formation_hits = sum(
            1 for f in fragments
            if set(f.thread_ids) & formation.thread_ids or set(f.knot_ids) & formation.knot_ids
        )
        formation_bonus = formation_hits / max(1, len(fragments))
    return (
        SUPPORT_EDGE_DENSITY_WEIGHT * edge_density
        + SUPPORT_THREAD_PRESENCE_WEIGHT * thread_presence
        + SUPPORT_KNOT_PRESENCE_WEIGHT * knot_presence
        + SUPPORT_FORMATION_WEIGHT * formation_bonus
    )


def _region_tension(
    store: MemoryStore,
    fragments: list[Fragment],
    formation: RelationFormation | None,
) -> float:
    """计算区域张力。

    基于未解决度平均值，加上 Hurt/Boundary 通道奖励，
    以及边界事件超过修复事件的奖励。

    Args:
        store: 记忆存储。
        fragments: 片段列表。
        formation: 关系形成记录。

    Returns:
        张力评分。
    """
    base = sum(f.unresolvedness for f in fragments) / len(fragments)
    channels = set(cluster_dominant_channels(store, fragments))
    if TraceChannel.HURT in channels:
        base += 0.08
    if TraceChannel.BOUNDARY in channels:
        base += 0.08
    if formation is not None and formation.boundary_events > formation.repair_events:
        base += 0.06
    return base


def _region_coherence(
    store: MemoryStore,
    fragments: list[Fragment],
    formation: RelationFormation | None,
) -> float:
    """计算区域连贯性。

    基于基础值、关键词重叠、轨迹重叠、边密度、形成线程命中。

    Args:
        store: 记忆存储。
        fragments: 片段列表。
        formation: 关系形成记录。

    Returns:
        连贯性评分。
    """
    coherence = (
        COHERENCE_BASE
        + cluster_keyword_overlap(fragments)
        + cluster_trace_overlap(store, fragments)
    )
    coherence += COHERENCE_EDGE_DENSITY_WEIGHT * region_edge_density(
        store, tuple(f.fragment_id for f in fragments)
    )
    if formation is not None:
        thread_hit = any(set(f.thread_ids) & formation.thread_ids for f in fragments)
        if thread_hit:
            coherence += COHERENCE_FORMATION_THREAD_BONUS
    return coherence


def _region_overlap(
    left_fragment_ids: tuple[str, ...],
    right_fragment_ids: tuple[str, ...],
) -> float:
    """计算两个区域的重叠度（Jaccard 相似度）。

    Args:
        left_fragment_ids: 左侧区域片段 ID 列表。
        right_fragment_ids: 右侧区域片段 ID 列表。

    Returns:
        重叠度（0.0–1.0）。
    """
    left = set(left_fragment_ids)
    right = set(right_fragment_ids)
    return len(left & right) / max(1, len(left | right))


def _should_form_knot(
    region: NarrativeRegion,
    formation: RelationFormation | None,
) -> bool:
    """判断是否应形成记忆结。

    当张力超过阈值时形成记忆结。阈值根据边界事件和主导通道调整。

    Args:
        region: 叙事区域。
        formation: 关系形成记录。

    Returns:
        是否形成记忆结。
    """
    threshold = KNOT_BASE_THRESHOLD
    if formation is not None and formation.boundary_events > formation.repair_events:
        threshold -= KNOT_BOUNDARY_DISCOUNT
    if (
        TraceChannel.BOUNDARY in region.dominant_channels
        or TraceChannel.HURT in region.dominant_channels
    ):
        threshold -= KNOT_CHANNEL_DISCOUNT
    return region.tension >= threshold


def _update_thread(
    existing: Thread,
    cluster: list[Fragment],
    dominant_channels: tuple[TraceChannel, ...],
    tension: float,
    coherence: float,
    now_ts: float,
) -> Thread:
    """更新线程。

    Args:
        existing: 现有线程。
        cluster: 片段簇。
        dominant_channels: 主导通道。
        tension: 张力。
        coherence: 连贯性。
        now_ts: 当前时间戳。

    Returns:
        更新后的线程。
    """
    return Thread(
        thread_id=existing.thread_id,
        relation_id=existing.relation_id,
        fragment_ids=tuple(f.fragment_id for f in cluster),
        dominant_channels=dominant_channels,
        tension=round(clamp(tension), 4),
        coherence=round(clamp(coherence), 4),
        created_at=existing.created_at,
        last_rewoven_at=now_ts,
    )


def _build_thread(
    relation_id: str,
    cluster: list[Fragment],
    dominant_channels: tuple[TraceChannel, ...],
    tension: float,
    coherence: float,
    now_ts: float,
) -> Thread:
    """构建新线程。

    Args:
        relation_id: 关系 ID。
        cluster: 片段簇。
        dominant_channels: 主导通道。
        tension: 张力。
        coherence: 连贯性。
        now_ts: 当前时间戳。

    Returns:
        新线程。
    """
    return Thread(
        thread_id=f"thread_{uuid4().hex[:12]}",
        relation_id=relation_id,
        fragment_ids=tuple(f.fragment_id for f in cluster),
        dominant_channels=dominant_channels,
        tension=round(clamp(tension), 4),
        coherence=round(clamp(coherence), 4),
        created_at=now_ts,
        last_rewoven_at=now_ts,
    )


def _update_knot(
    existing: Knot,
    cluster: list[Fragment],
    dominant_channels: tuple[TraceChannel, ...],
    intensity: float,
    now_ts: float,
) -> Knot:
    """更新记忆结。

    Args:
        existing: 现有记忆结。
        cluster: 片段簇。
        dominant_channels: 主导通道。
        intensity: 强度。
        now_ts: 当前时间戳。

    Returns:
        更新后的记忆结。
    """
    return Knot(
        knot_id=existing.knot_id,
        relation_id=existing.relation_id,
        fragment_ids=tuple(f.fragment_id for f in cluster),
        dominant_channels=dominant_channels,
        intensity=round(clamp(intensity), 4),
        resolved=existing.resolved,
        created_at=existing.created_at,
        last_rewoven_at=now_ts,
    )


def _build_knot(
    relation_id: str,
    cluster: list[Fragment],
    dominant_channels: tuple[TraceChannel, ...],
    intensity: float,
    now_ts: float,
) -> Knot:
    """构建新记忆结。

    Args:
        relation_id: 关系 ID。
        cluster: 片段簇。
        dominant_channels: 主导通道。
        intensity: 强度。
        now_ts: 当前时间戳。

    Returns:
        新记忆结。
    """
    return Knot(
        knot_id=f"knot_{uuid4().hex[:12]}",
        relation_id=relation_id,
        fragment_ids=tuple(f.fragment_id for f in cluster),
        dominant_channels=dominant_channels,
        intensity=round(clamp(intensity), 4),
        resolved=False,
        created_at=now_ts,
        last_rewoven_at=now_ts,
    )
