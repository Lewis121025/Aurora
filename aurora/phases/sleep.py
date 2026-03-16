"""sleep 相位执行模块。

实现 sleep 相位的逻辑：
1. 进入 sleep 状态
2. 执行记忆重织（reweave）
3. 执行沉积清理（sediment）
4. 吸收 sleep 结果到定向系统
5. 整理代谢状态
"""
from __future__ import annotations

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.llm.provider import LLMProvider
from aurora.memory.narrative import restructure
from aurora.memory.reweave_engine import reweave
from aurora.memory.sediment import sediment
from aurora.memory.semantic import SemanticScorer
from aurora.memory.store import MemoryStore
from aurora.phases.outcomes import PhaseOutcome
from aurora.phases.transitions import phase_transition
from aurora.relation.store import RelationStore
from aurora.runtime.contracts import Phase, TraceChannel


def _apply_thread_remap(
    relation_store: RelationStore,
    orientation: Orientation,
    memory_store: MemoryStore,
    thread_remap: dict[str, str],
) -> None:
    """将被合并线程的映射同步到长期状态。

    narrative.restructure 可能移除线程 ID。这里统一修正：
    - relation formation 的 thread_ids
    - orientation 的 anchor_thread_ids
    """
    if not thread_remap:
        return

    live_threads = set(memory_store.threads)

    for relation_id, formation in relation_store.formations.items():
        remapped = {
            thread_remap.get(thread_id, thread_id)
            for thread_id in formation.thread_ids
        }
        remapped = {thread_id for thread_id in remapped if thread_id in live_threads}
        if remapped != formation.thread_ids:
            formation.thread_ids = remapped
            relation_store._dirty_formations.add(relation_id)

    remapped_anchors: list[str] = []
    for thread_id in orientation.anchor_thread_ids:
        mapped = thread_remap.get(thread_id, thread_id)
        if mapped not in live_threads or mapped in remapped_anchors:
            continue
        remapped_anchors.append(mapped)
    if tuple(remapped_anchors) != orientation.anchor_thread_ids:
        orientation.anchor_thread_ids = tuple(remapped_anchors)


def run_sleep(
    metabolic: MetabolicState,
    orientation: Orientation,
    memory_store: MemoryStore,
    relation_store: RelationStore,
    now_ts: float,
    llm: LLMProvider | None = None,
) -> PhaseOutcome:
    """执行 sleep 相位。

    sleep 是记忆整合阶段：
    1. 重织：创建/更新线程和记忆结，强化关联，软化片段
    2. 沉积：清理低显著性、长期未激活的记忆
    3. 吸收：将整合结果纳入定向系统

    Args:
        metabolic: 代谢状态。
        orientation: 本体定向。
        memory_store: 记忆存储。
        relation_store: 关系存储。
        now_ts: 当前时间戳。

    Returns:
        PhaseOutcome: sleep 相位产出。
    """
    previous = metabolic.phase
    metabolic.enter_phase(Phase.SLEEP, now_ts)

    # 执行记忆重织
    scorer = SemanticScorer(llm) if llm is not None else None
    mutation = reweave(
        store=memory_store,
        relation_formations=relation_store.formations,
        now_ts=now_ts,
        pending_relations=metabolic.pending_sleep_relation_ids or None,
        semantic_scorer=scorer,
    )

    # reweave 直接修改 formation 对象，需补充脏标记
    for rid in mutation.affected_relation_ids:
        relation_store._dirty_formations.add(rid)

    # 叙事重组：冲突→修复弧检测、持久锚点提升、主题收敛
    thread_remap: dict[str, str] = {}
    for rid in mutation.affected_relation_ids:
        result = restructure(memory_store, relation_store, rid, now_ts)
        thread_remap.update(result.thread_remap)
    _apply_thread_remap(
        relation_store=relation_store,
        orientation=orientation,
        memory_store=memory_store,
        thread_remap=thread_remap,
    )

    # 执行沉积清理
    sediment(memory_store, now_ts)

    # 将合并线程映射应用到 mutation 输出，过滤已移除的线程
    live_thread_ids = tuple(
        thread_remap.get(tid, tid) for tid in mutation.created_thread_ids
        if thread_remap.get(tid, tid) in memory_store.threads
    )

    # 收集主导通道
    dominant: set[TraceChannel] = set()
    for thread_id in live_thread_ids:
        if thread_id in memory_store.threads:
            dominant.update(memory_store.threads[thread_id].dominant_channels)
    for knot_id in mutation.created_knot_ids:
        if knot_id in memory_store.knots:
            dominant.update(memory_store.knots[knot_id].dominant_channels)

    # 吸收 sleep 结果到定向系统
    orientation.absorb_sleep(
        thread_ids=live_thread_ids,
        knot_ids=mutation.created_knot_ids,
        dominant_channels=tuple(sorted(dominant, key=lambda item: item.value)),
        now_ts=now_ts,
    )

    # 从拓扑精细推导证据
    topology_threads = tuple(
        memory_store.threads[tid] for tid in live_thread_ids
        if tid in memory_store.threads
    )
    topology_knots = tuple(
        memory_store.knots[kid] for kid in mutation.created_knot_ids
        if kid in memory_store.knots
    )
    topology_formations = tuple(
        relation_store.formations[rid] for rid in mutation.affected_relation_ids
        if rid in relation_store.formations
    )
    orientation.absorb_topology(
        threads=topology_threads,
        knots=topology_knots,
        formations=topology_formations,
        now_ts=now_ts,
    )

    # 更新代谢状态
    metabolic.set_active_knots(mutation.created_knot_ids)
    metabolic.settle_after_sleep()

    return PhaseOutcome(
        phase=Phase.SLEEP,
        transition=phase_transition(previous, Phase.SLEEP, "manual_sleep", now_ts),
        mutation=mutation,
    )
