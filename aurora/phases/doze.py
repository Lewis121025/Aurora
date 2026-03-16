"""doze 相位执行模块。

实现 doze 相位的逻辑：
1. 进入 doze 状态
2. 对活跃关系执行 hover 操作（提升显著性）
3. 执行 decay 操作（时间衰减）
4. 微量增加睡眠需求
"""
from __future__ import annotations

from aurora.being.metabolic_state import MetabolicState
from aurora.memory.doze_ops import decay_for_doze, hover_for_doze
from aurora.memory.store import MemoryStore
from aurora.phases.outcomes import PhaseOutcome
from aurora.phases.transitions import phase_transition
from aurora.runtime.contracts import Phase


def run_doze(
    metabolic: MetabolicState,
    memory_store: MemoryStore,
    now_ts: float,
) -> PhaseOutcome:
    """执行 doze 相位。

    doze 是低活跃状态，用于在 awake 间隔期间维护记忆：
    - hover：提升活跃片段及其邻居的显著性
    - decay：随时间衰减片段的显著性和未解决度

    Args:
        metabolic: 代谢状态。
        memory_store: 记忆存储。
        now_ts: 当前时间戳。

    Returns:
        PhaseOutcome: doze 相位产出。
    """
    previous = metabolic.phase
    metabolic.enter_phase(Phase.DOZE, now_ts)

    # 收集目标关系（活跃关系 + 待处理关系）
    relation_ids = tuple(
        dict.fromkeys([*metabolic.active_relation_ids, *metabolic.pending_sleep_relation_ids])
    )

    # 执行 hover 和 decay 操作
    if relation_ids:
        hover_for_doze(memory_store, relation_ids=relation_ids, now_ts=now_ts)
    decay_for_doze(memory_store, now_ts, relation_ids=relation_ids)

    # 微量增加睡眠需求
    metabolic.bump_sleep_need(0.10)

    return PhaseOutcome(
        phase=Phase.DOZE,
        transition=phase_transition(previous, Phase.DOZE, "manual_doze", now_ts),
    )
