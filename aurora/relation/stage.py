"""关系阶段推断模块。

从 RelationFormation 历史推断当前关系所处的长期阶段，
供认知上下文使用。
"""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from aurora.relation.formation import RelationFormation

RelationStage = Literal[
    "initial",       # 初识：极少交互
    "developing",    # 发展中：有一定交互但尚未形成结构
    "established",   # 已建立：有线程/结构积累
    "strained",      # 紧张：边界事件多于修复
    "repairing",     # 修复中：正在修复且有修复记录
]


def infer_stage(formation: RelationFormation) -> RelationStage:
    """从关系形成记录推断当前阶段。

    Args:
        formation: 关系形成记录。

    Returns:
        当前关系阶段。
    """
    total_events = (
        formation.boundary_events
        + formation.repair_events
        + formation.resonance_events
    )

    if total_events <= 1 and not formation.thread_ids:
        return "initial"

    if formation.boundary_events > formation.repair_events:
        if formation.repair_events > 0:
            return "repairing"
        return "strained"

    if formation.repair_events > 0 and formation.boundary_events > 0:
        return "repairing"

    if formation.thread_ids or total_events >= 4:
        return "established"

    return "developing"
