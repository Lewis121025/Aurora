"""关系时刻模块。

定义关系时刻（RelationMoment），记录单次交互的关系动态：
- 用户和 Aurora 的转换 ID
- 用户触发的通道
- Aurora 的行为选择
- 边界/修复事件标记
- 交互摘要
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import AuroraMove, TraceChannel


@dataclass(frozen=True, slots=True)
class RelationMoment:
    """关系时刻。

    记录单次交互中关系层面的动态变化，用于：
    - 追踪关系演化历史
    - 计算关系形成记录的事件统计
    - 支持关系动力学评估

    Attributes:
        moment_id: 时刻唯一 ID。
        relation_id: 所属关系 ID。
        user_turn_id: 用户转换 ID。
        aurora_turn_id: Aurora 转换 ID（可选）。
        user_channels: 用户触发的轨迹通道列表。
        aurora_move: Aurora 的行为选择。
        boundary_event: 是否为边界事件。
        repair_event: 是否为修复事件。
        summary: 交互摘要。
        created_at: 创建时间戳。
    """

    moment_id: str
    relation_id: str
    user_turn_id: str
    aurora_turn_id: str | None
    user_channels: tuple[TraceChannel, ...]
    aurora_move: AuroraMove
    boundary_event: bool
    repair_event: bool
    summary: str
    created_at: float
