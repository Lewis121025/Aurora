"""记忆轨迹模块。

定义记忆轨迹（Trace），记录片段在认知过程中触发的通道及强度，包含：
- 通道类型（warmth/hurt/recognition 等）
- 强度（intensity）
- 携带度（carry）：影响轨迹衰减速度
- 时间戳
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class Trace:
    """记忆轨迹。

    记录片段在认知过程中触发的轨迹通道及强度。
    轨迹用于追踪记忆的情感/认知特征，支持检索和重织过程。

    Attributes:
        trace_id: 轨迹唯一 ID。
        relation_id: 所属关系 ID。
        fragment_id: 关联片段 ID。
        channel: 轨迹通道类型。
        intensity: 强度（0.0–1.0）。
        carry: 携带度（0.0–1.0），影响衰减速率。
        created_at: 创建时间戳。
        last_touched_at: 最后触碰时间戳。
    """

    trace_id: str
    relation_id: str
    fragment_id: str
    channel: TraceChannel
    intensity: float
    carry: float
    created_at: float
    last_touched_at: float
