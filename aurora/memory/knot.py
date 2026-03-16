"""记忆结模块。

定义记忆结（Knot），表示未完全整合的记忆张力结构，包含：
- 关联的片段 ID 列表
- 主导通道
- 强度
- 解决状态
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class Knot:
    """记忆结。

    表示未完全整合的记忆张力结构，通常由未解决的交互或冲突形成。
    记忆结在 sleep 阶段可能被解开（整合）或保持。

    Attributes:
        knot_id: 记忆结唯一 ID。
        relation_id: 所属关系 ID。
        fragment_ids: 关联的片段 ID 列表。
        dominant_channels: 主导轨迹通道列表。
        intensity: 强度（0.0–1.0）。
        resolved: 是否已解决。
        created_at: 创建时间戳。
        last_rewoven_at: 最后重织时间戳。
    """

    knot_id: str
    relation_id: str
    fragment_ids: tuple[str, ...]
    dominant_channels: tuple[TraceChannel, ...]
    intensity: float
    resolved: bool
    created_at: float
    last_rewoven_at: float
