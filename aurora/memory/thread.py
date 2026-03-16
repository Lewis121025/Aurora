"""记忆线程模块。

定义记忆线程（Thread），表示由多个片段组成的叙事线索，包含：
- 成员片段 ID 列表
- 主导通道
- 张力和连贯性
- 创建和重织时间戳
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class Thread:
    """记忆线程。

    由多个相关片段组成的叙事线索，在 sleep 阶段通过重织形成或更新。
    线程用于组织记忆结构，支持连续性检索。

    Attributes:
        thread_id: 线程唯一 ID。
        relation_id: 所属关系 ID。
        fragment_ids: 成员片段 ID 列表。
        dominant_channels: 主导轨迹通道。
        tension: 张力（0.0–1.0）。
        coherence: 连贯性（0.0–1.0）。
        created_at: 创建时间戳。
        last_rewoven_at: 最后重织时间戳。
    """

    thread_id: str
    relation_id: str
    fragment_ids: tuple[str, ...]
    dominant_channels: tuple[TraceChannel, ...]
    tension: float
    coherence: float
    created_at: float
    last_rewoven_at: float
