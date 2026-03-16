"""记忆重织接口模块。

定义重织过程的输出结构：
- NarrativeRegion：叙事区域，表示 sleep 阶段整合的记忆簇
- SleepMutation：sleep 变异结果，记录创建/更新的结构
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class NarrativeRegion:
    """叙事区域。

    sleep 阶段由片段簇形成的叙事结构，包含：
    - 锚定片段和成员片段
    - 主导通道
    - 张力、连贯性、支持度

    Attributes:
        relation_id: 所属关系 ID。
        anchor_fragment_id: 锚定片段 ID。
        fragment_ids: 成员片段 ID 列表。
        dominant_channels: 主导轨迹通道。
        tension: 张力（0.0–1.0）。
        coherence: 连贯性（0.0–1.0）。
        support: 支持度（0.0–1.0）。
    """

    relation_id: str
    anchor_fragment_id: str
    fragment_ids: tuple[str, ...]
    dominant_channels: tuple[TraceChannel, ...]
    tension: float
    coherence: float
    support: float


@dataclass(frozen=True, slots=True)
class SleepMutation:
    """Sleep 变异结果。

    记录 sleep 阶段对记忆结构的修改：
    - 创建的线程和记忆结
    - 强化的关联边
    - 软化的片段
    - 受影响的关系
    - 检索偏置（用于后续 recall）

    Attributes:
        created_thread_ids: 新创建的线程 ID 列表。
        created_knot_ids: 新创建的记忆结 ID 列表。
        strengthened_edge_ids: 被强化的关联边 ID 列表。
        softened_fragment_ids: 被软化的片段 ID 列表。
        affected_relation_ids: 受影响的关系 ID 列表。
        recall_bias: 检索偏置（关系 ID -> 线程 ID 列表）。
    """

    created_thread_ids: tuple[str, ...]
    created_knot_ids: tuple[str, ...]
    strengthened_edge_ids: tuple[str, ...]
    softened_fragment_ids: tuple[str, ...]
    affected_relation_ids: tuple[str, ...]
    recall_bias: dict[str, tuple[str, ...]]
