"""记忆关联模块。

定义记忆片段之间的关联边（Association），记录：
- 源片段和目标片段
- 关联类型（RELATION/BOUNDARY/REPAIR/CONTRAST）
- 权重和证据
- 时间戳
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import AssocKind


@dataclass(frozen=True, slots=True)
class Association:
    """记忆片段关联边。

    连接两个记忆片段，记录关联的类型、权重和证据。
    关联边用于构建记忆网络结构，支持检索和重织操作。

    Attributes:
        edge_id: 关联边唯一 ID。
        src_fragment_id: 源片段 ID。
        dst_fragment_id: 目标片段 ID。
        kind: 关联类型（RELATION/BOUNDARY/REPAIR/CONTRAST）。
        weight: 关联权重（0.0–1.0）。
        evidence: 证据源 ID 列表（如关系时刻 ID）。
        created_at: 创建时间戳。
        last_touched_at: 最后触碰时间戳。
    """

    edge_id: str
    src_fragment_id: str
    dst_fragment_id: str
    kind: AssocKind
    weight: float
    evidence: tuple[str, ...]
    created_at: float
    last_touched_at: float
