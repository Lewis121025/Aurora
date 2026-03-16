"""关系投影模块。

从 RelationFormation 派生非本体投影值（trust、distance、warmth）。
投影结果仅用于表达层和 surface 展示，不写入本体图。
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.relation.formation import RelationFormation
from aurora.runtime.contracts import clamp


@dataclass(frozen=True, slots=True)
class RelationProjection:
    """关系投影值。

    Attributes:
        trust: 信任度（0.0–1.0），由共鸣/修复事件积累、边界事件抑制。
        distance: 距离感（0.0–1.0），边界事件推远、共鸣事件拉近。
        warmth: 温暖度（0.0–1.0），由共鸣密度和结构深度决定。
    """

    trust: float
    distance: float
    warmth: float


# 投影参数
_RESONANCE_TRUST_UNIT = 0.06
_REPAIR_TRUST_UNIT = 0.08
_BOUNDARY_TRUST_PENALTY = 0.04
_THREAD_TRUST_BONUS = 0.02
_KNOT_TRUST_PENALTY = 0.01

_BOUNDARY_DISTANCE_UNIT = 0.07
_RESONANCE_DISTANCE_UNIT = 0.04
_REPAIR_DISTANCE_UNIT = 0.05

_RESONANCE_WARMTH_UNIT = 0.05
_THREAD_WARMTH_BONUS = 0.03
_KNOT_WARMTH_PENALTY = 0.02
_STALENESS_HOURS_CAP = 168.0


def project_relation(formation: RelationFormation, now_ts: float) -> RelationProjection:
    """从 RelationFormation 派生投影值。

    Args:
        formation: 关系形成记录。
        now_ts: 当前时间戳。

    Returns:
        RelationProjection: 投影结果。
    """
    total_events = (
        formation.resonance_events + formation.boundary_events + formation.repair_events
    )
    if total_events == 0:
        return RelationProjection(trust=0.0, distance=1.0, warmth=0.0)

    # 信任度：共鸣和修复累积，边界事件抑制，结构加成
    trust_raw = (
        _RESONANCE_TRUST_UNIT * formation.resonance_events
        + _REPAIR_TRUST_UNIT * formation.repair_events
        - _BOUNDARY_TRUST_PENALTY * formation.boundary_events
        + _THREAD_TRUST_BONUS * len(formation.thread_ids)
        - _KNOT_TRUST_PENALTY * len(formation.knot_ids)
    )

    # 距离感：边界推远，共鸣和修复拉近
    distance_raw = (
        0.8
        + _BOUNDARY_DISTANCE_UNIT * formation.boundary_events
        - _RESONANCE_DISTANCE_UNIT * formation.resonance_events
        - _REPAIR_DISTANCE_UNIT * formation.repair_events
    )

    # 温暖度：共鸣密度和结构深度
    warmth_raw = (
        _RESONANCE_WARMTH_UNIT * formation.resonance_events
        + _THREAD_WARMTH_BONUS * len(formation.thread_ids)
        - _KNOT_WARMTH_PENALTY * len(formation.knot_ids)
    )

    # 时间衰减：长期无接触会轻微推远距离、降低温暖度
    hours_since = max(0.0, (now_ts - formation.last_contact_at) / 3600.0)
    staleness = min(1.0, hours_since / _STALENESS_HOURS_CAP)
    distance_raw += 0.1 * staleness
    warmth_raw -= 0.08 * staleness

    return RelationProjection(
        trust=clamp(trust_raw),
        distance=clamp(distance_raw),
        warmth=clamp(warmth_raw),
    )
