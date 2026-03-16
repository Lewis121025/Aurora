"""关系决策模块。

基于 RelationFormation 历史为 Aurora 的行为选择提供偏置信号。
决策结果不替代 LLM 认知，仅作为认知上下文的补充输入。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aurora.relation.projectors import project_relation
from aurora.runtime.contracts import clamp

if TYPE_CHECKING:
    from aurora.relation.formation import RelationFormation


@dataclass(frozen=True, slots=True)
class RelationBias:
    """关系偏置信号。

    Attributes:
        approach_weight: 接近倾向（0.0–1.0）。
        caution_weight: 谨慎倾向（0.0–1.0）。
        repair_urgency: 修复紧迫度（0.0–1.0）。
        hint: 自然语言偏置提示，注入 LLM 认知上下文。
    """

    approach_weight: float
    caution_weight: float
    repair_urgency: float
    hint: str


_UNREPAIRED_THRESHOLD = 2
_HIGH_TRUST_THRESHOLD = 0.5
_HIGH_CAUTION_THRESHOLD = 0.55


def compute_bias(formation: RelationFormation, now_ts: float) -> RelationBias:
    """计算关系偏置信号。

    Args:
        formation: 关系形成记录。
        now_ts: 当前时间戳。

    Returns:
        RelationBias: 偏置信号。
    """
    projection = project_relation(formation, now_ts)
    unrepaired = max(0, formation.boundary_events - formation.repair_events)

    approach_weight = clamp(projection.trust * 0.6 + projection.warmth * 0.4)
    caution_weight = clamp(projection.distance * 0.5 + 0.1 * min(unrepaired, 5) / 5.0)
    repair_urgency = clamp(0.15 * unrepaired) if unrepaired >= _UNREPAIRED_THRESHOLD else 0.0

    hint = _build_hint(approach_weight, caution_weight, repair_urgency)

    return RelationBias(
        approach_weight=round(approach_weight, 4),
        caution_weight=round(caution_weight, 4),
        repair_urgency=round(repair_urgency, 4),
        hint=hint,
    )


def _build_hint(approach: float, caution: float, repair: float) -> str:
    """生成偏置提示文本。"""
    if repair > 0.25:
        return "Unrepaired boundary tension is present; consider careful repair if the moment allows."
    if caution > _HIGH_CAUTION_THRESHOLD:
        return "Relational caution is elevated; hold space rather than pressing closer."
    if approach > _HIGH_TRUST_THRESHOLD:
        return "Relational warmth supports openness; genuine approach is safe here."
    return ""
