"""相位转换辅助模块。

提供相位转换记录的创建函数。
"""
from __future__ import annotations

from uuid import uuid4

from aurora.runtime.contracts import Phase, PhaseTransition


def phase_transition(
    from_phase: Phase,
    to_phase: Phase,
    reason: str,
    created_at: float,
) -> PhaseTransition:
    """创建相位转换记录。

    Args:
        from_phase: 源相位。
        to_phase: 目标相位。
        reason: 转换原因（如 "incoming_turn"、"manual_doze"）。
        created_at: 转换时间戳。

    Returns:
        PhaseTransition: 相位转换记录。
    """
    return PhaseTransition(
        transition_id=f"pt_{uuid4().hex[:10]}",
        from_phase=from_phase,
        to_phase=to_phase,
        reason=reason,
        created_at=created_at,
    )
