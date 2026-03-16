"""相位产出模块。

定义相位执行的通用产出结构（PhaseOutcome），
包含相位类型、转换记录、可选的 sleep 变异结果。
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.memory.reweave import SleepMutation
from aurora.runtime.contracts import Phase, PhaseTransition


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    """相位产出。

    记录相位执行的结果，用于持久化和状态投影。

    Attributes:
        phase: 相位类型（doze/sleep）。
        transition: 相位转换记录。
        mutation: sleep 变异结果（仅 sleep 相位有值）。
    """

    phase: Phase
    """相位类型。"""

    transition: PhaseTransition
    """相位转换记录。"""

    mutation: SleepMutation | None = None
    """Sleep 变异结果（仅 sleep 相位有值）。"""
