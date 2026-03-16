"""运行时状态模块。

定义 Aurora 运行时状态（RuntimeState），包含：
- 本体定向（Orientation）
- 代谢状态（MetabolicState）
- 相位转换历史
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aurora.being.metabolic_state import MetabolicState
from aurora.being.orientation import Orientation
from aurora.runtime.contracts import PhaseTransition


TRANSITION_CAP = 256  # 相位转换历史记录上限


@dataclass(slots=True)
class RuntimeState:
    """Aurora 运行时状态。

    封装 Aurora 的核心状态组件，管理相位转换历史。

    Attributes:
        orientation: 本体定向状态。
        metabolic: 代谢状态。
        transitions: 相位转换历史记录列表。
    """

    orientation: Orientation
    """本体定向状态。"""

    metabolic: MetabolicState
    """代谢状态。"""

    transitions: list[PhaseTransition] = field(default_factory=list)
    """相位转换历史记录列表。"""

    def append_transition(self, transition: PhaseTransition) -> None:
        """添加相位转换记录。

        超出容量上限时自动移除最旧的记录。

        Args:
            transition: 相位转换对象。
        """
        self.transitions.append(transition)
        if len(self.transitions) > TRANSITION_CAP:
            del self.transitions[: len(self.transitions) - TRANSITION_CAP]
