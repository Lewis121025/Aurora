"""热路径表达上下文。"""

from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import RecallHit


@dataclass(frozen=True, slots=True)
class ExpressionContext:
    """单次回复生成所需的上下文。"""

    input_text: str
    relation_segment: str
    open_loop_segment: str
    recalled_hits: tuple[RecallHit, ...] = ()
