"""Hot path expression context."""

from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import RecallHit


@dataclass(frozen=True, slots=True)
class ExpressionContext:
    """Context required for single response generation."""

    input_text: str
    relation_segment: str
    open_loop_segment: str
    recalled_hits: tuple[RecallHit, ...] = ()
