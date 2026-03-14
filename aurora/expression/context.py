from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class ExpressionContext:
    input_text: str
    relation_snapshot: dict[str, float | tuple[str, ...] | int]
    dominant_channels: tuple[TraceChannel, ...]
    has_knots: bool
