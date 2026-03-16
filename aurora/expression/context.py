from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aurora.runtime.contracts import TraceChannel


@dataclass(frozen=True, slots=True)
class ExpressionContext:
    input_text: str
    dominant_channels: tuple[TraceChannel, ...]
    has_knots: bool
    recalled_surfaces: tuple[str, ...] = ()
    recent_summaries: tuple[str, ...] = ()
    orientation_snapshot: dict[str, Any] | None = None
