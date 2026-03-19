"""Hot path expression context."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpressionContext:
    """Context required for single response generation."""

    input_text: str
    memory_brief: str
    session_transcript: str
