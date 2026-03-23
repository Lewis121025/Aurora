"""Generation-layer context for a single response turn."""

from __future__ import annotations

from dataclasses import dataclass

from aurora.core.types import Workspace


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """Context required for response generation."""

    input_text: str
    workspace: Workspace
    rendered_workspace: str
