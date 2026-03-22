"""Expression-layer projections for structured workspaces."""

from __future__ import annotations

from aurora.core.types import Workspace
from aurora.readout import WorkspaceSerializer


def render_workspace_for_llm(workspace: Workspace, *, cue: str = "", prompt: str = "") -> str:
    serializer = WorkspaceSerializer()
    return serializer.render_text(workspace, cue=cue, prompt=prompt)


__all__ = ["render_workspace_for_llm"]
