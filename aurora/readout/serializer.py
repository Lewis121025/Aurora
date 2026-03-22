"""Workspace serialization for model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aurora.core.types import Workspace


@dataclass(slots=True)
class WorkspaceSerializer:
    """Render a workspace as structured text or JSON-friendly sections."""

    section_order: tuple[str, ...] = (
        "ongoing_context",
        "salient_episodes",
        "active_hypotheses",
        "relevant_procedures",
        "provenance_anchors",
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def sections(self, workspace: Workspace, cue: str = "") -> dict[str, list[str]]:
        active = list(workspace.active_trace_ids)
        posterior = list(workspace.posterior_groups)
        return {
            "ongoing_context": [f"trace:{trace_id} weight={weight:.4f}" for trace_id, weight in zip(active, workspace.weights, strict=False)],
            "salient_episodes": [f"trace:{trace_id}" for trace_id in active[: min(8, len(active))]],
            "active_hypotheses": [self._format_group(group) for group in posterior],
            "relevant_procedures": [f"trace:{trace_id}" for trace_id in workspace.active_procedure_ids],
            "provenance_anchors": [anchor for anchor in workspace.anchor_refs],
        }

    def render_text(self, workspace: Workspace, *, cue: str = "", prompt: str = "") -> str:
        sections = self.sections(workspace, cue=cue)
        lines = []
        if prompt.strip():
            lines.append(prompt.strip())
        if cue.strip():
            lines.append(f"cue: {cue.strip()}")
        lines.append(f"summary_vector: {workspace.summary_vector}")
        for section in self.section_order:
            lines.append(f"{section}:")
            entries = sections.get(section, [])
            if not entries:
                lines.append("- none")
                continue
            lines.extend(f"- {entry}" for entry in entries)
        return "\n".join(lines)

    def to_payload(self, workspace: Workspace, *, cue: str = "", prompt: str = "") -> dict[str, Any]:
        return {
            "cue": cue,
            "prompt": prompt,
            "summary_vector": workspace.summary_vector,
            "sections": self.sections(workspace, cue=cue),
            "metadata": {**self.metadata, **workspace.metadata},
        }

    def _format_group(self, group: dict[str, Any]) -> str:
        trace_ids = ",".join(group.get("trace_ids", []))
        weights = ",".join(f"{weight:.4f}" for weight in group.get("weights", []))
        null_weight = float(group.get("null_weight", 0.0))
        return f"group:{group.get('group_id')} traces=[{trace_ids}] weights=[{weights}] null={null_weight:.4f}"


__all__ = ["WorkspaceSerializer"]
