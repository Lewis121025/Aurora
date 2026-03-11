"""Response context builders for Aurora V5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

from aurora.integrations.llm.Prompt.response_prompt import (
    RESPONSE_SYSTEM_PROMPT,
    build_response_user_prompt,
)
from aurora.runtime.results import (
    EvidenceRef,
    QueryHit,
    RetrievalTraceSummary,
    StructuredMemoryContext,
)


@dataclass(frozen=True)
class ResponsePrompt:
    system_prompt: str
    user_prompt: str
    rendered_memory_brief: str


class ResponseContextBuilder:
    def __init__(self, *, memory: Any) -> None:
        self._memory = memory

    def build(self, *, hits: Iterable[QueryHit], max_items: int = 6) -> StructuredMemoryContext:
        identity = self._memory.snapshot_identity()
        summary = self._memory.narrative_summary()
        evidence_refs: List[EvidenceRef] = []
        retrieval_hits: List[str] = []
        overlay_hits: List[str] = []
        for hit in list(hits)[:max_items]:
            evidence_refs.append(
                EvidenceRef(id=hit.id, kind=hit.kind, score=float(hit.score), role="retrieved")
            )
            rendered = self._hit_summary(hit)
            if hit.kind == "event":
                overlay_hits.append(rendered)
            else:
                retrieval_hits.append(rendered)
        return StructuredMemoryContext(
            mode=identity.current_mode,
            narrative_pressure=summary.pressure,
            intuition=self._memory.intuition_keywords(limit=3),
            identity=identity,
            narrative_summary=summary,
            retrieval_hits=retrieval_hits,
            overlay_hits=overlay_hits,
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def render_memory_brief(context: StructuredMemoryContext) -> str:
        identity = context.identity
        summary = context.narrative_summary
        lines = [
            "[Identity Snapshot]",
            f"- mode: {context.mode}",
            f"- pressure: {context.narrative_pressure:.3f}",
        ]
        if identity is not None:
            axis_items = sorted(identity.axis_state.items(), key=lambda item: abs(item[1]), reverse=True)
            top_axes = ", ".join(f"{name}={value:+.2f}" for name, value in axis_items[:4]) or "none"
            lines.extend(
                [
                    f"- active_energy: {identity.active_energy:.3f}",
                    f"- repressed_energy: {identity.repressed_energy:.3f}",
                    f"- top_axes: {top_axes}",
                    f"- repairs/dreams/mode_shifts: {identity.repair_count} / {identity.dream_count} / {identity.mode_change_count}",
                ]
            )
        lines.append("")
        lines.append("[Narrative Summary]")
        if summary is not None:
            lines.append(f"- {summary.text}")
            lines.append(f"- salient_axes: {', '.join(summary.salient_axes[:4]) or 'none'}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("[System Intuition]")
        if context.intuition:
            lines.extend(f"- {item}" for item in context.intuition)
        else:
            lines.append("- none")
        lines.append("")
        lines.append("[Relevant Memory]")
        if context.retrieval_hits:
            lines.extend(f"- {item}" for item in context.retrieval_hits)
        else:
            lines.append("- none")
        lines.append("")
        lines.append("[Recent Accepted Events]")
        if context.overlay_hits:
            lines.extend(f"- {item}" for item in context.overlay_hits)
        else:
            lines.append("- none")
        return "\n".join(lines)

    @staticmethod
    def build_prompt(*, user_message: str, rendered_memory_brief: str) -> ResponsePrompt:
        return ResponsePrompt(
            system_prompt=RESPONSE_SYSTEM_PROMPT,
            user_prompt=build_response_user_prompt(
                user_message=user_message,
                rendered_memory_brief=rendered_memory_brief,
            ),
            rendered_memory_brief=rendered_memory_brief,
        )

    @staticmethod
    def trace_summary(
        *,
        query: str,
        attractor_path_len: int,
        graph_kinds: List[str],
        overlay_hit_count: int,
        query_type: str | None = None,
        time_relation: str | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
        time_anchor_event: str | None = None,
    ) -> RetrievalTraceSummary:
        return RetrievalTraceSummary(
            query=query,
            attractor_path_len=attractor_path_len,
            hit_count=len(graph_kinds) + overlay_hit_count,
            overlay_hit_count=overlay_hit_count,
            ranked_kinds=graph_kinds,
            query_type=query_type,
            time_relation=time_relation,
            time_start=time_start,
            time_end=time_end,
            time_anchor_event=time_anchor_event,
        )

    def _hit_summary(self, hit: QueryHit) -> str:
        if hit.kind == "event":
            return f"event score={hit.score:.3f} text={hit.snippet[:140]}"
        if hit.kind == "plot":
            plot = self._memory.plots.get(hit.id)
            if plot is None:
                return f"plot:{hit.id} score={hit.score:.3f}"
            return f"plot score={hit.score:.3f} source={plot.source} text={plot.text[:140]}"
        if hit.kind == "summary":
            summary = self._memory.summaries.get(hit.id)
            if summary is None:
                return f"summary:{hit.id} score={hit.score:.3f}"
            return f"summary score={hit.score:.3f} text={summary.text[:140]}"
        if hit.kind == "story":
            story = self._memory.stories.get(hit.id)
            if story is None:
                return f"story:{hit.id} score={hit.score:.3f}"
            return f"story score={hit.score:.3f} plots={len(story.plot_ids)} status={story.status}"
        theme = self._memory.themes.get(hit.id)
        if theme is None:
            return f"theme:{hit.id} score={hit.score:.3f}"
        label = theme.name or theme.description or hit.id
        return f"theme score={hit.score:.3f} {label[:140]}"
