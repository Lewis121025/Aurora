from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from aurora.core.soul_memory import RetrievalTrace
from aurora.integrations.llm.Prompt import RESPONSE_SYSTEM_PROMPT, build_response_user_prompt
from aurora.runtime.results import EvidenceRef, RetrievalTraceSummary, StructuredMemoryContext


@dataclass(frozen=True)
class ResponsePrompt:
    system_prompt: str
    user_prompt: str
    rendered_memory_brief: str


class ResponseContextBuilder:
    def __init__(self, *, memory: Any) -> None:
        self._memory = memory

    def build(self, *, trace: RetrievalTrace, max_items: int = 6) -> StructuredMemoryContext:
        identity = self._memory.snapshot_identity()
        summary = self._memory.narrative_summary()
        evidence_refs: List[EvidenceRef] = []
        retrieval_hits: List[str] = []
        for node_id, score, kind in trace.ranked[:max_items]:
            evidence_refs.append(EvidenceRef(id=node_id, kind=kind, score=float(score), role="retrieved"))
            retrieval_hits.append(self._hit_summary(node_id=node_id, kind=kind, score=score))
        return StructuredMemoryContext(
            phase=identity.phase,
            narrative_pressure=summary.pressure,
            intuition=self._memory.intuition_keywords(limit=2),
            identity=identity,
            narrative_summary=summary,
            retrieval_hits=retrieval_hits,
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def summarize_trace(trace: RetrievalTrace) -> RetrievalTraceSummary:
        return RetrievalTraceSummary(
            query=trace.query,
            attractor_path_len=len(trace.attractor_path),
            hit_count=len(trace.ranked),
            ranked_kinds=[kind for _, _, kind in trace.ranked],
        )

    @staticmethod
    def render_memory_brief(context: StructuredMemoryContext) -> str:
        identity = context.identity
        summary = context.narrative_summary
        lines = [
            "[Identity Snapshot]",
            f"- phase: {context.phase}",
            f"- pressure: {context.narrative_pressure:.3f}",
        ]
        if identity is not None:
            lines.extend(
                [
                    f"- active_energy: {identity.active_energy:.3f}",
                    f"- repressed_energy: {identity.repressed_energy:.3f}",
                    f"- trust/autonomy/defensiveness/coherence: "
                    f"{identity.traits.get('trust', 0.0):.2f} / "
                    f"{identity.traits.get('autonomy', 0.0):.2f} / "
                    f"{identity.traits.get('defensiveness', 0.0):.2f} / "
                    f"{identity.traits.get('coherence', 0.0):.2f}",
                ]
            )
        lines.append("")
        lines.append("[Narrative Summary]")
        lines.append(f"- {(summary.text if summary is not None else 'none')}")
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

    def _hit_summary(self, *, node_id: str, kind: str, score: float) -> str:
        if kind == "plot":
            plot = self._memory.plots.get(node_id)
            if plot is None:
                return f"plot:{node_id} score={score:.3f}"
            return f"plot score={score:.3f} source={plot.source} text={plot.text[:140]}"
        if kind == "story":
            story = self._memory.stories.get(node_id)
            if story is None:
                return f"story:{node_id} score={score:.3f}"
            return f"story score={score:.3f} plots={len(story.plot_ids)} status={story.status}"
        theme = self._memory.themes.get(node_id)
        if theme is None:
            return f"theme:{node_id} score={score:.3f}"
        label = theme.name or theme.description or node_id
        return f"theme score={score:.3f} {label[:140]}"
