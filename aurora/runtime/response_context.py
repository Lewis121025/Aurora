"""
aurora/runtime/response_context.py
响应上下文构建模块：负责将检索到的原始记忆（RetrievalTrace）和身份状态（IdentityState）
转换为 LLM 能够理解的结构化上下文和 Prompt。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from aurora.integrations.llm.Prompt.response_prompt import RESPONSE_SYSTEM_PROMPT, build_response_user_prompt
from aurora.runtime.results import EvidenceRef, RetrievalTraceSummary, StructuredMemoryContext
from aurora.soul.models import RetrievalTrace


@dataclass(frozen=True)
class ResponsePrompt:
    """包含了最终发送给 LLM 的系统 Prompt、用户 Prompt 以及渲染后的记忆摘要。"""
    system_prompt: str
    user_prompt: str
    rendered_memory_brief: str


class ResponseContextBuilder:
    """
    上下文构建器：将底层复杂的记忆拓扑和心理状态“降维”成文本，以便 LLM 处理。
    """
    def __init__(self, *, memory: Any) -> None:
        self._memory = memory # 传入 AuroraSoul 实例

    def build(self, *, trace: RetrievalTrace, max_items: int = 6) -> StructuredMemoryContext:
        """
        构建结构化记忆上下文：
        1. 获取身份快照。
        2. 获取叙事总结。
        3. 处理检索命中的情节、故事或主题。
        """
        identity = self._memory.snapshot_identity()
        summary = self._memory.narrative_summary()
        evidence_refs: List[EvidenceRef] = []
        retrieval_hits: List[str] = []
        
        # 提取前 N 个检索结果
        for node_id, score, kind in trace.ranked[:max_items]:
            evidence_refs.append(EvidenceRef(id=node_id, kind=kind, score=float(score), role="retrieved"))
            retrieval_hits.append(self._hit_summary(node_id=node_id, kind=kind, score=score))
            
        return StructuredMemoryContext(
            mode=identity.current_mode,
            narrative_pressure=summary.pressure,
            intuition=self._memory.intuition_keywords(limit=3),
            identity=identity,
            narrative_summary=summary,
            retrieval_hits=retrieval_hits,
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def summarize_trace(trace: RetrievalTrace) -> RetrievalTraceSummary:
        """为检索追踪生成简要概括（用于日志或评估）。"""
        return RetrievalTraceSummary(
            query=trace.query,
            attractor_path_len=len(trace.attractor_path),
            hit_count=len(trace.ranked),
            ranked_kinds=[kind for _, _, kind in trace.ranked],
            query_type=trace.query_type.name if trace.query_type is not None else None,
            time_relation=trace.time_range.relation if trace.time_range is not None else None,
            time_start=trace.time_range.start if trace.time_range is not None else None,
            time_end=trace.time_range.end if trace.time_range is not None else None,
            time_anchor_event=trace.time_range.anchor_event if trace.time_range is not None else None,
        )

    @staticmethod
    def render_memory_brief(context: StructuredMemoryContext) -> str:
        """
        将结构化上下文渲染为人类（和 LLM）可读的字符串。
        包含：身份快照、叙事摘要、系统直觉和相关记忆片段。
        """
        identity = context.identity
        summary = context.narrative_summary
        salient_axes = "none"
        lines = [
            "[Identity Snapshot]",
            f"- mode: {context.mode}",
            f"- pressure: {context.narrative_pressure:.3f}",
        ]
        if identity is not None:
            # 排序并展示最重要的性格轴
            axis_items = sorted(identity.axis_state.items(), key=lambda item: abs(item[1]), reverse=True)
            top_axes = ", ".join(f"{name}={value:+.2f}" for name, value in axis_items[:4]) or "none"
            # 展示潜意识直觉
            intuition = ", ".join(
                f"{name}={value:+.2f}" for name, value in sorted(identity.intuition_axes.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
                if abs(value) >= 0.05
            ) or "none"
            lines.extend(
                [
                    f"- active_energy: {identity.active_energy:.3f}",
                    f"- repressed_energy: {identity.repressed_energy:.3f}",
                    f"- top_axes: {top_axes}",
                    f"- intuition_axes: {intuition}",
                    f"- repairs/dreams/mode_shifts: {identity.repair_count} / {identity.dream_count} / {identity.mode_change_count}",
                ]
            )
        if summary is not None:
            salient_axes = ", ".join(summary.salient_axes[:4]) or "none"
            
        lines.append("")
        lines.append("[Narrative Summary]")
        if summary is not None:
            lines.append(f"- {summary.text}")
            lines.append(f"- salient_axes: {salient_axes}")
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
            
        return "\n".join(lines)

    @staticmethod
    def build_prompt(*, user_message: str, rendered_memory_brief: str) -> ResponsePrompt:
        """组装最终的 Prompt 对象。"""
        return ResponsePrompt(
            system_prompt=RESPONSE_SYSTEM_PROMPT,
            user_prompt=build_response_user_prompt(
                user_message=user_message,
                rendered_memory_brief=rendered_memory_brief,
            ),
            rendered_memory_brief=rendered_memory_brief,
        )

    def _hit_summary(self, *, node_id: str, kind: str, score: float) -> str:
        """为检索命中的单个节点生成简要描述。"""
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
