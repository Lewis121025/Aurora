from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from aurora.integrations.llm.Prompt import (
    MEMORY_BRIEF_COMPILATION_SYSTEM_PROMPT,
    MEMORY_BRIEF_COMPILATION_USER_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
    build_response_user_prompt,
    instruction,
    render,
)
from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.schemas import MemoryBriefCompilation
from aurora.core.models.story import StoryArc
from aurora.core.models.trace import KnowledgeTimeline, RetrievalTrace
from aurora.core.retrieval.query_analysis import QueryType
from aurora.runtime.results import EvidenceRef, RetrievalTraceSummary, StructuredMemoryContext


@dataclass(frozen=True)
class ResponsePrompt:
    system_prompt: str
    user_prompt: str
    rendered_memory_brief: str


class ResponseContextBuilder:
    """Build a structured memory brief for response generation."""

    def __init__(
        self,
        *,
        memory: Any,
        doc_store: Any,
        llm: LLMProvider,
        compile_timeout_s: float = 8.0,
        compile_max_retries: int = 1,
    ):
        self._memory = memory
        self._doc_store = doc_store
        self._llm = llm
        self._compile_timeout_s = compile_timeout_s
        self._compile_max_retries = compile_max_retries

    def build(self, *, trace: RetrievalTrace, max_items: int = 6) -> StructuredMemoryContext:
        evidence_refs = self._build_evidence_refs(trace=trace, max_items=max_items)
        draft = StructuredMemoryContext(
            known_facts=self._collect_known_facts(trace=trace, limit=max_items),
            preferences=self._collect_preferences(trace=trace, limit=max_items),
            relationship_state=self._collect_relationship_state(trace=trace, limit=3),
            active_narratives=self._collect_active_narratives(trace=trace, limit=3),
            temporal_context=self._collect_temporal_context(trace=trace, limit=3),
            system_intuition=self._collect_system_intuition(trace=trace),
            cautions=self._collect_cautions(trace=trace, evidence_refs=evidence_refs),
            evidence_refs=evidence_refs,
        )
        compiled = self._compile_memory_brief(trace=trace, draft=draft, max_items=max_items)
        if compiled is None:
            return draft

        return StructuredMemoryContext(
            known_facts=self._merge_section(compiled.known_facts, draft.known_facts, max_items),
            preferences=self._merge_section(compiled.preferences, draft.preferences, max_items),
            relationship_state=self._merge_section(compiled.relationship_state, draft.relationship_state, 3),
            active_narratives=self._merge_section(compiled.active_narratives, draft.active_narratives, 3),
            temporal_context=self._merge_section(compiled.temporal_context, draft.temporal_context, 3),
            system_intuition=draft.system_intuition,
            cautions=self._merge_section(compiled.cautions, draft.cautions, 4),
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def summarize_trace(trace: RetrievalTrace) -> RetrievalTraceSummary:
        timeline_group = trace.timeline_group
        abstention = getattr(trace, "abstention", None)
        return RetrievalTraceSummary(
            query=trace.query,
            query_type=getattr(getattr(trace, "query_type", None), "name", "UNKNOWN"),
            attractor_path_len=len(trace.attractor_path),
            hit_count=len(trace.ranked),
            timeline_count=len(timeline_group.timelines) if timeline_group else 0,
            standalone_count=len(timeline_group.standalone_results) if timeline_group else 0,
            abstain=bool(getattr(abstention, "should_abstain", False)),
            abstention_reason=str(getattr(abstention, "reason", "") or ""),
            asker_id=trace.asker_id,
            activated_identity=trace.activated_identity,
        )

    @staticmethod
    def render_memory_brief(context: StructuredMemoryContext) -> str:
        sections = [
            ("Known Facts", context.known_facts),
            ("Preferences", context.preferences),
            ("Relationship State", context.relationship_state),
            ("Active Narratives", context.active_narratives),
            ("Temporal Context", context.temporal_context),
            ("System Intuition", context.system_intuition),
            ("Cautions", context.cautions),
            (
                "Evidence Refs",
                [
                    f"{ref.kind}:{ref.id} score={ref.score:.3f} role={ref.role}"
                    for ref in context.evidence_refs
                ],
            ),
        ]

        lines: List[str] = []
        for title, items in sections:
            lines.append(f"[{title}]")
            if items:
                lines.extend(f"- {item}" for item in items)
            else:
                lines.append("- none")
            lines.append("")

        return "\n".join(lines).strip()

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

    def _compile_memory_brief(
        self,
        *,
        trace: RetrievalTrace,
        draft: StructuredMemoryContext,
        max_items: int,
    ) -> Optional[MemoryBriefCompilation]:
        candidate_memory = {
            "known_facts": draft.known_facts,
            "preferences": draft.preferences,
            "relationship_state": draft.relationship_state,
            "active_narratives": draft.active_narratives,
            "temporal_context": draft.temporal_context,
            "cautions": draft.cautions,
        }
        evidence_summaries = self._build_evidence_summaries(trace=trace, max_items=max_items)
        user_prompt = render(
            MEMORY_BRIEF_COMPILATION_USER_PROMPT,
            instruction=instruction("MemoryBriefCompilation"),
            user_message=trace.query,
            query_type=getattr(getattr(trace, "query_type", None), "name", "UNKNOWN"),
            abstention_reason=str(getattr(getattr(trace, "abstention", None), "reason", "") or ""),
            candidate_memory=json.dumps(candidate_memory, ensure_ascii=False, indent=2),
            evidence_summaries=json.dumps(evidence_summaries, ensure_ascii=False, indent=2),
        )
        try:
            return self._llm.complete_json(
                system=MEMORY_BRIEF_COMPILATION_SYSTEM_PROMPT,
                user=user_prompt,
                schema=MemoryBriefCompilation,
                temperature=0.1,
                timeout_s=self._compile_timeout_s,
                max_retries=self._compile_max_retries,
                metadata={"operation": "memory_brief_compilation"},
            )
        except Exception:
            return None

    def _build_evidence_summaries(self, *, trace: RetrievalTrace, max_items: int) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        refs = self._build_evidence_refs(trace=trace, max_items=max_items)
        for ref in refs:
            summaries.append(
                {
                    "id": ref.id,
                    "kind": ref.kind,
                    "role": ref.role,
                    "score": round(ref.score, 4),
                    "summary": self._evidence_summary(ref.id, ref.kind),
                }
            )
        return summaries

    def _evidence_summary(self, node_id: str, kind: str) -> str:
        if kind == "plot":
            doc = self._doc_store.get(node_id)
            plot = self._memory.plots.get(node_id)
            parts: List[str] = []
            if plot is not None:
                if plot.knowledge_type:
                    parts.append(f"knowledge_type={plot.knowledge_type}")
                parts.append(f"status={plot.status}")
                parts.append(f"exposure={plot.exposure}")
                if plot.source == "seed":
                    parts.append(f"seed={plot.text}")
            if doc is not None:
                claims = [self._format_claim(claim) for claim in doc.body.get("claims", []) or []]
                claims = [claim for claim in claims if claim]
                if claims:
                    parts.append(f"claims={'; '.join(claims[:2])}")
                action = str(doc.body.get("action", "") or "").strip()
                outcome = str(doc.body.get("outcome", "") or "").strip()
                if action:
                    parts.append(f"action={action}")
                if outcome:
                    parts.append(f"outcome={outcome}")
            return " | ".join(parts)

        if kind == "story":
            story = self._memory.stories.get(node_id)
            if story is None:
                return ""
            return self._story_narrative_summary(story)

        if kind == "theme":
            theme = self._memory.themes.get(node_id)
            if theme is None:
                return ""
            return self._theme_fact(theme)

        return ""

    @staticmethod
    def _merge_section(compiled: Sequence[str], fallback: Sequence[str], limit: int) -> List[str]:
        merged: List[str] = []
        seen: Set[str] = set()
        for item in list(compiled) + list(fallback):
            normalized = ResponseContextBuilder._sanitize_brief_item(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
            if len(merged) >= limit:
                break
        return merged

    @staticmethod
    def _sanitize_brief_item(item: str) -> str:
        text = " ".join(str(item).split()).strip()
        if not text:
            return ""
        banned = ("USER:", "AGENT:", "User:", "Agent:", "用户：", "助手：")
        if any(token in text for token in banned):
            return ""
        return text

    def _collect_known_facts(self, *, trace: RetrievalTrace, limit: int) -> List[str]:
        facts: List[str] = []
        seen: Set[str] = set()

        for plot_id, _score in self._iter_current_plot_ids(trace):
            for entry in self._plot_facts(plot_id):
                if entry in seen or self._looks_like_preference(entry):
                    continue
                seen.add(entry)
                facts.append(entry)
                if len(facts) >= limit:
                    return facts

        for _nid, _score, kind in trace.ranked:
            if kind != "theme":
                continue
            theme = self._memory.themes.get(_nid)
            if theme is None:
                continue
            entry = self._theme_fact(theme)
            if not entry or entry in seen or self._looks_like_preference(entry):
                continue
            seen.add(entry)
            facts.append(entry)
            if len(facts) >= limit:
                break

        return facts

    def _collect_preferences(self, *, trace: RetrievalTrace, limit: int) -> List[str]:
        preferences: List[str] = []
        seen: Set[str] = set()

        narrative = self._memory.self_narrative_engine.narrative
        for relationship in narrative.relationships.values():
            for key, value in sorted(relationship.preferences.items(), key=lambda item: abs(item[1]), reverse=True):
                direction = "偏好" if value > 0 else "不喜欢"
                entry = f"{direction}{key}"
                if entry in seen:
                    continue
                seen.add(entry)
                preferences.append(entry)
                if len(preferences) >= limit:
                    return preferences

        for plot_id, _score in self._iter_current_plot_ids(trace):
            for entry in self._plot_facts(plot_id):
                if not self._looks_like_preference(entry) or entry in seen:
                    continue
                seen.add(entry)
                preferences.append(entry)
                if len(preferences) >= limit:
                    return preferences

        for _nid, _score, kind in trace.ranked:
            if kind != "theme":
                continue
            theme = self._memory.themes.get(_nid)
            if theme is None or theme.theme_type != "preference":
                continue
            entry = theme.description or theme.identity_dimension or theme.name
            if not entry or entry in seen:
                continue
            seen.add(entry)
            preferences.append(entry)
            if len(preferences) >= limit:
                break

        return preferences

    def _collect_relationship_state(self, *, trace: RetrievalTrace, limit: int) -> List[str]:
        entries: List[str] = []
        seen: Set[str] = set()
        candidate_ids: List[str] = []
        if trace.asker_id:
            candidate_ids.append(trace.asker_id)
        candidate_ids.extend(self._relationship_entity_candidates(trace))

        narrative = self._memory.self_narrative_engine.narrative
        for entity_id in candidate_ids:
            if entity_id in seen:
                continue
            seen.add(entity_id)

            story = self._memory.get_relationship_story(entity_id)
            if story is not None:
                entries.append(self._story_relationship_summary(story))
                if len(entries) >= limit:
                    return entries

            relationship = narrative.relationships.get(entity_id)
            if relationship is not None:
                entries.append(relationship.to_narrative())
                if len(entries) >= limit:
                    return entries

        return entries

    def _collect_active_narratives(self, *, trace: RetrievalTrace, limit: int) -> List[str]:
        stories: Dict[str, StoryArc] = {}

        for nid, _score, kind in trace.ranked:
            if kind == "story":
                story = self._memory.stories.get(nid)
                if story is not None:
                    stories[story.id] = story
            elif kind == "plot":
                plot = self._memory.plots.get(nid)
                if plot is not None and plot.story_id:
                    story = self._memory.stories.get(plot.story_id)
                    if story is not None:
                        stories[story.id] = story

        developing = [story for story in stories.values() if story.status == "developing"]
        fallback = [story for story in stories.values() if story.status != "developing"]
        ordered = sorted(developing, key=lambda story: story.updated_ts, reverse=True) + sorted(
            fallback,
            key=lambda story: story.updated_ts,
            reverse=True,
        )
        summaries = [self._story_narrative_summary(story) for story in ordered[:limit]]
        if summaries:
            return summaries
        if getattr(trace.query_type, "name", "") == QueryType.IDENTITY.name:
            seed_narrative = self._memory.self_narrative_engine.narrative.seed_narrative
            return [seed_narrative] if seed_narrative else []
        return []

    def _collect_temporal_context(self, *, trace: RetrievalTrace, limit: int) -> List[str]:
        if getattr(trace.query_type, "name", "") != QueryType.TEMPORAL.name:
            return []
        if trace.timeline_group is None:
            return []

        entries: List[str] = []
        for timeline in trace.timeline_group.timelines:
            if not timeline.has_evolution():
                continue
            summary = self._timeline_summary(timeline)
            if summary:
                entries.append(summary)
            if len(entries) >= limit:
                break
        return entries

    def _collect_system_intuition(self, *, trace: RetrievalTrace) -> List[str]:
        return self._memory.generate_system_intuition(trace)

    def _collect_cautions(self, *, trace: RetrievalTrace, evidence_refs: Sequence[EvidenceRef]) -> List[str]:
        cautions: List[str] = []
        abstention = getattr(trace, "abstention", None)
        if bool(getattr(abstention, "should_abstain", False)):
            cautions.append(
                f"当前检索证据不足，涉及历史事实时应保守回答：{getattr(abstention, 'reason', '')}".strip()
            )
        if not evidence_refs:
            cautions.append("没有检索到足够相关的记忆证据。")
        elif evidence_refs and evidence_refs[0].score < 0.45:
            cautions.append("当前最相关证据分数偏低，记忆判断应保守。")
        if any(ref.role == "historical_fact" for ref in evidence_refs):
            cautions.append("本轮涉及历史演变信息，要区分过去状态与当前状态。")
        return cautions

    def _build_evidence_refs(self, *, trace: RetrievalTrace, max_items: int) -> List[EvidenceRef]:
        refs: List[EvidenceRef] = []
        seen: Set[Tuple[str, str]] = set()

        historical_ids: Set[str] = set()
        current_ids: Set[str] = set()
        if trace.timeline_group is not None:
            for timeline in trace.timeline_group.timelines:
                current_ids.update([timeline.current_id] if timeline.current_id else [])
                historical_ids.update(timeline.get_historical_ids())

        for nid, score, kind in trace.ranked:
            key = (nid, kind)
            if key in seen:
                continue
            seen.add(key)
            role = "support"
            if kind == "plot":
                if nid in historical_ids:
                    role = "historical_fact"
                elif nid in current_ids or kind == "plot":
                    role = "current_fact"
            elif kind == "story":
                role = "narrative"
            elif kind == "theme":
                role = "theme"

            refs.append(EvidenceRef(id=nid, kind=kind, score=float(score), role=role))
            if len(refs) >= max_items:
                break

        return refs

    def _iter_current_plot_ids(self, trace: RetrievalTrace) -> Iterable[Tuple[str, float]]:
        seen: Set[str] = set()
        if trace.timeline_group is not None:
            for timeline in trace.timeline_group.timelines:
                if timeline.current_id and timeline.current_id not in seen:
                    plot = self._memory.plots.get(timeline.current_id)
                    if plot is None or not self._memory.allow_plot_for_query(plot, trace.query_type):
                        continue
                    seen.add(timeline.current_id)
                    yield timeline.current_id, timeline.match_score
            for nid, score, kind in trace.timeline_group.standalone_results:
                if kind == "plot" and nid not in seen:
                    plot = self._memory.plots.get(nid)
                    if plot is None or not self._memory.allow_plot_for_query(plot, trace.query_type):
                        continue
                    seen.add(nid)
                    yield nid, score
            return

        for nid, score, kind in trace.ranked:
            if kind == "plot" and nid not in seen:
                seen.add(nid)
                yield nid, score

    def _relationship_entity_candidates(self, trace: RetrievalTrace) -> List[str]:
        entities: List[str] = []
        for plot_id, _score in self._iter_current_plot_ids(trace):
            plot = self._memory.plots.get(plot_id)
            if plot is None or plot.relational is None:
                continue
            entity = plot.relational.with_whom
            if entity and entity not in entities:
                entities.append(entity)
        return entities

    def _plot_facts(self, plot_id: str) -> List[str]:
        doc = self._doc_store.get(plot_id)
        plot = self._memory.plots.get(plot_id)
        if plot is None:
            return []

        facts: List[str] = []
        if doc is not None:
            for claim in doc.body.get("claims", []) or []:
                formatted = self._format_claim(claim)
                if formatted:
                    facts.append(formatted)
            if not facts:
                action = str(doc.body.get("action", "") or "").strip()
                outcome = str(doc.body.get("outcome", "") or "").strip()
                if action and outcome:
                    facts.append(f"{action} -> {outcome}")
                elif outcome:
                    facts.append(outcome)
                elif action:
                    facts.append(action)

        if not facts and plot.fact_keys:
            facts.extend(str(fact) for fact in plot.fact_keys if str(fact).strip())

        if not facts and plot.source == "seed":
            facts.append(plot.text)

        if not facts and plot.relational is not None:
            facts.append(plot.relational.what_this_says_about_us)

        return [fact.strip() for fact in facts if fact and fact.strip()]

    @staticmethod
    def _format_claim(claim: Dict[str, Any]) -> str:
        subject = str(claim.get("subject", "") or "").strip()
        predicate = str(claim.get("predicate", "") or "").strip()
        obj = str(claim.get("object", "") or "").strip()
        if not any([subject, predicate, obj]):
            return ""

        qualifiers = claim.get("qualifiers") or {}
        qualifier_text = ""
        if qualifiers:
            rendered = ", ".join(f"{key}={value}" for key, value in qualifiers.items())
            qualifier_text = f" ({rendered})"

        polarity = claim.get("polarity", "positive")
        if polarity == "negative" and predicate:
            predicate = f"not {predicate}"

        return " ".join(part for part in [subject, predicate, obj] if part).strip() + qualifier_text

    @staticmethod
    def _theme_fact(theme: Any) -> str:
        if theme.theme_type == "identity":
            return theme.to_identity_narrative()
        return str(theme.description or theme.name or "").strip()

    @staticmethod
    def _looks_like_preference(text: str) -> bool:
        lowered = text.lower()
        keywords = (
            "prefer",
            "preference",
            "like ",
            "likes ",
            "dislike",
            "喜欢",
            "偏好",
            "不喜欢",
        )
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _story_relationship_summary(story: StoryArc) -> str:
        parts: List[str] = []
        if story.relationship_with:
            parts.append(f"关系对象={story.relationship_with}")
        parts.append(f"状态={story.status}")
        if story.my_identity_in_this_relationship:
            parts.append(f"我的角色={story.my_identity_in_this_relationship}")
        parts.append(f"关系健康度={story.relationship_health:.2f}")
        if story.lessons_from_relationship:
            parts.append(f"经验={story.lessons_from_relationship[0]}")
        return " | ".join(parts)

    @staticmethod
    def _story_narrative_summary(story: StoryArc) -> str:
        parts: List[str] = [f"story={story.id}", f"status={story.status}"]
        if story.relationship_with:
            parts.append(f"relationship={story.relationship_with}")
        if story.my_identity_in_this_relationship:
            parts.append(f"identity={story.my_identity_in_this_relationship}")
        if story.central_conflict:
            parts.append(f"conflict={story.central_conflict}")
        if story.lessons_from_relationship:
            parts.append(f"lesson={story.lessons_from_relationship[0]}")
        if story.resolution:
            parts.append(f"resolution={story.resolution}")
        return " | ".join(parts)

    def _timeline_summary(self, timeline: KnowledgeTimeline) -> str:
        current = self._timeline_state_label(timeline.current_id) if timeline.current_id else ""
        historical = [self._timeline_state_label(plot_id) for plot_id in timeline.get_historical_ids()]
        historical = [entry for entry in historical if entry]
        if not current and not historical:
            return ""
        if current and historical:
            return f"曾经: {'; '.join(historical[:2])} -> 现在: {current}"
        if current:
            return f"当前状态: {current}"
        return f"历史状态: {'; '.join(historical[:2])}"

    def _timeline_state_label(self, plot_id: Optional[str]) -> str:
        if not plot_id:
            return ""
        facts = self._plot_facts(plot_id)
        if facts:
            return facts[0]
        plot = self._memory.plots.get(plot_id)
        if plot is None:
            return ""
        if plot.knowledge_type:
            return f"{plot.knowledge_type} memory"
        return plot.id
