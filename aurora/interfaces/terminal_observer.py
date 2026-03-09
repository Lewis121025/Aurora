from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TextIO

from aurora.core.models.trace import RetrievalTrace
from aurora.runtime.results import ChatTurnResult, StructuredMemoryContext
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings
from aurora.utils.logging import setup_logging

OBSERVE_MODES = ("off", "brief", "full")
COLOR_PALETTE = {
    "accent": "38;5;81",
    "assistant": "38;5;114",
    "user": "38;5;117",
    "prompt": "38;5;221",
    "success": "38;5;78",
    "warning": "38;5;214",
    "danger": "38;5;203",
    "muted": "38;5;245",
    "border": "38;5;240",
    "title": "38;5;255",
}
BADGE_PALETTE = {
    "accent": ("48;5;24", "38;5;255"),
    "assistant": ("48;5;28", "38;5;255"),
    "user": ("48;5;25", "38;5;255"),
    "prompt": ("48;5;94", "38;5;255"),
    "success": ("48;5;22", "38;5;255"),
    "warning": ("48;5;130", "38;5;255"),
    "muted": ("48;5;238", "38;5;255"),
}
PANEL_TONES = {
    "accent": "accent",
    "assistant": "assistant",
    "prompt": "prompt",
    "runtime": "success",
    "retrieval": "accent",
    "artifact": "warning",
    "memory": "muted",
    "neutral": "border",
}

PIPELINE_TEXT = """AURORA 终端观测脚本的每一轮会执行这条主链：
1. 用当前用户输入做检索，观察命中的 plot/story/theme 和时间线结构。
2. 把检索证据整理成结构化 Memory Brief，只把这份摘要给 LLM。
3. 生成回复后，将 user_message + agent_message 作为一次 interaction 摄入。
4. 将事件写入 SQLite event log，并把 plot / ingest_result 写入 doc store。
5. 更新内存图、向量索引、故事和主题统计。
6. 你可以随时用 /context /query /events /inspect /coherence /narrative 看内部状态。
"""


@dataclass(frozen=True)
class Command:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class QueryObservation:
    trace: RetrievalTrace
    retrieval_ms: float


def parse_command(line: str) -> Optional[Command]:
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None

    body = stripped[1:].strip()
    if not body:
        return Command(name="help", args=())

    parts = shlex.split(body)
    if not parts:
        return Command(name="help", args=())
    return Command(name=parts[0].lower(), args=tuple(parts[1:]))


class TerminalObserver:
    def __init__(
        self,
        runtime: AuroraRuntime,
        *,
        session_id: str,
        max_hits: int = 6,
        observe_mode: str = "full",
        output: TextIO = sys.stdout,
    ):
        if observe_mode not in OBSERVE_MODES:
            raise ValueError(f"Invalid observe mode: {observe_mode}")

        self.runtime = runtime
        self.session_id = session_id
        self.max_hits = max_hits
        self.observe_mode = observe_mode
        self.output = output
        self.turn_count = 0
        self.last_observation: Optional[ChatTurnResult] = None
        self._is_tty = bool(getattr(output, "isatty", lambda: False)())
        self._use_color = self._is_tty and os.environ.get("NO_COLOR") is None
        self._status_visible = False

    def run(self) -> None:
        self._print_welcome()
        while True:
            try:
                line = input(self._input_prompt())
            except EOFError:
                self._write("\nbye")
                return
            except KeyboardInterrupt:
                self._write("\nbye")
                return

            if not self.handle_line(line):
                return

    def handle_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True

        command = parse_command(stripped)
        if command is not None:
            return self._handle_command(command)

        result = self.run_chat_turn(stripped)
        self._render_chat_result(result)
        return True

    def run_chat_turn(self, user_message: str) -> ChatTurnResult:
        self.turn_count += 1
        self._show_live_status("Building memory brief", detail=user_message)
        result = self.runtime.respond(
            session_id=self.session_id,
            user_message=user_message,
            k=self.max_hits,
        )
        self._clear_live_status()
        return result

    def _observe_query(self, text: str) -> QueryObservation:
        started = time.perf_counter()
        trace = self.runtime.mem.query_with_timeline(text, k=self.max_hits, asker_id="user")
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        return QueryObservation(trace=trace, retrieval_ms=retrieval_ms)

    def _handle_command(self, command: Command) -> bool:
        name = command.name
        args = list(command.args)

        if name in {"exit", "quit"}:
            self._write("bye")
            return False
        if name == "help":
            self._render_help()
            return True
        if name == "pipeline":
            self._render_panel("Pipeline", PIPELINE_TEXT.rstrip().splitlines(), tone="accent")
            return True
        if name == "observe":
            mode = args[0].lower() if args else "full"
            if mode not in OBSERVE_MODES:
                self._write(f"observe mode 必须是 {', '.join(OBSERVE_MODES)}")
                return True
            self.observe_mode = mode
            self._render_command_bar(note=f"observe mode -> {mode}")
            return True
        if name == "clear":
            self._clear_screen()
            self._print_welcome()
            return True
        if name == "stats":
            self._render_stats()
            return True
        if name == "prompt":
            self._render_last_prompt()
            return True
        if name == "context":
            self._render_last_context()
            return True
        if name == "query":
            if not args:
                self._write("用法: /query <text>")
                return True
            query = self._observe_query(" ".join(args))
            self._render_query_observation(query)
            return True
        if name == "events":
            limit = self._parse_limit(args, default=5)
            self._render_events(limit)
            return True
        if name == "plots":
            limit = self._parse_limit(args, default=5)
            self._render_plots(limit)
            return True
        if name == "stories":
            limit = self._parse_limit(args, default=5)
            self._render_stories(limit)
            return True
        if name == "themes":
            limit = self._parse_limit(args, default=5)
            self._render_themes(limit)
            return True
        if name == "inspect":
            if not args:
                self._write("用法: /inspect <id>")
                return True
            self._render_inspect(args[0])
            return True
        if name == "coherence":
            self._render_coherence()
            return True
        if name == "narrative":
            self._render_narrative()
            return True
        if name == "evolve":
            self._render_evolve()
            return True

        self._write(f"未知命令: /{name}，输入 /help 查看可用命令。")
        return True

    def _render_chat_result(self, result: ChatTurnResult) -> None:
        self.last_observation = result
        self._render_panel("Assistant", [result.reply], tone="assistant")
        if self.observe_mode == "off":
            return

        self._render_panel("Memory Brief", result.rendered_memory_brief.splitlines(), tone="accent")

        ingest = result.ingest_result
        state_lines = [
            (
                f"retrieve={result.timings.retrieval_ms:.1f}ms | generate={result.timings.generation_ms:.1f}ms | "
                f"ingest={result.timings.ingest_ms:.1f}ms | total={result.timings.total_ms:.1f}ms"
            ),
            (
                f"event={result.event_id} | plot={ingest.plot_id} | story={ingest.story_id or '-'} | "
                f"layer={ingest.memory_layer}"
            ),
            (
                f"tension={ingest.tension:.3f} | surprise={ingest.surprise:.3f} | "
                f"pred_error={ingest.pred_error:.3f} | redundancy={ingest.redundancy:.3f}"
            ),
        ]
        if result.llm_error:
            state_lines.append(f"llm_fallback={result.llm_error}")
        self._render_panel("Runtime State", state_lines, tone="runtime")

        self._render_trace_summary(result)

        if self.observe_mode == "full":
            self._render_prompt(result)
            self._render_artifacts_and_evidence(result)
            self._render_memory_totals()
        self._render_command_bar()

    def _render_trace_summary(self, result: ChatTurnResult) -> None:
        summary = result.retrieval_trace_summary
        lines = [
            (
                f"query_type={summary.query_type} | hits={summary.hit_count} | "
                f"attractor_path={summary.attractor_path_len} | timelines={summary.timeline_count} | "
                f"standalone={summary.standalone_count} | abstain={summary.abstain}"
            )
        ]
        if summary.activated_identity:
            lines.append(f"activated_identity={summary.activated_identity}")
        if summary.abstention_reason:
            lines.append(f"abstention_reason={summary.abstention_reason}")

        if result.memory_context.evidence_refs:
            for index, ref in enumerate(result.memory_context.evidence_refs[: self.max_hits], start=1):
                snippet = self._resolve_snippet(node_id=ref.id, kind=ref.kind)
                lines.append(
                    f"{index}. [{ref.kind}] {ref.id} role={ref.role} score={ref.score:.3f} "
                    f"{self._truncate(snippet, 120)}"
                )
        else:
            lines.append("no evidence refs")

        self._render_panel("Retrieval Trace", lines, tone="retrieval")

    def _render_query_observation(self, observation: QueryObservation) -> None:
        trace = observation.trace
        query_type = getattr(getattr(trace, "query_type", None), "name", "UNKNOWN")
        abstention = getattr(trace, "abstention", None)
        abstain_flag = bool(getattr(abstention, "should_abstain", False))
        abstain_reason = getattr(abstention, "reason", "")
        timeline_group = getattr(trace, "timeline_group", None)
        timeline_count = len(timeline_group.timelines) if timeline_group else 0
        standalone_count = len(timeline_group.standalone_results) if timeline_group else 0

        lines = [
            (
                f"retrieve={observation.retrieval_ms:.1f}ms | query_type={query_type} | "
                f"hits={len(trace.ranked)} | attractor_path={len(trace.attractor_path)} | "
                f"timelines={timeline_count} | standalone={standalone_count} | "
                f"abstain={abstain_flag}"
            )
        ]
        if abstain_reason:
            lines.append(f"abstention_reason={abstain_reason}")

        if not trace.ranked:
            lines.append("no hits")
            self._render_panel("Retrieval Trace", lines, tone="retrieval")
            return

        for index, (node_id, score, kind) in enumerate(trace.ranked[: self.max_hits], start=1):
            snippet = self._resolve_snippet(node_id=node_id, kind=kind)
            lines.append(
                f"{index}. [{kind}] {node_id} score={score:.3f} {self._truncate(snippet, 120)}"
            )

        self._render_panel("Retrieval Trace", lines, tone="retrieval")

    def _render_artifacts_and_evidence(self, result: ChatTurnResult) -> None:
        lines: List[str] = []
        for ref in result.memory_context.evidence_refs[: self.max_hits]:
            snippet = self._resolve_snippet(node_id=ref.id, kind=ref.kind)
            lines.append(
                f"{ref.kind}:{ref.id} | role={ref.role} | score={ref.score:.3f} | "
                f"{self._truncate(snippet, 96)}"
            )

        plot_doc = self.runtime.doc_store.get(result.ingest_result.plot_id)
        if plot_doc is not None:
            body = plot_doc.body
            actors = body.get("resolved_actors") or body.get("actors") or []
            claims = body.get("claims") or []
            lines.extend(
                [
                    "",
                    f"latest_plot.actors={actors}",
                    f"latest_plot.action={body.get('action', '')}",
                    f"latest_plot.outcome={body.get('outcome', '')}",
                    f"latest_plot.claims={len(claims)}",
                ]
            )

        self._render_panel("Artifacts / Evidence", lines or ["no evidence"], tone="artifact")

    def _render_memory_totals(self) -> None:
        self._render_panel(
            "Memory Totals",
            [
                f"plots={len(self.runtime.mem.plots)}",
                f"stories={len(self.runtime.mem.stories)}",
                f"themes={len(self.runtime.mem.themes)}",
                f"last_seq={self.runtime.last_seq}",
            ],
            tone="memory",
        )

    def _render_stats(self) -> None:
        settings = self.runtime.settings
        payload = {
            "session_id": self.session_id,
            "observe_mode": self.observe_mode,
            "data_dir": settings.data_dir,
            "llm_provider": self._llm_label(settings),
            "embedding_provider": self._embedding_label(settings),
            "plots": len(self.runtime.mem.plots),
            "stories": len(self.runtime.mem.stories),
            "themes": len(self.runtime.mem.themes),
            "last_seq": self.runtime.last_seq,
        }
        self._render_panel("Stats", json.dumps(payload, ensure_ascii=False, indent=2).splitlines(), tone="accent")

    def _render_events(self, limit: int) -> None:
        if self.runtime.last_seq <= 0:
            self._write("还没有事件。")
            return

        after_seq = max(0, self.runtime.last_seq - limit)
        rows = list(self.runtime.event_log.iter_events(after_seq=after_seq))
        if not rows:
            self._write("还没有事件。")
            return

        lines: List[str] = []
        for seq, event in rows:
            user_message = self._truncate(str(event.payload.get("user_message", "")), 56)
            agent_message = self._truncate(str(event.payload.get("agent_message", "")), 56)
            lines.append(
                f"{seq:>4} | {self._fmt_ts(event.ts)} | {event.id} | {event.session_id} | "
                f"user={user_message}"
            )
            lines.append(f"       agent={agent_message}")
        self._render_panel("Recent Events", lines, tone="neutral")

    def _render_plots(self, limit: int) -> None:
        docs = list(self.runtime.doc_store.iter_kind(kind="plot", limit=limit))
        if not docs:
            self._write("还没有 plot 文档。")
            return

        lines: List[str] = []
        for doc in docs:
            state = doc.body.get("plot_state", {})
            lines.append(
                f"{doc.id} | {self._fmt_ts(doc.ts)} | story={state.get('story_id') or '-'} | "
                f"status={state.get('status', '-')}"
            )
            lines.append(f"    {self._truncate(str(state.get('text', '')), 120)}")
        self._render_panel("Recent Plots", lines, tone="neutral")

    def _render_stories(self, limit: int) -> None:
        stories = sorted(
            self.runtime.mem.stories.values(),
            key=lambda story: story.updated_ts,
            reverse=True,
        )[:limit]
        if not stories:
            self._write("还没有 story。")
            return

        lines: List[str] = []
        for story in stories:
            lines.append(
                f"{story.id} | plots={len(story.plot_ids)} | status={story.status} | "
                f"relationship={story.relationship_with or '-'} | identity={story.my_identity_in_this_relationship or '-'}"
            )
        self._render_panel("Recent Stories", lines, tone="neutral")

    def _render_themes(self, limit: int) -> None:
        themes = sorted(
            self.runtime.mem.themes.values(),
            key=lambda theme: theme.updated_ts,
            reverse=True,
        )[:limit]
        if not themes:
            self._write("还没有 theme。")
            return

        lines: List[str] = []
        for theme in themes:
            lines.append(
                f"{theme.id} | type={theme.theme_type} | confidence={theme.confidence():.3f} | "
                f"identity={theme.identity_dimension or theme.name or '-'}"
            )
        self._render_panel("Recent Themes", lines, tone="neutral")

    def _render_inspect(self, item_id: str) -> None:
        plot = self.runtime.mem.plots.get(item_id)
        if plot is not None:
            payload = {
                "id": plot.id,
                "ts": plot.ts,
                "story_id": plot.story_id,
                "status": plot.status,
                "actors": list(plot.actors),
                "text": plot.text,
                "knowledge_type": plot.knowledge_type,
                "redundancy_type": plot.redundancy_type,
                "supersedes_id": plot.supersedes_id,
                "superseded_by_id": plot.superseded_by_id,
                "tension": plot.tension,
                "surprise": plot.surprise,
                "pred_error": plot.pred_error,
                "redundancy": plot.redundancy,
            }
            self._render_panel("Inspect Plot", json.dumps(payload, ensure_ascii=False, indent=2).splitlines(), tone="neutral")
            return

        story = self.runtime.mem.stories.get(item_id)
        if story is not None:
            payload = {
                "id": story.id,
                "plot_count": len(story.plot_ids),
                "status": story.status,
                "relationship_with": story.relationship_with,
                "relationship_type": story.relationship_type,
                "identity": story.my_identity_in_this_relationship,
                "central_conflict": story.central_conflict,
                "moral": story.moral,
            }
            self._render_panel("Inspect Story", json.dumps(payload, ensure_ascii=False, indent=2).splitlines(), tone="neutral")
            return

        theme = self.runtime.mem.themes.get(item_id)
        if theme is not None:
            payload = {
                "id": theme.id,
                "identity_dimension": theme.identity_dimension,
                "theme_type": theme.theme_type,
                "confidence": theme.confidence(),
                "supporting_relationships": theme.supporting_relationships,
                "tensions_with": theme.tensions_with,
                "harmonizes_with": theme.harmonizes_with,
            }
            self._render_panel("Inspect Theme", json.dumps(payload, ensure_ascii=False, indent=2).splitlines(), tone="neutral")
            return

        doc = self.runtime.doc_store.get(item_id)
        if doc is not None:
            payload = {
                "id": doc.id,
                "kind": doc.kind,
                "ts": doc.ts,
                "keys": sorted(doc.body.keys()),
            }
            if doc.kind == "plot":
                plot_state = doc.body.get("plot_state", {})
                payload["summary"] = {
                    "text": plot_state.get("text", ""),
                    "story_id": plot_state.get("story_id"),
                    "status": plot_state.get("status"),
                }
            self._render_panel("Inspect Document", json.dumps(payload, ensure_ascii=False, indent=2).splitlines(), tone="neutral")
            return

        self._write(f"没有找到 {item_id}")

    def _render_coherence(self) -> None:
        result = self.runtime.check_coherence()
        payload = {
            "overall_score": result.overall_score,
            "conflict_count": result.conflict_count,
            "unfinished_story_count": result.unfinished_story_count,
            "recommendations": result.recommendations,
        }
        self._render_panel("Coherence", json.dumps(payload, ensure_ascii=False, indent=2).splitlines(), tone="accent")

    def _render_narrative(self) -> None:
        narrative = self.runtime.get_self_narrative()
        payload = {
            "profile_id": narrative["profile_id"],
            "identity_statement": narrative["identity_statement"],
            "seed_narrative": narrative["seed_narrative"],
            "capability_narrative": narrative["capability_narrative"],
            "coherence_score": narrative["coherence_score"],
            "trait_beliefs": narrative["trait_beliefs"],
            "capability_count": len(narrative["capabilities"]),
            "relationship_count": len(narrative["relationships"]),
            "subconscious": narrative["subconscious"],
            "unresolved_tensions": narrative["unresolved_tensions"],
        }
        self._render_panel("Self Narrative", json.dumps(payload, ensure_ascii=False, indent=2).splitlines(), tone="accent")

    def _render_evolve(self) -> None:
        before = {
            "plots": len(self.runtime.mem.plots),
            "stories": len(self.runtime.mem.stories),
            "themes": len(self.runtime.mem.themes),
        }
        self.runtime.evolve()
        after = {
            "plots": len(self.runtime.mem.plots),
            "stories": len(self.runtime.mem.stories),
            "themes": len(self.runtime.mem.themes),
        }
        payload = {"before": before, "after": after}
        self._render_panel("Evolution", json.dumps(payload, ensure_ascii=False, indent=2).splitlines(), tone="accent")

    def _render_help(self) -> None:
        self._render_panel(
            "Commands",
            [
                "/help                 查看帮助",
                "/pipeline             查看每轮处理链路",
                "/observe MODE         观测模式: off | brief | full",
                "/prompt               回看上一次发给 LLM 的最终 prompt",
                "/context              回看上一次结构化 memory context",
                "/stats                查看运行时统计",
                "/query <text>         只做检索，不生成回复",
                "/events [n]           查看最近 n 条事件",
                "/plots [n]            查看最近 n 个 plot 文档",
                "/stories [n]          查看最近 n 个故事",
                "/themes [n]           查看最近 n 个主题",
                "/inspect <id>         检查 plot/story/theme/doc",
                "/coherence            查看一致性报告",
                "/narrative            查看自我叙事摘要",
                "/evolve               手动触发演化",
                "/clear                清屏并重绘控制台",
                "/quit                 退出",
                "",
                "直接输入文本就是正常对话。",
            ],
            tone="neutral",
        )
        self._render_command_bar()

    def _print_welcome(self) -> None:
        settings = self.runtime.settings
        badges = " ".join(
            [
                self._badge(f"session {self.session_id}", tone="user"),
                self._badge(f"llm {self._llm_label(settings)}", tone="accent"),
                self._badge(f"embed {self._embedding_label(settings)}", tone="muted"),
                self._badge(f"mode {self.observe_mode}", tone="prompt"),
            ]
        )
        self._render_panel(
            "AURORA Live Console",
            [
                badges,
                f"data_dir={settings.data_dir}",
                "",
                *PIPELINE_TEXT.rstrip().splitlines(),
                "",
                "直接输入文本即可对话，输入 /help 查看命令。",
            ],
            tone="accent",
        )
        self._render_command_bar()

    def _render_prompt(self, observation: ChatTurnResult) -> None:
        system_lines = self._numbered_block("system", observation.system_prompt)
        user_lines = self._numbered_block("user", observation.user_prompt)
        lines = [
            self._badge(
                f"system {len(observation.system_prompt)} chars / {len(observation.system_prompt.splitlines()) or 1} lines",
                tone="assistant",
            ),
            *system_lines,
            "",
            self._badge(
                f"user {len(observation.user_prompt)} chars / {len(observation.user_prompt.splitlines()) or 1} lines",
                tone="prompt",
            ),
            *user_lines,
        ]
        self._render_panel("LLM Prompt", lines, tone="prompt")

    def _render_last_prompt(self) -> None:
        if self.last_observation is None:
            self._write("还没有可显示的 prompt。")
            return
        self._render_prompt(self.last_observation)

    def _render_last_context(self) -> None:
        if self.last_observation is None:
            self._write("还没有可显示的 memory context。")
            return
        self._render_context_panel(self.last_observation.memory_context)

    def _render_context_panel(self, context: StructuredMemoryContext) -> None:
        lines: List[str] = []
        sections = [
            ("known_facts", context.known_facts),
            ("preferences", context.preferences),
            ("relationship_state", context.relationship_state),
            ("active_narratives", context.active_narratives),
            ("temporal_context", context.temporal_context),
            ("cautions", context.cautions),
        ]
        for name, items in sections:
            lines.append(f"[{name}]")
            if items:
                lines.extend(f"- {item}" for item in items)
            else:
                lines.append("- none")
            lines.append("")

        lines.append("[evidence_refs]")
        if context.evidence_refs:
            lines.extend(
                f"- {ref.kind}:{ref.id} score={ref.score:.3f} role={ref.role}"
                for ref in context.evidence_refs
            )
        else:
            lines.append("- none")
        self._render_panel("Memory Context", lines, tone="accent")

    def _resolve_snippet(self, *, node_id: str, kind: str) -> str:
        if kind == "plot":
            plot = self.runtime.mem.plots.get(node_id)
            return plot.text if plot else ""

        doc = self.runtime.doc_store.get(node_id)
        if doc is not None:
            return str(doc.body.get("summary") or doc.body.get("description") or "")

        if kind == "story":
            story = self.runtime.mem.stories.get(node_id)
            if story is not None:
                return story.central_conflict or story.moral or ""

        if kind == "theme":
            theme = self.runtime.mem.themes.get(node_id)
            if theme is not None:
                return theme.identity_dimension or theme.description or ""

        return ""

    def _render_command_bar(self, *, note: Optional[str] = None) -> None:
        commands = [
            self._badge("/prompt", tone="prompt"),
            self._badge("/context", tone="accent"),
            self._badge("/query", tone="accent"),
            self._badge("/events", tone="muted"),
            self._badge("/stats", tone="muted"),
            self._badge("/clear", tone="warning"),
            self._badge("/quit", tone="warning"),
        ]
        line = " ".join(commands)
        if note:
            line = f"{line}  {note}"
        self._write(self._style(line, "muted"))

    def _show_live_status(self, label: str, *, detail: Optional[str] = None) -> None:
        if not self._is_tty:
            return
        message = f"… {label}"
        if detail:
            message = f"{message}  {self._truncate(detail, 72)}"
        self.output.write("\r\033[2K" + self._style(message, "accent", "dim"))
        self.output.flush()
        self._status_visible = True

    def _clear_live_status(self) -> None:
        if not (self._is_tty and self._status_visible):
            return
        self.output.write("\r\033[2K")
        self.output.flush()
        self._status_visible = False

    @staticmethod
    def _llm_label(settings: AuroraSettings) -> str:
        if settings.llm_provider == "bailian":
            return f"bailian:{settings.bailian_llm_model}"
        if settings.llm_provider == "ark":
            return f"ark:{settings.ark_llm_model}"
        return "mock"

    @staticmethod
    def _embedding_label(settings: AuroraSettings) -> str:
        if settings.embedding_provider == "bailian":
            return f"bailian:{settings.bailian_embedding_model}"
        if settings.embedding_provider == "ark":
            return "ark"
        return settings.embedding_provider

    @staticmethod
    def _parse_limit(args: Sequence[str], *, default: int) -> int:
        if not args:
            return default
        try:
            return max(1, int(args[0]))
        except ValueError:
            return default

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _render_panel(self, title: str, lines: Sequence[str], *, tone: str = "neutral") -> None:
        body_width = max(72, min(self._terminal_width() - 4, 140))
        border_tone = PANEL_TONES.get(tone, "border")
        title_text = f" {title} "
        top = self._style("┌", border_tone) + self._style(title_text, "bold", "title") + self._style(
            "─" * max(0, body_width - len(title_text)) + "┐",
            border_tone,
        )
        bottom = self._style("└" + "─" * body_width + "┘", border_tone)

        self._write(top)
        for raw_line in lines or [""]:
            logical_lines = str(raw_line).splitlines() or [""]
            for logical_line in logical_lines:
                wrapped = textwrap.wrap(
                    logical_line,
                    width=body_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                ) or [""]
                for chunk in wrapped:
                    self._write(
                        self._style("│", border_tone)
                        + chunk.ljust(body_width)
                        + self._style("│", border_tone)
                    )
        self._write(bottom)

    @staticmethod
    def _terminal_width() -> int:
        return shutil.get_terminal_size(fallback=(120, 24)).columns

    def _input_prompt(self) -> str:
        label = self._badge("you", tone="user")
        return f"{label} "

    def _badge(self, text: str, *, tone: str) -> str:
        if not self._use_color:
            return f"[{text}]"
        bg, fg = BADGE_PALETTE.get(tone, BADGE_PALETTE["muted"])
        return f"\033[1;{bg};{fg}m {text} \033[0m"

    def _style(self, text: str, *tokens: str) -> str:
        if not self._use_color:
            return text

        codes: List[str] = []
        for token in tokens:
            if token == "bold":
                codes.append("1")
            elif token == "dim":
                codes.append("2")
            else:
                color = COLOR_PALETTE.get(token)
                if color:
                    codes.append(color)
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def _numbered_block(self, label: str, text: str) -> List[str]:
        lines = [f"[{label}]"]
        source_lines = text.splitlines() or [""]
        for index, line in enumerate(source_lines, start=1):
            lines.append(f"{index:>3} | {line}")
        return lines

    def _clear_screen(self) -> None:
        if not self._is_tty:
            return
        self.output.write("\033[2J\033[H")
        self.output.flush()

    def _write(self, message: str) -> None:
        self.output.write(message + "\n")
        self.output.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="observe_runtime",
        description="在终端对话并观察 AURORA 运行过程。",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录，默认读取 AURORA_DATA_DIR 或 ./data")
    parser.add_argument("--session-id", type=str, default="terminal_observer", help="会话 ID")
    parser.add_argument("--max-hits", type=int, default=6, help="每次展示的最大检索命中数")
    parser.add_argument(
        "--observe",
        choices=list(OBSERVE_MODES),
        default="full",
        help="观测输出级别",
    )
    return parser


def run_observer(
    *,
    data_dir: Optional[str] = None,
    session_id: str = "terminal_observer",
    max_hits: int = 6,
    observe_mode: str = "full",
) -> None:
    setup_logging("WARNING")
    settings_kwargs: Dict[str, Any] = {}
    if data_dir:
        settings_kwargs["data_dir"] = data_dir
    settings = AuroraSettings(**settings_kwargs)
    runtime = AuroraRuntime(settings=settings)
    observer = TerminalObserver(
        runtime,
        session_id=session_id,
        max_hits=max_hits,
        observe_mode=observe_mode,
    )
    observer.run()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_observer(
        data_dir=args.data_dir,
        session_id=args.session_id,
        max_hits=args.max_hits,
        observe_mode=args.observe,
    )


if __name__ == "__main__":
    main()
