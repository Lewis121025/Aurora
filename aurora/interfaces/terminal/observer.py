from __future__ import annotations

import argparse
import os
import shlex
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, TextIO

from aurora.runtime.results import ChatTurnResult, StructuredMemoryContext
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings

OBSERVE_MODES = ("chat", "brief", "full")
COLORS = {
    "accent": "38;5;81",
    "assistant": "38;5;114",
    "user": "38;5;117",
    "muted": "38;5;245",
    "border": "38;5;240",
    "danger": "38;5;203",
    "prompt": "38;5;221",
}


@dataclass(frozen=True)
class Command:
    name: str
    args: tuple[str, ...]


def parse_command(line: str) -> Optional[Command]:
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:].strip()
    if not body:
        return Command("help", ())
    parts = shlex.split(body)
    if not parts:
        return Command("help", ())
    return Command(parts[0].lower(), tuple(parts[1:]))


class TerminalObserver:
    def __init__(
        self,
        runtime: AuroraRuntime,
        *,
        session_id: str,
        max_hits: int = 6,
        observe_mode: str = "chat",
        output: TextIO = sys.stdout,
    ) -> None:
        if observe_mode not in OBSERVE_MODES:
            raise ValueError(f"Invalid observe mode: {observe_mode}")
        self.runtime = runtime
        self.session_id = session_id
        self.max_hits = max_hits
        self.observe_mode = observe_mode
        self.output = output
        self.turn_count = 0
        self.last_result: Optional[ChatTurnResult] = None
        self._is_tty = bool(getattr(output, "isatty", lambda: False)())
        self._use_color = self._is_tty and os.environ.get("NO_COLOR") is None
        self._prompt_session = self._build_prompt_session()

    def _build_prompt_session(self) -> Optional[Any]:
        if not self._is_tty:
            return None
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
            from prompt_toolkit.history import FileHistory
        except Exception:
            return None

        history_dir = str(self.runtime.settings.data_dir or ".")
        os.makedirs(history_dir, exist_ok=True)
        history = FileHistory(os.path.join(history_dir, ".aurora_history"))
        return PromptSession(history=history, auto_suggest=AutoSuggestFromHistory(), reserve_space_for_menu=0)

    def run(self) -> None:
        self._render_welcome()
        while True:
            try:
                line = self._read_line()
            except (EOFError, KeyboardInterrupt):
                self._write("\nbye")
                return
            if not self.handle_line(line):
                return

    def _read_line(self) -> str:
        prompt_text = self._prompt_text()
        if self._prompt_session is None:
            return input(prompt_text)
        from prompt_toolkit.formatted_text import ANSI

        return self._prompt_session.prompt(
            ANSI(self._ansi(prompt_text, "prompt")),
            bottom_toolbar=ANSI(self._ansi(self._toolbar_text(), "muted")),
            mouse_support=False,
            wrap_lines=False,
        )

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
        self._status("Aurora is thinking")
        result = self.runtime.respond(
            session_id=self.session_id,
            user_message=user_message,
            k=self.max_hits,
        )
        self._clear_status()
        self.last_result = result
        return result

    def _handle_command(self, command: Command) -> bool:
        name = command.name
        args = list(command.args)
        if name in {"quit", "exit"}:
            self._write("bye")
            return False
        if name in {"chat", "brief", "full"}:
            self.observe_mode = name
            self._write(self._muted(f"mode -> {name}"))
            return True
        if name == "observe":
            mode = args[0].lower() if args else "brief"
            if mode not in OBSERVE_MODES:
                self._write("observe mode 必须是 chat / brief / full")
                return True
            self.observe_mode = mode
            self._write(self._muted(f"mode -> {mode}"))
            return True
        if name == "help":
            self._render_help()
            return True
        if name == "clear":
            if self._is_tty:
                self.output.write("\033[2J\033[H")
                self.output.flush()
            self._render_welcome()
            return True
        if name == "identity":
            self._render_identity()
            return True
        if name == "stats":
            self._render_stats()
            return True
        if name == "context":
            if self.last_result is None:
                self._write(self._muted("还没有可展示的上下文。"))
            else:
                self._render_context(self.last_result.memory_context)
            return True
        if name == "prompt":
            if self.last_result is None:
                self._write(self._muted("还没有 prompt。"))
            else:
                self._render_panel("Prompt", self.last_result.system_prompt.splitlines() + ["", self.last_result.user_prompt], tone="muted")
            return True
        if name == "query":
            if not args:
                self._write("用法: /query <text>")
                return True
            result = self.runtime.query(text=" ".join(args), k=self.max_hits)
            lines = [f"query={result.query}", f"attractor_path_len={result.attractor_path_len}", ""]
            for idx, hit in enumerate(result.hits, start=1):
                lines.append(f"{idx}. [{hit.kind}] {hit.score:.3f} {hit.snippet}")
            self._render_panel("Query", lines, tone="accent")
            return True
        if name == "events":
            self._render_events(limit=self._parse_limit(args, default=6))
            return True
        if name == "plots":
            self._render_plots(limit=self._parse_limit(args, default=6))
            return True
        if name == "stories":
            self._render_stories(limit=self._parse_limit(args, default=6))
            return True
        if name == "themes":
            self._render_themes(limit=self._parse_limit(args, default=6))
            return True
        self._write(self._muted(f"未知命令: /{name}"))
        return True

    def _render_welcome(self) -> None:
        identity = self.runtime.get_identity()
        summary = identity["narrative_summary"]
        top_axes = self._format_axis_pairs(identity["identity"]["axis_state"], limit=3)
        lines = [
            self._accent("Aurora Soul"),
            self._muted(
                f"mode={identity['identity']['current_mode']}  pressure={summary['pressure']:.3f}  top_axes={top_axes}"
            ),
            "",
            "直接输入文本即可对话。",
            "常用命令: /identity  /stats  /brief  /full  /context  /quit",
        ]
        self._write("\n".join(lines))

    def _render_help(self) -> None:
        self._render_panel(
            "Commands",
            [
                "/chat                只显示对话",
                "/brief               回复后显示一屏摘要",
                "/full                展开完整上下文",
                "/identity            查看当前身份快照",
                "/stats               查看 plot/story/theme 与能量统计",
                "/context             查看最近一轮 soul-memory brief",
                "/prompt              查看最近一轮 prompt",
                "/query <text>        单独观察检索结果",
                "/events [n]          最近事件",
                "/plots [n]           最近 plot",
                "/stories [n]         story 概览",
                "/themes [n]          theme 概览",
                "/clear               清屏",
                "/quit                退出",
            ],
            tone="muted",
        )

    def _render_chat_result(self, result: ChatTurnResult) -> None:
        self._write("")
        self._write(self._assistant(f"Aurora · Turn {self.turn_count:02d}"))
        self._write(self._wrap(result.reply))
        if self.observe_mode == "chat":
            self._write("")
            return
        self._render_summary(result)
        if self.observe_mode == "full":
            self._render_context(result.memory_context)

    def _render_summary(self, result: ChatTurnResult) -> None:
        ctx = result.memory_context
        identity = ctx.identity
        top_axes = self._format_axis_pairs(identity.axis_state, limit=3) if identity else "-"
        lines = [
            f"mode={ctx.mode} | pressure={ctx.narrative_pressure:.3f}",
            f"active={identity.active_energy:.3f} | repressed={identity.repressed_energy:.3f}" if identity else "",
            f"top_axes={top_axes}",
            f"intuition={', '.join(ctx.intuition) if ctx.intuition else '-'}",
            f"hits={len(ctx.evidence_refs)} | retrieval_ms={result.timings.retrieval_ms:.1f} | "
            f"gen_ms={result.timings.generation_ms:.1f} | ingest_ms={result.timings.ingest_ms:.1f}",
        ]
        lines.extend(f"- {hit}" for hit in ctx.retrieval_hits[: self.max_hits])
        self._render_panel("Observe", [line for line in lines if line], tone="accent")

    def _render_context(self, context: StructuredMemoryContext) -> None:
        identity = context.identity
        summary = context.narrative_summary
        lines = [
            f"mode={context.mode}",
            f"pressure={context.narrative_pressure:.3f}",
            f"intuition={', '.join(context.intuition) if context.intuition else '-'}",
            "",
            f"summary: {summary.text if summary else '-'}",
        ]
        if identity is not None:
            lines.extend(
                [
                    "",
                    f"top_axes: {self._format_axis_pairs(identity.axis_state, limit=6)}",
                    f"intuition_axes: {self._format_axis_pairs(identity.intuition_axes, limit=4)}",
                    f"persona_axes={len(identity.persona_axes)} aliases={len(identity.axis_aliases)} modes={len(identity.modes)}",
                    f"repair_count={identity.repair_count} dream_count={identity.dream_count} mode_change_count={identity.mode_change_count}",
                    "narrative_tail:",
                ]
            )
            lines.extend(f"- {item}" for item in identity.narrative_tail[-6:])
        if summary is not None and summary.salient_axes:
            lines.extend(["", f"salient_axes: {', '.join(summary.salient_axes[:6])}"])
        if context.retrieval_hits:
            lines.append("")
            lines.append("retrieval_hits:")
            lines.extend(f"- {item}" for item in context.retrieval_hits)
        self._render_panel("Context", lines, tone="muted")

    def _render_identity(self) -> None:
        report = self.runtime.get_identity()
        identity = report["identity"]
        summary = report["narrative_summary"]
        lines = [
            f"mode={identity['current_mode']}",
            f"pressure={summary['pressure']:.3f}",
            f"summary={summary['text']}",
            "",
            f"top_axes={self._format_axis_pairs(identity['axis_state'], limit=8)}",
            f"intuition_axes={self._format_axis_pairs(identity['intuition_axes'], limit=6)}",
            f"persona_axes={len(identity['persona_axes'])} aliases={len(identity['axis_aliases'])} modes={len(identity['modes'])}",
            f"active_energy={identity['active_energy']:.3f} repressed_energy={identity['repressed_energy']:.3f}",
            f"repair_count={identity['repair_count']} dream_count={identity['dream_count']} mode_change_count={identity['mode_change_count']}",
            "",
            "salient_axes:",
        ]
        lines.extend(f"- {item}" for item in summary["salient_axes"][:6])
        lines.extend(
            [
                "",
            "narrative_tail:",
            ]
        )
        lines.extend(f"- {item}" for item in identity["narrative_tail"][-6:])
        self._render_panel("Identity", lines, tone="accent")

    def _render_stats(self) -> None:
        stats = self.runtime.get_stats()
        lines = [
            f"plots={stats['plot_count']} stories={stats['story_count']} themes={stats['theme_count']}",
            f"mode={stats['current_mode']} pressure={stats['pressure']:.3f}",
            f"dream_count={stats['dream_count']} repair_count={stats['repair_count']}",
            f"active_energy={stats['active_energy']:.3f} repressed_energy={stats['repressed_energy']:.3f}",
        ]
        self._render_panel("Stats", lines, tone="muted")

    def _render_events(self, *, limit: int) -> None:
        lines = []
        for seq, event in self.runtime.event_log.iter_events(after_seq=max(0, self.runtime.last_seq - limit)):
            lines.append(f"{seq}. {event.id} session={event.session_id} ts={event.ts:.0f}")
            lines.append(f"   user={str(event.payload.get('user_message', ''))[:96]}")
        self._render_panel("Events", lines or ["no events"], tone="muted")

    def _render_plots(self, *, limit: int) -> None:
        plots = sorted(self.runtime.mem.plots.values(), key=lambda plot: plot.ts, reverse=True)[:limit]
        lines = [
            f"{plot.id[:8]} source={plot.source} story={plot.story_id or '-'} tension={plot.tension:.3f} :: {plot.text[:140]}"
            for plot in plots
        ]
        self._render_panel("Plots", lines or ["no plots"], tone="muted")

    def _render_stories(self, *, limit: int) -> None:
        stories = sorted(self.runtime.mem.stories.values(), key=lambda story: story.updated_ts, reverse=True)[:limit]
        lines = [
            f"{story.id[:8]} status={story.status} plots={len(story.plot_ids)} unresolved={story.unresolved_energy:.3f}"
            for story in stories
        ]
        self._render_panel("Stories", lines or ["no stories"], tone="muted")

    def _render_themes(self, *, limit: int) -> None:
        themes = sorted(self.runtime.mem.themes.values(), key=lambda theme: theme.updated_ts, reverse=True)[:limit]
        lines = [
            f"{theme.id[:8]} conf={theme.confidence():.2f} name={(theme.name or theme.description)[:120]}"
            for theme in themes
        ]
        self._render_panel("Themes", lines or ["no themes"], tone="muted")

    def _parse_limit(self, args: Sequence[str], *, default: int) -> int:
        if not args:
            return default
        try:
            return max(1, int(args[0]))
        except Exception:
            return default

    def _prompt_text(self) -> str:
        return "you › "

    def _toolbar_text(self) -> str:
        return f" mode={self.observe_mode}  /identity  /stats  /brief  /full  /help "

    @staticmethod
    def _format_axis_pairs(values: Dict[str, float], *, limit: int) -> str:
        items = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)
        formatted = [f"{name}={value:+.2f}" for name, value in items[:limit] if abs(value) >= 0.05]
        return ", ".join(formatted) if formatted else "-"

    def _render_panel(self, title: str, lines: Sequence[str], *, tone: str) -> None:
        border = "─" * 12
        self._write(self._style(f"{border} {title} {border}", tone))
        for line in lines:
            self._write(self._wrap(line))
        self._write("")

    def _status(self, label: str) -> None:
        if not self._is_tty:
            return
        self.output.write("\r" + self._ansi(f"… {label}", "muted"))
        self.output.flush()

    def _clear_status(self) -> None:
        if not self._is_tty:
            return
        self.output.write("\r\033[2K")
        self.output.flush()

    def _wrap(self, text: str, width: int = 96) -> str:
        return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False, drop_whitespace=False)) or text

    def _write(self, text: str) -> None:
        self.output.write(text + "\n")
        self.output.flush()

    def _style(self, text: str, tone: str) -> str:
        return self._ansi(text, tone)

    def _ansi(self, text: str, tone: str) -> str:
        if not self._use_color:
            return text
        code = COLORS.get(tone, COLORS["muted"])
        return f"\033[{code}m{text}\033[0m"

    def _accent(self, text: str) -> str:
        return self._ansi(text, "accent")

    def _assistant(self, text: str) -> str:
        return self._ansi(text, "assistant")

    def _muted(self, text: str) -> str:
        return self._ansi(text, "muted")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="observe_runtime", description="Aurora Soul terminal observer")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--session-id", type=str, default="terminal_observer")
    parser.add_argument("--max-hits", type=int, default=6)
    parser.add_argument("--observe", choices=list(OBSERVE_MODES), default="chat")
    return parser


def run_observer(
    *,
    data_dir: Optional[str] = None,
    session_id: str = "terminal_observer",
    max_hits: int = 6,
    observe_mode: str = "chat",
) -> None:
    settings = AuroraSettings(data_dir=data_dir or os.environ.get("AURORA_DATA_DIR", "./data"))
    runtime = AuroraRuntime(settings=settings)
    observer = TerminalObserver(runtime, session_id=session_id, max_hits=max_hits, observe_mode=observe_mode)
    observer.run()


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_observer(
        data_dir=args.data_dir,
        session_id=args.session_id,
        max_hits=args.max_hits,
        observe_mode=args.observe,
    )
