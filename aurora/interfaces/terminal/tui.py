from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence

if TYPE_CHECKING:
    from rich import box
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Input, Static, Tab, Tabs

    TERMINAL_UI_DEPS_AVAILABLE = True
else:
    try:
        from rich import box
        from rich.align import Align
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text
        from textual import on, work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical, VerticalScroll
        from textual.widgets import Input, Static, Tab, Tabs

        TERMINAL_UI_DEPS_AVAILABLE = True
    except ModuleNotFoundError:
        TERMINAL_UI_DEPS_AVAILABLE = False

        class _Box:
            ROUNDED = "rounded"

        box = _Box()

        class Align:
            @staticmethod
            def right(renderable: Any) -> Any:
                return renderable

            @staticmethod
            def center(renderable: Any) -> Any:
                return renderable

            @staticmethod
            def left(renderable: Any) -> Any:
                return renderable

        class Group(tuple[Any, ...]):
            def __new__(cls, *items: Any) -> "Group":
                return super().__new__(cls, items)

        class Panel:
            @staticmethod
            def fit(renderable: Any, **_: Any) -> Any:
                return renderable

        class Text:
            def __init__(self, text: str = "", *, style: str | None = None) -> None:
                self._parts = [text]
                self.style = style
                self.no_wrap = False

            def append(self, text: str, *, style: str | None = None) -> None:
                self._parts.append(text)

            def __str__(self) -> str:
                return "".join(self._parts)

        def on(*_args: Any, **_kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        def work(*_args: Any, **_kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        class App:
            CSS = ""
            BINDINGS: list[object] = []

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __class_getitem__(cls, _item: object) -> type["App"]:
                return cls

            def run(self) -> None:
                raise RuntimeError(
                    "Aurora terminal UI requires optional dependencies 'rich' and 'textual'."
                )

            def query_one(self, *_args: Any, **_kwargs: Any) -> "_WidgetStub":
                return _WidgetStub()

        ComposeResult = object

        class Binding:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

        class _WidgetStub:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> "_WidgetStub":
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
                return False

            def update(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def remove_children(self) -> None:
                pass

            def mount(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def scroll_end(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def focus(self) -> None:
                pass

            def remove_class(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def add_class(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def remove(self) -> None:
                pass

        class Horizontal(_WidgetStub):
            pass

        class Vertical(_WidgetStub):
            pass

        class VerticalScroll(_WidgetStub):
            pass

        class Input(_WidgetStub):
            class Submitted:
                def __init__(self, value: str = "", input: object | None = None) -> None:
                    self.value = value
                    self.input = input or _WidgetStub()

        class Static(_WidgetStub):
            pass

        class Tab(_WidgetStub):
            pass

        class Tabs(_WidgetStub):
            class TabActivated:
                def __init__(self, tab: object | None = None) -> None:
                    self.tab = tab or _WidgetStub()

from aurora.runtime.results import (
    ChatStreamEvent,
    ChatTurnResult,
    QueryResult,
    StructuredMemoryContext,
)
from aurora.runtime.runtime import AuroraRuntime
from aurora.soul.models import Message, TextPart, messages_to_text

SOURCE_LABELS = {
    "wake": "清醒编码",
    "dream": "梦整合",
    "repair": "叙事修复",
    "mode": "模式切换",
}

MAX_STORED_CHAT_ENTRIES = 120
MAX_STORED_TURNS = 60
MAX_VISIBLE_TURNS = 6

INSPECTOR_TABS: Sequence[tuple[str, str]] = (
    ("overview", "概览"),
    ("prompt", "提示词"),
    ("retrieval", "检索"),
    ("memory", "记忆"),
    ("writeback", "写回"),
)

TAB_LABELS = {key: label for key, label in INSPECTOR_TABS}
TAB_LOOKUP = {
    "overview": "overview",
    "概览": "overview",
    "prompt": "prompt",
    "提示": "prompt",
    "提示词": "prompt",
    "retrieval": "retrieval",
    "检索": "retrieval",
    "memory": "memory",
    "记忆": "memory",
    "context": "memory",
    "上下文": "memory",
    "writeback": "writeback",
    "写回": "writeback",
}

COMMAND_ALIASES = {
    "帮助": "help",
    "清屏": "clear",
    "退出": "quit",
    "身份": "identity",
    "状态": "stats",
    "上下文": "memory",
    "提示": "prompt",
    "查询": "query",
    "事件": "events",
    "情节": "plots",
    "故事": "stories",
    "主题": "themes",
    "面板": "tab",
    "回合": "turn",
}


@dataclass(frozen=True)
class ChatEntry:
    title: str
    body: str


@dataclass(frozen=True)
class StatusRecord:
    title: str
    body: str


@dataclass(frozen=True)
class TurnRecord:
    turn_no: int
    user_text: str
    result: ChatTurnResult
    live_identity: dict[str, Any]
    live_summary: dict[str, Any]
    live_stats: dict[str, Any]


def _format_axis_pairs(values: dict[str, float], *, limit: int) -> str:
    items = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)
    rendered = [f"{name}={value:+.2f}" for name, value in items[:limit] if abs(value) >= 0.05]
    return "，".join(rendered) if rendered else "-"


def _truncate_block(text: str, limit: int = 1400) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "-"
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "\n……<已截断>"


def _truncate_inline(text: str, limit: int = 46) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _section(title: str, body: str | Sequence[str]) -> str:
    rendered = body if isinstance(body, str) else "\n".join(body)
    rendered = rendered.strip() or "-"
    return f"【{title}】\n{rendered}"


def _entry_kind(title: str) -> str:
    if title.startswith("你"):
        return "user"
    if title.startswith("Aurora"):
        return "assistant"
    return "system"


def _turn_to_record(
    *,
    turn_no: int,
    user_text: str,
    result: ChatTurnResult,
    live_identity: dict[str, Any],
    live_summary: dict[str, Any],
    live_stats: dict[str, Any],
) -> TurnRecord:
    return TurnRecord(
        turn_no=turn_no,
        user_text=user_text,
        result=result,
        live_identity=live_identity,
        live_summary=live_summary,
        live_stats=live_stats,
    )


def build_turn_overview_block(record: TurnRecord) -> str:
    summary = record.result.memory_context.narrative_summary
    salient_axes = summary.salient_axes if summary is not None else []
    trace = record.result.retrieval_trace_summary
    lines = [
        f"回合：第 {record.turn_no:02d} 轮",
        f"用户输入：{record.user_text}",
        f"事件 ID：{record.result.event_id}",
        f"当前模式：{record.result.memory_context.mode}",
        f"叙事压力：{record.result.memory_context.narrative_pressure:.3f}",
        f"显著轴：{'，'.join(salient_axes[:6]) if salient_axes else '-'}",
        "耗时："
        f"检索 {record.result.timings.retrieval_ms:.1f}ms | "
        f"生成 {record.result.timings.generation_ms:.1f}ms | "
        f"接收 {record.result.timings.persist_ms:.1f}ms | "
        f"总计 {record.result.timings.total_ms:.1f}ms",
        f"检索路径：长度 {trace.attractor_path_len} | 命中 {trace.hit_count} | 类型 {trace.query_type or '-'}",
    ]
    if trace.ranked_kinds:
        lines.append(f"命中类型：{'，'.join(trace.ranked_kinds[:6])}")
    if record.result.llm_error:
        lines.append(f"LLM 异常：{record.result.llm_error}")
    return "\n\n".join(
        [
            _section("回合总览", lines),
            _section(
                "Aurora 回复",
                _truncate_block(messages_to_text((record.result.reply_message,)), limit=1200),
            ),
        ]
    )


def build_prompt_status_block(result: ChatTurnResult) -> str:
    return "\n\n".join(
        [
            _section("送入 LLM 的结构化记忆简报", _truncate_block(result.rendered_memory_brief)),
            _section(
                "Aurora 回复载荷",
                _truncate_block(messages_to_text((result.reply_message,), include_image_uris=True), limit=2400),
            ),
        ]
    )


def build_turn_retrieval_block(record: TurnRecord) -> str:
    trace = record.result.retrieval_trace_summary
    context = record.result.memory_context
    trace_lines = [
        f"查询：{trace.query}",
        f"吸引子路径长度：{trace.attractor_path_len}",
        f"命中数：{trace.hit_count}",
        f"查询类型：{trace.query_type or '-'}",
        f"时间关系：{trace.time_relation or '-'}",
        f"排序类型：{'，'.join(trace.ranked_kinds[:8]) if trace.ranked_kinds else '-'}",
    ]
    hit_lines = [
        f"{idx}. {item}" for idx, item in enumerate(context.retrieval_hits[:10], start=1)
    ] or ["-"]
    evidence_lines = [
        f"{idx}. [{ref.kind}] {ref.role} | {ref.id} | score {ref.score:.3f}"
        for idx, ref in enumerate(context.evidence_refs[:10], start=1)
    ] or ["-"]
    return "\n\n".join(
        [
            _section("检索轨迹", trace_lines),
            _section("检索命中", hit_lines),
            _section("证据引用", evidence_lines),
        ]
    )


def build_context_status_block(context: StructuredMemoryContext, rendered_memory_brief: str) -> str:
    identity = context.identity
    summary = context.narrative_summary
    context_lines = [
        f"模式：{context.mode}",
        f"压力：{context.narrative_pressure:.3f}",
        f"系统直觉：{'，'.join(context.intuition[:6]) if context.intuition else '-'}",
        f"检索命中：{len(context.retrieval_hits)}",
    ]
    if summary is not None:
        context_lines.append(f"叙事摘要：{summary.text}")
        context_lines.append(
            f"显著轴：{'，'.join(summary.salient_axes[:6]) if summary.salient_axes else '-'}"
        )
    if identity is not None:
        context_lines.append(f"主导轴：{_format_axis_pairs(identity.axis_state, limit=8)}")
        context_lines.append(f"直觉轴：{_format_axis_pairs(identity.intuition_axes, limit=6)}")
    return "\n\n".join(
        [
            _section("本轮记忆上下文", context_lines),
            _section(
                "渲染给回复模型的记忆摘要", _truncate_block(rendered_memory_brief, limit=2400)
            ),
        ]
    )


def build_turn_writeback_block(record: TurnRecord) -> str:
    receipt = record.result.persistence
    lines = [
        f"事件：{receipt.event_id}",
        f"任务：{receipt.job_id}",
        f"接收状态：{receipt.status}",
        f"投影状态：{receipt.projection_status}",
        f"接收时间：{receipt.accepted_at:.3f}",
    ]
    live_lines = [
        f"当前模式：{record.live_identity['current_mode']}",
        f"压力：{record.live_summary['pressure']:.3f}",
        f"主导轴：{_format_axis_pairs(record.live_identity['axis_state'], limit=6)}",
        f"直觉轴：{_format_axis_pairs(record.live_identity['intuition_axes'], limit=4)}",
        f"能量：活跃 {record.live_identity['active_energy']:.3f} / 压抑 {record.live_identity['repressed_energy']:.3f}",
        "计数："
        f"修复 {record.live_identity['repair_count']} | "
        f"梦整合 {record.live_identity['dream_count']} | "
        f"模式变化 {record.live_identity['mode_change_count']}",
        "语料："
        f"plot {record.live_stats['plot_count']} | "
        f"summary {record.live_stats.get('summary_count', 0)} | "
        f"story {record.live_stats['story_count']} | "
        f"theme {record.live_stats['theme_count']}",
    ]
    return "\n\n".join(
        [
            _section("写回结果", lines),
            _section("写回后的实时状态", live_lines),
        ]
    )


def build_turn_status_block(
    *,
    turn_no: int,
    user_text: str,
    result: ChatTurnResult,
    live_identity: dict[str, Any],
    live_summary: dict[str, Any],
    live_stats: dict[str, Any],
) -> str:
    record = _turn_to_record(
        turn_no=turn_no,
        user_text=user_text,
        result=result,
        live_identity=live_identity,
        live_summary=live_summary,
        live_stats=live_stats,
    )
    return "\n\n".join(
        [
            build_turn_overview_block(record),
            build_prompt_status_block(result),
            build_context_status_block(result.memory_context, result.rendered_memory_brief),
            build_turn_retrieval_block(record),
            build_turn_writeback_block(record),
        ]
    )


def build_live_state_block(
    *,
    session_id: str,
    data_dir: str,
    turn_count: int,
    busy: bool,
    report: dict[str, Any],
    stats: dict[str, Any],
    selected_turn: str,
    selected_view: str,
) -> str:
    identity = report["identity"]
    summary = report["narrative_summary"]
    status = "处理中" if busy else "就绪"
    lines = [
        "概况",
        f"模式 {identity['current_mode']}",
        f"状态 {status}",
        f"轮次 {turn_count}",
        f"查看 {selected_turn} · {selected_view}",
        "",
        "轴系",
        f"主导 {_format_axis_pairs(identity['axis_state'], limit=4)}",
        f"直觉 {_format_axis_pairs(identity['intuition_axes'], limit=3)}",
        f"显著 {'，'.join(summary['salient_axes'][:4]) if summary['salient_axes'] else '-'}",
        "",
        "能量",
        f"压力 {summary['pressure']:.3f}",
        f"活跃 {identity['active_energy']:.3f}",
        f"压抑 {identity['repressed_energy']:.3f}",
        "",
        "记忆",
        f"plot {stats['plot_count']} · story {stats['story_count']}",
        f"summary {stats.get('summary_count', 0)} · theme {stats['theme_count']}",
        f"修复 {identity['repair_count']}",
        f"梦整合 {identity['dream_count']} · 模式变化 {identity['mode_change_count']}",
        "",
        f"目录 {os.path.basename(data_dir.rstrip(os.sep)) or data_dir}",
        f"会话 {session_id}",
    ]
    return "\n".join(lines)


def build_turn_index_block(
    records: Sequence[TurnRecord], *, selected_index: int, limit: int = MAX_VISIBLE_TURNS
) -> str:
    if not records:
        return "还没有回合。\n发出第一句话后，这里会出现最近对话。"
    start = max(0, len(records) - limit)
    lines: list[str] = []
    for index in range(start, len(records)):
        record = records[index]
        marker = "›" if index == selected_index else " "
        mode = record.result.memory_context.mode
        lines.append(f"{marker} 第 {record.turn_no:02d} 轮")
        lines.append(f"{_truncate_inline(record.user_text, limit=30)}")
        lines.append(
            f"模式 {mode} · 压力 {record.result.memory_context.narrative_pressure:.2f} · "
            f"命中 {len(record.result.memory_context.retrieval_hits)}"
        )
        if index != len(records) - 1:
            lines.append("")
    return "\n".join(lines)


def build_identity_status_block(report: dict[str, Any]) -> str:
    identity = report["identity"]
    summary = report["narrative_summary"]
    lines = [
        f"当前模式：{identity['current_mode']}",
        f"压力：{summary['pressure']:.3f}",
        f"摘要：{summary['text']}",
        f"主导轴：{_format_axis_pairs(identity['axis_state'], limit=8)}",
        f"直觉轴：{_format_axis_pairs(identity['intuition_axes'], limit=6)}",
        f"人格轴：{len(identity['persona_axes'])} | 别名：{len(identity['axis_aliases'])} | 模式簇：{len(identity['modes'])}",
        f"能量：活跃 {identity['active_energy']:.3f} / 压抑 {identity['repressed_energy']:.3f}",
        f"计数：修复 {identity['repair_count']} | 梦整合 {identity['dream_count']} | 模式变化 {identity['mode_change_count']}",
    ]
    if summary["salient_axes"]:
        lines.append(f"显著轴：{'，'.join(summary['salient_axes'][:6])}")
    if identity["narrative_tail"]:
        lines.extend(["", "叙事尾迹："])
        lines.extend(f"- {item}" for item in identity["narrative_tail"][-6:])
    return _section("身份快照", lines)


def build_stats_status_block(stats: dict[str, Any]) -> str:
    graph_metrics = stats.get("graph_metrics", {})
    lines = [
        f"架构模式：{stats.get('architecture_mode', '-')}",
        f"当前模式：{stats['current_mode']}",
        f"压力：{stats['pressure']:.3f}",
        f"语料规模：plot {stats['plot_count']} | summary {stats.get('summary_count', 0)} | story {stats['story_count']} | theme {stats['theme_count']}",
        f"修复：{stats['repair_count']} | 梦整合：{stats['dream_count']}",
        f"能量：活跃 {stats['active_energy']:.3f} / 压抑 {stats['repressed_energy']:.3f}",
        f"队列：{stats.get('queue_depth', 0)} | 最老待处理 {stats.get('oldest_pending_age_s') or '-'}s",
    ]
    if graph_metrics:
        authoritative = graph_metrics.get("authoritative", {})
        if authoritative:
            lines.append(
                "图视图："
                f"edge v{authoritative.get('graph_edge_version', 0)} | "
                f"{'fresh' if authoritative.get('view_fresh') else 'stale'}"
            )
    return _section("统计面板", lines)


def build_query_status_block(result: QueryResult) -> str:
    lines = [
        f"查询：{result.query}",
        f"吸引子路径长度：{result.attractor_path_len}",
        f"命中数：{len(result.hits)}",
        "",
    ]
    for idx, hit in enumerate(result.hits[:10], start=1):
        lines.append(f"{idx}. [{hit.kind}] {hit.score:.3f} {hit.snippet}")
    return _section("检索观察", lines)


def build_help_block() -> str:
    lines = [
        "直接输入内容即可与 Aurora 对话。",
        "",
        "快捷键：",
        "- Enter 发送",
        "- Ctrl-U / Ctrl-D 滚动聊天区",
        "- Tab / Shift-Tab 切换右侧页签",
        "- PgUp / PgDn 切换最近回合",
        "- Ctrl-L 清屏",
        "- Ctrl-Q 退出",
        "",
        "命令：",
        "- /帮助",
        "- /身份",
        "- /状态",
        "- /提示",
        "- /上下文",
        "- /query <文本>",
        "- /events [n] /plots [n] /stories [n] /themes [n]",
        "- /tab <概览|提示词|检索|记忆|写回>",
        "- /turn <轮次>",
    ]
    return _section("使用说明", lines)


def build_idle_block() -> str:
    lines = [
        "Aurora 已准备好。",
        "直接输入一句话，我们就开始。",
        "右侧会同步显示本轮提示词、检索命中和记忆写回。",
        "需要帮助时输入 /帮助。",
    ]
    return _section("准备就绪", lines)


class ChatBubble(Static):
    DEFAULT_CSS = """
    ChatBubble {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    ChatBubble.kind-assistant {
        padding-right: 18;
    }

    ChatBubble.kind-user {
        padding-left: 18;
    }

    ChatBubble.kind-system {
        padding-left: 10;
        padding-right: 10;
    }
    """

    def __init__(self, entry: ChatEntry, *, message_id: str) -> None:
        super().__init__("", id=message_id)
        self.entry = entry
        self._apply_kind_class()

    def on_mount(self) -> None:
        self.update(self._render_entry())

    def set_entry(self, entry: ChatEntry) -> None:
        self.entry = entry
        self._apply_kind_class()
        self.update(self._render_entry())

    def append_body(self, chunk: str) -> None:
        self.entry = ChatEntry(title=self.entry.title, body=self.entry.body + chunk)
        self.update(self._render_entry())

    def _apply_kind_class(self) -> None:
        self.remove_class("kind-assistant")
        self.remove_class("kind-user")
        self.remove_class("kind-system")
        self.add_class(f"kind-{_entry_kind(self.entry.title)}")

    def _render_entry(self) -> Any:
        kind = _entry_kind(self.entry.title)
        meta_style = {
            "assistant": "bold #0c1118 on #d4b173",
            "user": "bold #f5f9ff on #2d76f9",
            "system": "bold #d7e1ec on #283242",
        }[kind]
        body_style = {
            "assistant": "#eef4fb on #101a27",
            "user": "#f8fbff on #2257bf",
            "system": "#d8e0e9 on #17202c",
        }[kind]
        border_style = {
            "assistant": "#2a3e57",
            "user": "#4a89f7",
            "system": "#334055",
        }[kind]
        title_label = {
            "assistant": " Aurora ",
            "user": " 你 ",
            "system": " 系统 ",
        }[kind]
        title = Text(title_label, style=meta_style)
        body = Text(self.entry.body.strip() or "-", style=body_style)
        body.no_wrap = False
        panel = Panel.fit(body, box=box.ROUNDED, padding=(1, 1), border_style=border_style)
        renderable = Group(title, panel)
        if kind == "user":
            return Align.right(renderable)
        if kind == "system":
            return Align.center(renderable)
        return Align.left(renderable)


class AuroraTerminalTUI(App[None]):
    CSS = """
    Screen {
        background: #060b12;
        color: #eef4fb;
    }

    #topbar {
        dock: top;
        height: 4;
        padding: 1 2 0 2;
        background: #060b12;
        border-bottom: solid #142132;
        color: #eef4fb;
    }

    #statusbar {
        dock: bottom;
        height: 2;
        padding: 0 2;
        background: #060b12;
        border-top: solid #101927;
        color: #8090a5;
    }

    #workspace {
        height: 1fr;
        padding: 1 2 2 2;
    }

    #chat-column {
        width: 7fr;
        min-width: 72;
        margin-right: 1;
        border: round #1a2a40;
        background: #0b131d;
    }

    #sidebar {
        width: 32;
        min-width: 30;
        max-width: 36;
        border: round #172538;
        background: #0a1018;
    }

    #chat-header {
        height: 4;
        padding: 1 3 0 3;
        background: #0e1622;
        border-bottom: solid #132031;
        color: #eef4fb;
    }

    #chat-scroll {
        height: 1fr;
        padding: 1 3 1 3;
        scrollbar-gutter: stable;
    }

    #composer-shell {
        height: 5;
        margin: 0 2 2 2;
        padding: 0 1 1 1;
        border: round #23405f;
        background: #0f1825;
    }

    #chat-input {
        margin: 0;
        border: none;
        background: #131f2f;
        color: #f0f5fb;
    }

    #chat-input:focus {
        border: none;
        background: #16263a;
    }

    #composer-title,
    #sidebar-header,
    .card-title {
        height: 1;
        padding: 0 1;
        color: #9db1c8;
        text-style: bold;
    }

    #composer-title {
        color: #efc98a;
    }

    #sidebar-header {
        height: 4;
        padding: 1 2 0 2;
        background: #0d1520;
        border-bottom: solid #122031;
    }

    .sidebar-card {
        margin: 1 1 0 1;
        border: round #1e3148;
        background: #0f1723;
    }

    #live-card {
        height: 18;
    }

    #turns-card {
        height: 11;
    }

    #inspector-card {
        height: 1fr;
        min-height: 18;
    }

    #live-state,
    #turn-list,
    #inspector-header,
    #inspector-body {
        padding: 0 1 1 1;
        color: #dbe5ef;
    }

    #turn-list {
        color: #d4deea;
    }

    #inspector-tabs {
        margin: 0 1 1 1;
    }

    #inspector-scroll {
        height: 1fr;
        scrollbar-gutter: stable;
    }

    #inspector-body {
        width: 100%;
    }

    Tabs {
        height: 1;
    }

    Tab {
        padding: 0 1;
        margin-right: 1;
        color: #9badc2;
        background: #172331;
        border: none;
    }

    Tab.-active {
        background: #d6b07a;
        color: #08111c;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", show=False, priority=True),
        Binding("ctrl+l", "clear_transcript", show=False, priority=True),
        Binding("ctrl+u", "scroll_chat_up", show=False, priority=True),
        Binding("ctrl+d", "scroll_chat_down", show=False, priority=True),
        Binding("tab", "next_tab", show=False, priority=True),
        Binding("shift+tab", "previous_tab", show=False, priority=True),
        Binding("pageup", "previous_turn", show=False, priority=True),
        Binding("pagedown", "next_turn", show=False, priority=True),
    ]

    def __init__(
        self,
        runtime: AuroraRuntime,
        *,
        session_id: str,
        max_hits: int = 6,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.session_id = session_id
        self.max_hits = max_hits
        self.turn_count = 0
        self.last_result: Optional[ChatTurnResult] = None
        self._busy = False
        self._chat_entries: List[ChatEntry] = []
        self._chat_ids: List[str] = []
        self._chat_widgets: dict[str, ChatBubble] = {}
        self._turn_records: List[TurnRecord] = []
        self._selected_turn_index = -1
        self._selected_tab = "overview"
        self._utility_panel: Optional[StatusRecord] = None
        self._live_state_text = ""
        self._footer_text = "Enter 发送 · Tab 页签 · PgUp/PgDn 回合 · Ctrl-Q 退出"
        self._cached_report: Optional[dict[str, Any]] = None
        self._cached_stats: Optional[dict[str, Any]] = None
        self._message_counter = 0
        self._mounted = False
        self._seed_initial_state()

    def compose(self) -> ComposeResult:
        yield Static("", id="topbar")
        with Horizontal(id="workspace"):
            with Vertical(id="chat-column"):
                yield Static("", id="chat-header")
                yield VerticalScroll(id="chat-scroll")
                with Vertical(id="composer-shell"):
                    yield Static("输入", id="composer-title")
                    yield Input(placeholder="直接输入消息，回车发送", id="chat-input")

            with Vertical(id="sidebar"):
                yield Static("", id="sidebar-header")
                with Vertical(id="live-card", classes="sidebar-card"):
                    yield Static("实时状态", classes="card-title")
                    yield Static("", id="live-state")
                with Vertical(id="turns-card", classes="sidebar-card"):
                    yield Static("最近回合", classes="card-title")
                    yield Static("", id="turn-list")
                with Vertical(id="inspector-card", classes="sidebar-card"):
                    yield Static("检查器", classes="card-title")
                    yield Tabs(
                        *[Tab(label, id=f"tab-{key}") for key, label in INSPECTOR_TABS],
                        active="tab-overview",
                        id="inspector-tabs",
                    )
                    yield Static("", id="inspector-header")
                    with VerticalScroll(id="inspector-scroll"):
                        yield Static("", id="inspector-body")
        yield Static("", id="statusbar")

    async def on_mount(self) -> None:
        self._mounted = True
        self._sync_chat_widgets()
        self._refresh_live_state()
        self._refresh_view()
        self.query_one("#chat-input", Input).focus()

    def _next_message_id(self) -> str:
        self._message_counter += 1
        return f"msg-{self._message_counter}"

    def _header_text(self) -> str:
        mode = "未建立"
        if self._cached_report is not None:
            mode = self._cached_report["identity"]["current_mode"]
        state = "处理中" if self._busy else "就绪"
        if self.session_id == "terminal_observer":
            return f"Aurora · {mode} · {state}"
        return f"Aurora · {mode} · {state} · {self.session_id}"

    def _render_topbar(self) -> Any:
        mode = "未建立"
        if self._cached_report is not None:
            mode = self._cached_report["identity"]["current_mode"]
        state = "处理中" if self._busy else "就绪"
        line = Text()
        line.append(" Aurora ", style="bold #08111a on #d5b276")
        line.append("  Soul Terminal  ", style="bold #eef4fb")
        line.append(f" 模式 {mode} ", style="bold #d8e3ef on #152435")
        line.append(" ")
        line.append(f" 状态 {state} ", style="bold #d8e3ef on #16283d")
        if self.session_id != "terminal_observer":
            line.append(" ")
            line.append(f" 会话 {self.session_id} ", style="bold #d8e3ef on #13202f")
        return line

    def _render_chat_header(self) -> Any:
        subtitle = "流式回复已开启，右侧同步展示提示词、检索与写回"
        if self.turn_count > 0:
            subtitle = f"已完成 {self.turn_count} 轮对话，当前回复会持续流式写入"
        return Group(
            Text("会话", style="bold #f4efe4"),
            Text(subtitle, style="#8fa3ba"),
        )

    def _render_sidebar_header(self) -> Any:
        active = self._inspector_title()
        return Group(
            Text("状态侧栏", style="bold #f4efe4"),
            Text(f"当前焦点：{active}", style="#8fa3ba"),
        )

    def _render_statusbar(self) -> Any:
        if self._busy:
            return Text(self._footer_text, style="bold #f3cf92")
        return Text(self._footer_text, style="#8090a5")

    def _inspector_title(self) -> str:
        if self._utility_panel is not None:
            return self._utility_panel.title
        return f"{self._selected_turn_label()} / {self._selected_tab_label()}"

    def _selected_turn_label(self) -> str:
        if self._utility_panel is not None:
            return "工具"
        if self._selected_turn_index < 0 or self._selected_turn_index >= len(self._turn_records):
            return "未选回合"
        return f"第 {self._turn_records[self._selected_turn_index].turn_no:02d} 轮"

    def _selected_tab_label(self) -> str:
        if self._utility_panel is not None:
            return self._utility_panel.title
        return TAB_LABELS.get(self._selected_tab, self._selected_tab)

    def _seed_initial_state(self) -> None:
        self._chat_entries = []
        self._chat_ids = []
        self._chat_widgets = {}
        self._append_chat_entry(
            "Aurora",
            "欢迎回来。\n直接告诉我你现在想聊什么，我会边回应边整理右侧的提示词、检索和记忆状态。",
        )
        self._utility_panel = StatusRecord(title="准备就绪", body=build_idle_block())
        self._refresh_live_state()

    def _sync_chat_widgets(self) -> None:
        if not self._mounted:
            return
        chat_view = self.query_one("#chat-scroll", VerticalScroll)
        chat_view.remove_children()
        self._chat_widgets = {}
        for message_id, entry in zip(self._chat_ids, self._chat_entries):
            widget = ChatBubble(entry, message_id=message_id)
            self._chat_widgets[message_id] = widget
            chat_view.mount(widget)
        chat_view.scroll_end(animate=False)

    def _append_chat_entry(self, title: str, body: str) -> str:
        entry = ChatEntry(title=title, body=body)
        message_id = self._next_message_id()
        self._chat_entries.append(entry)
        self._chat_ids.append(message_id)
        while len(self._chat_entries) > MAX_STORED_CHAT_ENTRIES:
            self._chat_entries.pop(0)
            old_id = self._chat_ids.pop(0)
            widget = self._chat_widgets.pop(old_id, None)
            if widget is not None:
                widget.remove()
        if self._mounted:
            widget = ChatBubble(entry, message_id=message_id)
            self._chat_widgets[message_id] = widget
            self.query_one("#chat-scroll", VerticalScroll).mount(widget)
            self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)
        return message_id

    def _replace_chat_entry(self, message_id: str, title: str, body: str) -> None:
        if message_id in self._chat_ids:
            index = self._chat_ids.index(message_id)
            entry = ChatEntry(title=title, body=body)
            self._chat_entries[index] = entry
            widget = self._chat_widgets.get(message_id)
            if widget is not None:
                widget.set_entry(entry)
            return
        self._append_chat_entry(title, body)

    def _append_chat_chunk(self, message_id: str, chunk: str) -> None:
        if message_id not in self._chat_ids:
            return
        index = self._chat_ids.index(message_id)
        current = self._chat_entries[index]
        updated = ChatEntry(title=current.title, body=current.body + chunk)
        self._chat_entries[index] = updated
        widget = self._chat_widgets.get(message_id)
        if widget is not None:
            widget.append_body(chunk)
            self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)

    def _refresh_live_state(
        self,
        *,
        report: Optional[dict[str, Any]] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        if report is not None:
            self._cached_report = report
        elif self._cached_report is None:
            self._cached_report = self.runtime.get_identity()

        if stats is not None:
            self._cached_stats = stats
        elif self._cached_stats is None:
            self._cached_stats = self.runtime.get_stats()

        self._live_state_text = build_live_state_block(
            session_id=self.session_id,
            data_dir=str(self.runtime.settings.data_dir),
            turn_count=self.turn_count,
            busy=self._busy,
            report=self._cached_report,
            stats=self._cached_stats,
            selected_turn=self._selected_turn_label(),
            selected_view=self._selected_tab_label(),
        )

    def _render_turn_index_text(self) -> str:
        return build_turn_index_block(self._turn_records, selected_index=self._selected_turn_index)

    def _render_inspector_text(self) -> str:
        if self._utility_panel is not None:
            return self._utility_panel.body
        if self._selected_turn_index < 0 or self._selected_turn_index >= len(self._turn_records):
            return build_idle_block()
        record = self._turn_records[self._selected_turn_index]
        if self._selected_tab == "overview":
            return build_turn_overview_block(record)
        if self._selected_tab == "prompt":
            return build_prompt_status_block(record.result)
        if self._selected_tab == "retrieval":
            return build_turn_retrieval_block(record)
        if self._selected_tab == "memory":
            return build_context_status_block(
                record.result.memory_context, record.result.rendered_memory_brief
            )
        return build_turn_writeback_block(record)

    def _refresh_view(self) -> None:
        if not self._mounted:
            return
        self.query_one("#topbar", Static).update(self._render_topbar())
        self.query_one("#statusbar", Static).update(self._render_statusbar())
        self.query_one("#chat-header", Static).update(self._render_chat_header())
        self.query_one("#sidebar-header", Static).update(self._render_sidebar_header())
        self.query_one("#live-state", Static).update(self._live_state_text)
        self.query_one("#turn-list", Static).update(self._render_turn_index_text())
        self.query_one("#inspector-header", Static).update(self._inspector_title())
        self.query_one("#inspector-body", Static).update(self._render_inspector_text())
        tabs = self.query_one("#inspector-tabs", Tabs)
        desired = f"tab-{self._selected_tab}"
        if tabs.active != desired:
            tabs.active = desired

    async def _handle_command(self, text: str) -> None:
        parts = text[1:].strip().split()
        name = parts[0].lower() if parts else "help"
        name = COMMAND_ALIASES.get(name, name)
        args = parts[1:] if parts else []

        if name in {"quit", "exit"}:
            self.exit()
            return
        if name == "clear":
            self.action_clear_transcript()
            return
        if name == "help":
            self._utility_panel = StatusRecord(title="帮助", body=build_help_block())
            self._refresh_view()
            return
        if name == "identity":
            report = await asyncio.to_thread(self.runtime.get_identity)
            self._utility_panel = StatusRecord(
                title="身份快照", body=build_identity_status_block(report)
            )
            self._refresh_live_state(report=report)
            self._refresh_view()
            return
        if name == "stats":
            stats = await asyncio.to_thread(self.runtime.get_stats)
            self._utility_panel = StatusRecord(
                title="统计面板", body=build_stats_status_block(stats)
            )
            self._refresh_live_state(stats=stats)
            self._refresh_view()
            return
        if name in {"prompt", "memory"}:
            self._select_tab("prompt" if name == "prompt" else "memory")
            return
        if name == "tab":
            if not args:
                self._utility_panel = StatusRecord(title="帮助", body=build_help_block())
                self._refresh_view()
                return
            tab_key = TAB_LOOKUP.get(args[0].lower())
            if tab_key is None:
                self._utility_panel = StatusRecord(
                    title="命令提示", body=_section("无效页签", f"不支持的页签：{args[0]}")
                )
                self._refresh_view()
                return
            self._select_tab(tab_key)
            return
        if name == "turn":
            if not args:
                self._utility_panel = StatusRecord(
                    title="命令提示", body=_section("用法", "/turn <轮次>")
                )
                self._refresh_view()
                return
            try:
                self._select_turn_by_no(int(args[0]))
            except ValueError:
                self._utility_panel = StatusRecord(
                    title="命令提示", body=_section("用法", "/turn <轮次>")
                )
                self._refresh_view()
            return
        if name == "query":
            if not args:
                self._utility_panel = StatusRecord(
                    title="命令提示", body=_section("用法", "/query <文本>")
                )
                self._refresh_view()
                return
            query_text = " ".join(args)
            result = await asyncio.to_thread(
                lambda: self.runtime.query(
                    messages=[Message(role="user", parts=(TextPart(text=query_text),))],
                    k=self.max_hits,
                )
            )
            self._utility_panel = StatusRecord(
                title="检索观察", body=build_query_status_block(result)
            )
            self._refresh_view()
            return
        if name in {"events", "plots", "stories", "themes"}:
            limit = 6
            if args:
                try:
                    limit = max(1, int(args[0]))
                except Exception:
                    limit = 6
            self._utility_panel = StatusRecord(
                title=self._listing_title(name),
                body=_section(
                    self._listing_title(name), self._render_listing(name=name, limit=limit)
                ),
            )
            self._refresh_view()
            return

        self._utility_panel = StatusRecord(
            title="命令提示",
            body=_section("未知命令", f"未识别：/{name}\n输入 /帮助 查看命令。"),
        )
        self._refresh_view()

    def _listing_title(self, name: str) -> str:
        return {
            "events": "最近事件",
            "plots": "最近 Plot",
            "stories": "最近 Story",
            "themes": "最近 Theme",
        }[name]

    def _render_listing(self, *, name: str, limit: int) -> str:
        if name == "events":
            lines = []
            for event in self.runtime.recent_events(limit=limit):
                lines.append(f"{event.seq}. {event.event_id} | 会话 {event.session_id} | 时间 {event.ts:.0f}")
                lines.append(f"   内容：{str(event.payload.get('search_text', ''))[:120]}")
            return "\n".join(lines or ["暂无事件"])
        if name == "plots":
            plots = sorted(self.runtime.mem.plots.values(), key=lambda plot: plot.ts, reverse=True)[
                :limit
            ]
            return (
                "\n".join(
                    f"{plot.id[:8]} | {SOURCE_LABELS.get(plot.source, plot.source)} | story {plot.story_id or '-'} | 张力 {plot.tension:.3f}\n  {plot.semantic_text[:140]}"
                    for plot in plots
                )
                or "暂无 plot"
            )
        if name == "stories":
            stories = sorted(
                self.runtime.mem.stories.values(), key=lambda story: story.updated_ts, reverse=True
            )[:limit]
            return (
                "\n".join(
                    f"{story.id[:8]} | 状态 {story.status} | plot {len(story.plot_ids)} | 未解能量 {story.unresolved_energy:.3f}"
                    for story in stories
                )
                or "暂无 story"
            )
        themes = sorted(
            self.runtime.mem.themes.values(), key=lambda theme: theme.updated_ts, reverse=True
        )[:limit]
        return (
            "\n".join(
                f"{theme.id[:8]} | 置信 {theme.confidence():.2f} | {(theme.name or theme.description)[:120]}"
                for theme in themes
            )
            or "暂无 theme"
        )

    def _select_tab(self, key: str) -> None:
        if key not in TAB_LABELS:
            return
        self._selected_tab = key
        self._utility_panel = None
        self._refresh_live_state()
        self._refresh_view()

    def _cycle_tab(self, delta: int) -> None:
        tabs = [key for key, _ in INSPECTOR_TABS]
        current = tabs.index(self._selected_tab)
        self._select_tab(tabs[(current + delta) % len(tabs)])

    def _move_turn(self, delta: int) -> None:
        if not self._turn_records:
            self._utility_panel = StatusRecord(title="帮助", body=build_help_block())
            self._refresh_view()
            return
        self._utility_panel = None
        self._selected_turn_index = max(
            0, min(len(self._turn_records) - 1, self._selected_turn_index + delta)
        )
        self._refresh_live_state()
        self._refresh_view()

    def _select_turn_by_no(self, turn_no: int) -> None:
        for index, record in enumerate(self._turn_records):
            if record.turn_no == turn_no:
                self._selected_turn_index = index
                self._utility_panel = None
                self._refresh_live_state()
                self._refresh_view()
                return
        self._utility_panel = StatusRecord(
            title="命令提示", body=_section("未找到回合", f"没有第 {turn_no} 轮。")
        )
        self._refresh_view()

    @on(Input.Submitted, "#chat-input")
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if self._busy:
            self._footer_text = "Aurora 正在处理上一条输入，请稍等 · Ctrl-Q 退出"
            self._refresh_view()
            return
        if text.startswith("/"):
            await self._handle_command(text)
            return

        self.turn_count += 1
        self._busy = True
        self._utility_panel = None
        self._footer_text = "Aurora 正在准备检索记忆…"
        self._append_chat_entry("你", text)
        placeholder_id = self._append_chat_entry("Aurora", "正在组织记忆与回复……")
        self._refresh_live_state()
        self._refresh_view()
        self._run_turn(user_text=text, placeholder_id=placeholder_id)

    @on(Tabs.TabActivated, "#inspector-tabs")
    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab.id is None or not event.tab.id.startswith("tab-"):
            return
        self._selected_tab = event.tab.id.removeprefix("tab-")
        if self._utility_panel is not None:
            self._utility_panel = None
        self._refresh_live_state()
        self._refresh_view()

    @work(thread=True, exclusive=True)
    def _run_turn(self, *, user_text: str, placeholder_id: str) -> None:
        try:
            final_result: Optional[ChatTurnResult] = None
            for event in self.runtime.respond_stream(
                session_id=self.session_id,
                user_messages=[Message(role="user", parts=(TextPart(text=user_text),))],
                k=self.max_hits,
            ):
                if event.kind == "status":
                    self.call_from_thread(self._apply_stream_status, event)
                elif event.kind == "reply_delta":
                    self.call_from_thread(self._append_chat_chunk, placeholder_id, event.text)
                elif event.kind == "done":
                    final_result = event.result
            if final_result is None:
                raise RuntimeError("Aurora stream ended before returning a final result.")
            live_report = self.runtime.get_identity()
            live_stats = self.runtime.get_stats()
            self.call_from_thread(
                self._complete_turn,
                user_text,
                placeholder_id,
                final_result,
                live_report,
                live_stats,
            )
        except Exception as exc:
            self.call_from_thread(self._fail_turn, placeholder_id, str(exc))

    def _apply_stream_status(self, event: ChatStreamEvent) -> None:
        status_map = {
            "retrieval": "Aurora 正在检索记忆……",
            "generation": "Aurora 正在流式生成回复……",
            "persist_accept": "Aurora 已接收本轮记忆，后台正在整合……",
            "done": "Aurora 已完成当前回合。",
        }
        headline = status_map.get(event.stage, "Aurora 正在工作……")
        self._footer_text = f"{headline} {event.text}".strip()
        self._refresh_view()

    def _complete_turn(
        self,
        user_text: str,
        placeholder_id: str,
        result: ChatTurnResult,
        live_report: dict[str, Any],
        live_stats: dict[str, Any],
    ) -> None:
        self.last_result = result
        self._replace_chat_entry(
            placeholder_id,
            "Aurora",
            messages_to_text((result.reply_message,)),
        )
        record = _turn_to_record(
            turn_no=self.turn_count,
            user_text=user_text,
            result=result,
            live_identity=live_report["identity"],
            live_summary=live_report["narrative_summary"],
            live_stats=live_stats,
        )
        self._turn_records.append(record)
        self._turn_records = self._turn_records[-MAX_STORED_TURNS:]
        self._selected_turn_index = len(self._turn_records) - 1
        self._cached_report = live_report
        self._cached_stats = live_stats
        self._busy = False
        self._footer_text = "Enter 发送 · Tab 页签 · PgUp/PgDn 回合 · Ctrl-Q 退出"
        self._refresh_live_state()
        self._refresh_view()

    def _fail_turn(self, placeholder_id: str, error_text: str) -> None:
        self._replace_chat_entry(placeholder_id, "系统", f"工作台执行失败：{error_text}")
        self._busy = False
        self._footer_text = "Aurora 处理失败 · Ctrl-Q 退出"
        self._refresh_live_state()
        self._refresh_view()

    def action_scroll_chat_up(self) -> None:
        if self._mounted:
            self.query_one("#chat-scroll", VerticalScroll).scroll_page_up(animate=False)

    def action_scroll_chat_down(self) -> None:
        if self._mounted:
            self.query_one("#chat-scroll", VerticalScroll).scroll_page_down(animate=False)

    def action_next_tab(self) -> None:
        self._cycle_tab(1)

    def action_previous_tab(self) -> None:
        self._cycle_tab(-1)

    def action_previous_turn(self) -> None:
        self._move_turn(-1)

    def action_next_turn(self) -> None:
        self._move_turn(1)

    def action_clear_transcript(self) -> None:
        self.turn_count = 0
        self.last_result = None
        self._busy = False
        self._turn_records.clear()
        self._selected_turn_index = -1
        self._selected_tab = "overview"
        self._utility_panel = None
        self._cached_report = None
        self._cached_stats = None
        self._footer_text = "Enter 发送 · Tab 页签 · PgUp/PgDn 回合 · Ctrl-Q 退出"
        self._seed_initial_state()
        if self._mounted:
            self._sync_chat_widgets()
            self._refresh_view()
