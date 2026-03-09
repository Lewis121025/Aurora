from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from aurora.core.models.trace import RetrievalTrace
from aurora.interfaces.terminal_observer import (
    TerminalObserver,
    main,
    parse_command,
    run_observer,
)
from aurora.runtime.results import (
    ChatTimings,
    ChatTurnResult,
    EvidenceRef,
    IngestResult,
    RetrievalTraceSummary,
    StructuredMemoryContext,
)


class FakeMem:
    def __init__(self, trace: RetrievalTrace):
        self._trace = trace
        self.plots = {
            "plot-1": SimpleNamespace(
                id="plot-1",
                text="USER: 我喜欢长期记忆系统\nAGENT: 我会记住这一点",
                story_id="story-1",
                status="active",
                actors=("user", "agent"),
                ts=123.0,
                knowledge_type="preference",
                redundancy_type="novel",
                supersedes_id=None,
                superseded_by_id=None,
                tension=0.4,
                surprise=0.2,
                pred_error=0.1,
                redundancy=0.05,
            )
        }
        self.stories = {
            "story-1": SimpleNamespace(
                id="story-1",
                updated_ts=100.0,
                plot_ids=["plot-1"],
                status="developing",
                relationship_with="user",
                relationship_type="user",
                my_identity_in_this_relationship="长期记忆助手",
                central_conflict=None,
                moral=None,
            )
        }
        self.themes = {}

    def query_with_timeline(self, text, k=6, asker_id="user"):
        return self._trace


class FakeDocStore:
    def get(self, doc_id):
        if doc_id == "plot-1":
            return SimpleNamespace(
                id="plot-1",
                kind="plot",
                ts=123.0,
                body={
                    "resolved_actors": ["user", "agent"],
                    "action": "表达偏好",
                    "outcome": "系统记住偏好",
                    "claims": [{"subject": "user", "predicate": "prefers", "object": "长期记忆系统"}],
                    "plot_state": {
                        "text": "USER: 我喜欢长期记忆系统\nAGENT: 我会记住这一点",
                        "story_id": "story-1",
                        "status": "active",
                    },
                },
            )
        return None

    def iter_kind(self, *, kind, limit=200):
        return []


class FakeEventLog:
    def iter_events(self, *, after_seq=0):
        return []


class FakeRuntime:
    def __init__(self, trace: RetrievalTrace):
        self.settings = SimpleNamespace(
            data_dir="./test-data",
            llm_provider="bailian",
            embedding_provider="local",
            llm_timeout=5.0,
            bailian_llm_model="qwen",
            bailian_embedding_model="embed",
            ark_llm_model="ark",
        )
        self.mem = FakeMem(trace)
        self.doc_store = FakeDocStore()
        self.event_log = FakeEventLog()
        self.last_seq = 1
        self.respond_calls = []

    def respond(self, **kwargs):
        self.respond_calls.append(kwargs)
        return ChatTurnResult(
            reply="这是一个测试回复",
            event_id="evt_obs_001",
            memory_context=StructuredMemoryContext(
                known_facts=["user prefers 长期记忆系统"],
                preferences=["偏好长期记忆系统"],
                relationship_state=["关系对象=user | 状态=developing | 我的角色=长期记忆助手 | 关系健康度=0.80"],
                active_narratives=["story=story-1 | status=developing | relationship=user | identity=长期记忆助手"],
                temporal_context=[],
                system_intuition=["隐约期待"],
                cautions=[],
                evidence_refs=[EvidenceRef(id="plot-1", kind="plot", score=0.91, role="current_fact")],
            ),
            rendered_memory_brief=(
                "[Known Facts]\n- user prefers 长期记忆系统\n\n"
                "[Preferences]\n- 偏好长期记忆系统"
            ),
            system_prompt="system prompt",
            user_prompt=(
                "Memory Brief:\n[Known Facts]\n- user prefers 长期记忆系统\n\n"
                "Current User Message:\n你记得我喜欢什么吗？"
            ),
            retrieval_trace_summary=RetrievalTraceSummary(
                query="你记得我喜欢什么吗？",
                query_type="FACTUAL",
                attractor_path_len=1,
                hit_count=1,
                timeline_count=0,
                standalone_count=0,
                abstain=False,
            ),
            ingest_result=IngestResult(
                event_id="evt_obs_001",
                plot_id="plot-1",
                story_id="story-1",
                memory_layer="explicit",
                tension=0.4,
                surprise=0.2,
                pred_error=0.1,
                redundancy=0.05,
            ),
            timings=ChatTimings(
                retrieval_ms=1.0,
                generation_ms=2.0,
                ingest_ms=3.0,
                total_ms=6.0,
            ),
            llm_error=None,
        )

    def check_coherence(self):
        return SimpleNamespace(
            overall_score=1.0,
            conflict_count=0,
            unfinished_story_count=1,
            recommendations=[],
        )

    def get_self_narrative(self):
        return {
            "profile_id": "aurora-v2-child-elara",
            "identity_statement": "我是一个长期记忆助手",
            "seed_narrative": "我带着长期记忆人格底色进入对话",
            "capability_narrative": "我会记住长期上下文",
            "coherence_score": 1.0,
            "capabilities": {},
            "trait_beliefs": {},
            "relationships": {},
            "subconscious": {"dark_matter_count": 0, "repressed_count": 0, "last_intuition": []},
            "unresolved_tensions": [],
        }

    def evolve(self):
        return None


def make_trace() -> RetrievalTrace:
    trace = RetrievalTrace(
        query="你记得我喜欢什么吗？",
        query_emb=np.zeros(4, dtype=np.float32),
        attractor_path=[np.zeros(4, dtype=np.float32)],
        ranked=[("plot-1", 0.91, "plot")],
    )
    trace.query_type = SimpleNamespace(name="FACTUAL")
    trace.abstention = SimpleNamespace(should_abstain=False, reason="confident")
    trace.timeline_group = SimpleNamespace(timelines=[], standalone_results=[])
    return trace


def test_parse_command_supports_quoted_args():
    command = parse_command('/inspect "plot-1"')

    assert command is not None
    assert command.name == "inspect"
    assert command.args == ("plot-1",)


def test_default_chat_mode_renders_reply_without_observe_panels():
    runtime = FakeRuntime(make_trace())
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="s-observe", output=output)

    result = observer.run_chat_turn("你记得我喜欢什么吗？")
    observer._render_chat_result(result)

    assert result.reply == "这是一个测试回复"
    assert runtime.respond_calls[0]["session_id"] == "s-observe"
    assert runtime.respond_calls[0]["k"] == 6
    rendered = output.getvalue()
    assert "aurora · turn 01" in rendered
    assert "这是一个测试回复" in rendered
    assert "Observe · Turn 01" not in rendered
    assert "Assistant · Turn 01" not in rendered


def test_brief_mode_renders_reply_and_turn_summary():
    runtime = FakeRuntime(make_trace())
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="s-observe", observe_mode="brief", output=output)

    result = observer.run_chat_turn("你记得我喜欢什么吗？")
    observer._render_chat_result(result)

    rendered = output.getvalue()
    assert "Assistant" in rendered
    assert "Observe · Turn 01" in rendered
    assert "mode=brief" in rendered
    assert "Prompt · LLM" not in rendered
    assert "Artifacts / Evidence" not in rendered


def test_full_observe_mode_renders_debug_panels():
    runtime = FakeRuntime(make_trace())
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="s-observe", observe_mode="full", output=output)

    result = observer.run_chat_turn("你记得我喜欢什么吗？")
    observer._render_chat_result(result)

    rendered = output.getvalue()
    assert "Memory Brief" in rendered
    assert "Retrieval" in rendered
    assert "Prompt · LLM" in rendered
    assert "Artifacts / Evidence" in rendered


def test_prompt_command_renders_last_prompt():
    runtime = FakeRuntime(make_trace())
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="s-observe", output=output)

    result = observer.run_chat_turn("你记得我喜欢什么吗？")
    observer._render_chat_result(result)
    output.truncate(0)
    output.seek(0)

    command_result = observer.handle_line("/prompt")

    assert command_result is True
    rendered = output.getvalue()
    assert "Prompt · LLM" in rendered
    assert "你记得我喜欢什么吗？" in rendered


def test_context_command_renders_last_context():
    runtime = FakeRuntime(make_trace())
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="s-observe", output=output)

    result = observer.run_chat_turn("你记得我喜欢什么吗？")
    observer._render_chat_result(result)
    output.truncate(0)
    output.seek(0)

    command_result = observer.handle_line("/context")

    assert command_result is True
    rendered = output.getvalue()
    assert "Context · Memory" in rendered
    assert "known_facts" in rendered
    assert "长期记忆系统" in rendered


def test_handle_observe_command_updates_mode():
    runtime = FakeRuntime(make_trace())
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="s-observe", output=output)

    result = observer.handle_line("/observe brief")

    assert result is True
    assert observer.observe_mode == "brief"
    assert "observe mode -> brief" in output.getvalue()


def test_short_mode_command_updates_mode():
    runtime = FakeRuntime(make_trace())
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="s-observe", observe_mode="brief", output=output)

    result = observer.handle_line("/chat")

    assert result is True
    assert observer.observe_mode == "chat"
    assert "now in chat mode" in output.getvalue()


def test_input_prompt_wraps_ansi_sequences_for_readline():
    runtime = FakeRuntime(make_trace())
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="s-observe", output=output)
    observer._use_color = True

    prompt = observer._input_prompt()

    assert "\001" in prompt
    assert "\002" in prompt
    assert "you" in prompt


def test_main_uses_default_settings_when_data_dir_is_omitted(monkeypatch):
    runtime = MagicMock()
    observer = MagicMock()
    observer.run = MagicMock()

    settings_calls = []

    def fake_settings(**kwargs):
        settings_calls.append(kwargs)
        return SimpleNamespace(data_dir="./data")

    monkeypatch.setattr("aurora.interfaces.terminal_observer.AuroraSettings", fake_settings)
    monkeypatch.setattr(
        "aurora.interfaces.terminal_observer.AuroraRuntime",
        lambda settings: runtime,
    )
    monkeypatch.setattr(
        "aurora.interfaces.terminal_observer.TerminalObserver",
        lambda *args, **kwargs: observer,
    )

    main(["--observe", "brief"])

    assert settings_calls == [{}]
    observer.run.assert_called_once_with()


def test_run_observer_forwards_parameters(monkeypatch):
    runtime = MagicMock()
    observer = MagicMock()
    observer.run = MagicMock()

    settings_calls = []

    def fake_settings(**kwargs):
        settings_calls.append(kwargs)
        return SimpleNamespace(data_dir=kwargs.get("data_dir", "./data"))

    monkeypatch.setattr("aurora.interfaces.terminal_observer.AuroraSettings", fake_settings)
    monkeypatch.setattr(
        "aurora.interfaces.terminal_observer.AuroraRuntime",
        lambda settings: runtime,
    )
    monkeypatch.setattr(
        "aurora.interfaces.terminal_observer.TerminalObserver",
        lambda *args, **kwargs: observer,
    )

    run_observer(
        data_dir="./observe-data",
        session_id="s-observe",
        max_hits=9,
        observe_mode="brief",
    )

    assert settings_calls == [{"data_dir": "./observe-data"}]
    observer.run.assert_called_once_with()


def test_observe_runtime_script_runs_from_repo_root():
    project_root = Path(__file__).resolve().parents[3]
    script_path = project_root / "scripts" / "observe_runtime.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "observe_runtime" in result.stdout
