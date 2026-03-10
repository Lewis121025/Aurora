from __future__ import annotations

from types import SimpleNamespace

from aurora.interfaces.terminal.observer import run_observer
from aurora.interfaces.terminal.tui import (
    AuroraTerminalTUI,
    build_turn_index_block,
    build_turn_status_block,
)
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings


def test_build_turn_status_block_contains_prompt_and_live_state(tmp_path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        )
    )

    result = runtime.respond(session_id="tui", user_message="你记得我刚才说了什么吗？")
    report = runtime.get_identity()
    stats = runtime.get_stats()

    block = build_turn_status_block(
        turn_no=1,
        user_message="你记得我刚才说了什么吗？",
        result=result,
        live_identity=report["identity"],
        live_summary=report["narrative_summary"],
        live_stats=stats,
    )

    assert "送入 LLM 的 System Prompt" in block
    assert "送入 LLM 的 User Prompt" in block
    assert "本轮记忆上下文" in block
    assert "写回后的实时状态" in block
    assert "当前模式：" in block


def test_run_observer_uses_tui(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, *, settings):
            observed["data_dir"] = settings.data_dir
            observed["llm_timeout"] = settings.llm_timeout
            observed["llm_max_retries"] = settings.llm_max_retries

    class FakeTUI:
        def __init__(self, runtime, *, session_id, max_hits):
            observed["runtime"] = runtime
            observed["session_id"] = session_id
            observed["max_hits"] = max_hits

        def run(self) -> None:
            observed["ran_tui"] = True

    import aurora.interfaces.terminal.observer as observer

    monkeypatch.setattr(observer, "AuroraRuntime", FakeRuntime)
    monkeypatch.setattr(observer, "AuroraTerminalTUI", FakeTUI)

    run_observer(data_dir=str(tmp_path), session_id="tty-session", max_hits=9)

    assert observed["data_dir"] == str(tmp_path)
    assert observed["llm_timeout"] == 10.0
    assert observed["llm_max_retries"] == 1
    assert observed["session_id"] == "tty-session"
    assert observed["max_hits"] == 9
    assert observed["ran_tui"] is True


def test_build_turn_index_block_marks_selected_turn(tmp_path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        )
    )

    first = runtime.respond(session_id="tui", user_message="第一轮，记录一下我的语气。")
    first_report = runtime.get_identity()
    first_stats = runtime.get_stats()

    second = runtime.respond(session_id="tui", user_message="第二轮，看看索引是不是更清楚。")
    second_report = runtime.get_identity()
    second_stats = runtime.get_stats()

    from aurora.interfaces.terminal.tui import _turn_to_record

    block = build_turn_index_block(
        [
            _turn_to_record(
                turn_no=1,
                user_message="第一轮，记录一下我的语气。",
                result=first,
                live_identity=first_report["identity"],
                live_summary=first_report["narrative_summary"],
                live_stats=first_stats,
            ),
            _turn_to_record(
                turn_no=2,
                user_message="第二轮，看看索引是不是更清楚。",
                result=second,
                live_identity=second_report["identity"],
                live_summary=second_report["narrative_summary"],
                live_stats=second_stats,
            ),
        ],
        selected_index=1,
    )

    assert "› 第 02 轮" in block
    assert "第一轮，记录一下我的语气" in block
    assert "模式" in block


def test_refresh_live_state_reuses_cached_runtime_state(tmp_path) -> None:
    calls = {"identity": 0, "stats": 0}

    class FakeRuntime:
        settings = SimpleNamespace(data_dir=str(tmp_path))

        def get_identity(self):
            calls["identity"] += 1
            return {
                "identity": {
                    "current_mode": "steady",
                    "axis_state": {"clarity": 0.4},
                    "intuition_axes": {},
                    "active_energy": 0.2,
                    "repressed_energy": 0.1,
                    "repair_count": 0,
                    "dream_count": 0,
                    "mode_change_count": 0,
                    "persona_axes": [],
                    "axis_aliases": {},
                    "modes": [],
                    "narrative_tail": [],
                },
                "narrative_summary": {
                    "pressure": 0.15,
                    "salient_axes": ["clarity"],
                    "text": "steady",
                },
            }

        def get_stats(self):
            calls["stats"] += 1
            return {
                "plot_count": 0,
                "story_count": 0,
                "theme_count": 0,
                "current_mode": "steady",
                "pressure": 0.15,
                "dream_count": 0,
                "repair_count": 0,
                "active_energy": 0.2,
                "repressed_energy": 0.1,
            }

    tui = AuroraTerminalTUI(FakeRuntime(), session_id="cache-test")
    assert calls == {"identity": 1, "stats": 1}

    tui._refresh_live_state()
    assert calls == {"identity": 1, "stats": 1}


def test_header_text_hides_default_session_id(tmp_path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        )
    )

    default_tui = AuroraTerminalTUI(runtime, session_id="terminal_observer")
    custom_tui = AuroraTerminalTUI(runtime, session_id="custom-session")

    assert "terminal_observer" not in default_tui._header_text()
    assert "custom-session" in custom_tui._header_text()
