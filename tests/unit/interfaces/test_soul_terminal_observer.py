from __future__ import annotations

from io import StringIO

from aurora.interfaces.terminal.observer import TerminalObserver
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings


def test_terminal_observer_uses_mode_axis_vocabulary(tmp_path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        )
    )
    output = StringIO()
    observer = TerminalObserver(runtime, session_id="term", observe_mode="brief", output=output)

    observer.handle_line("/identity")
    observer.handle_line("你好")

    rendered = output.getvalue()
    assert "Identity" in rendered
    assert "mode=" in rendered
    assert "top_axes" in rendered
    assert "profile_id" not in rendered
    assert "self_narrative" not in rendered
