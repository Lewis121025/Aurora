from __future__ import annotations

import argparse
import os
from typing import Optional

from aurora.interfaces.terminal.tui import AuroraTerminalTUI
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings, DEFAULT_DATA_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="observe_runtime", description="Aurora Soul fullscreen terminal"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--session-id", type=str, default="terminal_observer")
    parser.add_argument("--max-hits", type=int, default=6)
    return parser


def run_observer(
    *,
    data_dir: Optional[str] = None,
    session_id: str = "terminal_observer",
    max_hits: int = 6,
) -> None:
    settings = AuroraSettings(
        data_dir=data_dir or os.environ.get("AURORA_DATA_DIR", DEFAULT_DATA_DIR)
    )
    settings.llm_timeout = min(float(settings.llm_timeout), 10.0)
    settings.llm_max_retries = 1
    runtime = AuroraRuntime(settings=settings)
    AuroraTerminalTUI(runtime, session_id=session_id, max_hits=max_hits).run()


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_observer(
        data_dir=args.data_dir,
        session_id=args.session_id,
        max_hits=args.max_hits,
    )
