from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from aurora.llm.config import load_llm_config
from aurora.runtime.engine import AuroraKernel


def test_live_llm_smoke(tmp_path: Path) -> None:
    if os.environ.get("AURORA_RUN_LIVE_TESTS") != "1":
        pytest.skip("set AURORA_RUN_LIVE_TESTS=1 to enable live LLM smoke tests")
    if load_llm_config() is None:
        pytest.skip("LLM config missing; set AURORA_LLM_BASE_URL, AURORA_LLM_MODEL, and AURORA_LLM_API_KEY")

    kernel = AuroraKernel.create(data_dir=str(tmp_path / "aurora-live"))
    session_id = f"live-{uuid4().hex[:8]}"

    try:
        turn = kernel.turn(session_id, "请用一句话确认你在线，并保持直接。", now_ts=100.0)
        assert turn.response_text.strip()

        report = kernel.compile_pending(session_id, now_ts=101.0)
        assert report.failures == ()

        snapshot = kernel.snapshot(session_id)
        assert snapshot.pending_compile_count == 0

        follow_up = kernel.turn(session_id, "继续，保持同样的语气。", now_ts=102.0)
        assert follow_up.response_text.strip()
    finally:
        kernel.close()
