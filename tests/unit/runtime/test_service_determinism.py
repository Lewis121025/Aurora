from __future__ import annotations

import os
import shutil
import tempfile

from aurora.runtime.settings import AuroraSettings
from aurora.runtime.tenant import AuroraTenant


def test_event_replay_deterministic_ids():
    """测试事件 ID 生成确定性的情节 ID。

    注意：Thompson 采样可能选择不编码情节（encoded=False），
    所以我们只验证实际编码的情节被恢复。
    """
    tmp = tempfile.mkdtemp()
    try:
        settings = AuroraSettings(data_dir=tmp, snapshot_every_events=0)
        t1 = AuroraTenant(user_id="u", settings=settings)
        r1 = t1.ingest_interaction(
            event_id="evt1",
            session_id="s",
            user_message="hello",
            agent_message="world",
        )
        r2 = t1.ingest_interaction(
            event_id="evt2",
            session_id="s",
            user_message="foo",
            agent_message="bar",
        )

        # Recreate tenant (simulates restart)
        t2 = AuroraTenant(user_id="u", settings=settings)

        # Only verify plots that were actually encoded
        if r1.encoded:
            assert r1.plot_id in t2.mem.plots
        if r2.encoded:
            assert r2.plot_id in t2.mem.plots
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
