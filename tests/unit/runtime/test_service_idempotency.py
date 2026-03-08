from __future__ import annotations

import os
import shutil
import tempfile

from aurora.runtime.settings import AuroraSettings
from aurora.runtime.tenant import AuroraTenant


def test_ingest_idempotent():
    tmp = tempfile.mkdtemp()
    try:
        settings = AuroraSettings(data_dir=tmp, snapshot_every_events=0)
        t = AuroraTenant(user_id="u", settings=settings)
        r1 = t.ingest_interaction(
            event_id="evt1",
            session_id="s",
            user_message="hello",
            agent_message="world",
        )
        r2 = t.ingest_interaction(
            event_id="evt1",
            session_id="s",
            user_message="hello",
            agent_message="world",
        )
        assert r1.plot_id == r2.plot_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
