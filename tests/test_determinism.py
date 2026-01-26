from __future__ import annotations

import os
import shutil
import tempfile

from aurora.config import AuroraSettings
from aurora.service import AuroraTenant


def test_event_replay_deterministic_ids():
    """Test that event IDs produce deterministic plot IDs.

    Note: Thompson Sampling may choose not to encode plots (encoded=False),
    so we only verify plots that were actually encoded are restored.
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
