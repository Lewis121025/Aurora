from __future__ import annotations

from pathlib import Path

import pytest

from aurora.exceptions import ConfigurationError
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings


@pytest.fixture
def runtime_settings(temp_data_dir: Path) -> AuroraSettings:
    return AuroraSettings(
        data_dir=str(temp_data_dir),
        llm_provider="mock",
        embedding_provider="hash",
        dim=64,
        pii_redaction_enabled=False,
        snapshot_every_events=1,
        memory_seed=11,
    )


def test_v2_runtime_bootstraps_seed_plots_once(runtime_settings: AuroraSettings):
    first = AuroraRuntime(settings=runtime_settings)
    seed_ids = list(first.mem.self_narrative_engine.narrative.seed_plot_ids)
    assert seed_ids

    first.ingest_interaction(
        event_id="evt_bootstrap_v2",
        session_id="chat_v2",
        user_message="帮我记住我喜欢稳定的人格设定。",
        agent_message="我会在同一人格底色上继续和你交流。",
    )

    second = AuroraRuntime(settings=runtime_settings)
    assert second.mem.self_narrative_engine.narrative.seed_plot_ids == seed_ids
    assert len([pid for pid in second.mem.plots if pid in seed_ids]) == len(seed_ids)


def test_v2_runtime_rejects_legacy_pickle_snapshot(tmp_path: Path):
    data_dir = tmp_path / "legacy"
    snapshot_dir = data_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "snapshot_1.pkl").write_bytes(b"legacy")

    settings = AuroraSettings(
        data_dir=str(data_dir),
        llm_provider="mock",
        embedding_provider="hash",
        pii_redaction_enabled=False,
        snapshot_every_events=0,
    )

    with pytest.raises(ConfigurationError, match="legacy pickle snapshots"):
        AuroraRuntime(settings=settings)
