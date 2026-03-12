from __future__ import annotations

from pathlib import Path

from aurora.host_runtime.config import AuroraSettings
from aurora.host_runtime.runtime import AuroraRuntime
from tests.helpers.fakes import FakeCollapseProvider, FakeSubstrateClient, assert_no_internal_leaks


def test_runtime_handles_input_and_persists_opaque_blob(tmp_path: Path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(data_dir=str(tmp_path)),
        substrate_client=FakeSubstrateClient(),
        provider=FakeCollapseProvider(),
    )
    outcome = runtime.handle_input("remember this", language="en")
    assert outcome.event_id
    assert (tmp_path / "sealed_state.blob").exists()
    provider = runtime.provider
    assert isinstance(provider, FakeCollapseProvider)
    assert provider.requests
    assert_no_internal_leaks(provider.requests[-1])


def test_runtime_exposes_no_background_threads(tmp_path: Path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(data_dir=str(tmp_path)),
        substrate_client=FakeSubstrateClient(),
        provider=FakeCollapseProvider(),
    )
    assert not any(name.endswith("thread") for name in vars(runtime))
    assert runtime.process_wake() is not None


def test_runtime_integrity_reports_process_boundary(tmp_path: Path) -> None:
    runtime = AuroraRuntime(
        settings=AuroraSettings(data_dir=str(tmp_path)),
        substrate_client=FakeSubstrateClient(),
        provider=FakeCollapseProvider(),
    )
    report = runtime.integrity()
    assert report.runtime_boundary == "process-opaque"
    assert report.substrate_transport == "in-process"
