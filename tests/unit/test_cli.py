from __future__ import annotations

from pathlib import Path

import aurora.surface_api.cli as cli
from aurora.host_runtime.config import AuroraSettings
from aurora.host_runtime.runtime import AuroraRuntime
from tests.helpers.fakes import FakeCollapseProvider, FakeSubstrateClient


def _runtime(tmp_path: Path) -> AuroraRuntime:
    return AuroraRuntime(
        settings=AuroraSettings(data_dir=str(tmp_path)),
        substrate_client=FakeSubstrateClient(),
        provider=FakeCollapseProvider(),
    )


def test_cli_health_and_integrity_commands(monkeypatch, capsys, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(cli, "AuroraRuntime", lambda: runtime)

    monkeypatch.setattr("sys.argv", ["aurora", "health"])
    cli.main()
    out = capsys.readouterr().out
    assert "substrate_alive=True" in out

    monkeypatch.setattr("sys.argv", ["aurora", "integrity"])
    cli.main()
    out = capsys.readouterr().out
    assert "runtime_boundary=process-opaque" in out


def test_cli_only_exposes_seed_v1_commands() -> None:
    parser = cli._build_parser()
    subcommands = set(parser._subparsers._group_actions[0].choices.keys())
    assert subcommands == {"chat", "health", "integrity"}
