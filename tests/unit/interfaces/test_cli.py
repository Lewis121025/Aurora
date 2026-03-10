from __future__ import annotations

import aurora.interfaces.cli.entry as cli


def test_cli_observe_dispatches_to_terminal_observer(monkeypatch):
    observed = {}

    def fake_run_observer(*, data_dir, session_id, max_hits):
        observed["data_dir"] = data_dir
        observed["session_id"] = session_id
        observed["max_hits"] = max_hits

    monkeypatch.setattr(cli, "run_observer", fake_run_observer)

    cli.main(
        [
            "--data-dir",
            "./observe-data",
            "observe",
            "--session-id",
            "session-cli",
            "--max-hits",
            "7",
        ]
    )

    assert observed == {
        "data_dir": "./observe-data",
        "session_id": "session-cli",
        "max_hits": 7,
    }


def test_cli_without_subcommand_enters_live_mode(monkeypatch):
    observed = {}

    def fake_run_observer(*, data_dir, session_id, max_hits):
        observed["data_dir"] = data_dir
        observed["session_id"] = session_id
        observed["max_hits"] = max_hits

    monkeypatch.setattr(cli, "run_observer", fake_run_observer)

    cli.main(["--data-dir", "./live-data"])

    assert observed == {
        "data_dir": "./live-data",
        "session_id": "terminal_observer",
        "max_hits": 6,
    }
