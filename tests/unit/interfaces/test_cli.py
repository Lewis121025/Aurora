from __future__ import annotations

import json
from types import SimpleNamespace

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


def test_cli_stats_closes_runtime(monkeypatch, capsys):
    observed = {"closed": 0}

    class FakeRuntime:
        def get_stats(self):
            return {
                "plot_count": 1,
                "story_count": 0,
                "theme_count": 0,
                "architecture_mode": "shadow",
                "current_mode": "steady",
                "pressure": 0.1,
                "dream_count": 0,
                "repair_count": 0,
                "active_energy": 0.2,
                "repressed_energy": 0.1,
                "graph_metrics": {},
                "background_evolver": {},
            }

        def close(self):
            observed["closed"] += 1

    monkeypatch.setattr(cli, "_get_runtime", lambda data_dir=None: FakeRuntime())

    cli.main(["stats"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["architecture_mode"] == "shadow"
    assert observed["closed"] == 1


def test_cli_evolve_reports_counts_and_closes_runtime(monkeypatch, capsys):
    observed = {"closed": 0}

    class FakeRuntime:
        def evolve(self, *, dreams=None):
            return [SimpleNamespace(source="dream"), SimpleNamespace(source="repair")]

        def get_stats(self):
            return {
                "plot_count": 2,
                "story_count": 0,
                "theme_count": 0,
                "architecture_mode": "shadow",
                "current_mode": "steady",
                "pressure": 0.2,
                "dream_count": 1,
                "repair_count": 1,
                "active_energy": 0.3,
                "repressed_energy": 0.1,
                "graph_metrics": {},
                "background_evolver": {},
            }

        def close(self):
            observed["closed"] += 1

    monkeypatch.setattr(cli, "_get_runtime", lambda data_dir=None: FakeRuntime())

    cli.main(["evolve", "--dreams", "2"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["dreams"] == 1
    assert payload["repairs"] == 1
    assert payload["total"] == 2
    assert observed["closed"] == 1
