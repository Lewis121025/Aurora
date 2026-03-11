from __future__ import annotations

import json

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
                "summary_count": 0,
                "story_count": 0,
                "theme_count": 0,
                "architecture_mode": "graph_first",
                "current_mode": "steady",
                "pressure": 0.1,
                "dream_count": 0,
                "repair_count": 0,
                "active_energy": 0.2,
                "repressed_energy": 0.1,
                "graph_metrics": {},
                "queue_depth": 0,
                "oldest_pending_age_s": None,
                "last_projected_seq": 0,
                "last_fade_ts": None,
                "last_evolve_ts": None,
            }

        def close(self):
            observed["closed"] += 1

    monkeypatch.setattr(cli, "_get_runtime", lambda data_dir=None: FakeRuntime())
    cli.main(["stats"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["architecture_mode"] == "graph_first"
    assert observed["closed"] == 1


def test_cli_job_command_reads_runtime_status(monkeypatch, capsys):
    observed = {"closed": 0}

    class FakeRuntime:
        def get_job_status(self, job_id):
            return {"job_id": job_id, "status": "queued", "job_type": "project_interaction"}

        def close(self):
            observed["closed"] += 1

    monkeypatch.setattr(cli, "_get_runtime", lambda data_dir=None: FakeRuntime())
    cli.main(["job", "job_123"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "job_123"
    assert payload["status"] == "queued"
    assert observed["closed"] == 1
