from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Iterator, cast
import urllib.request


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _json_object(raw: str) -> dict[str, object]:
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _int_field(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    assert isinstance(value, int)
    return value


def _run_cli(data_dir: Path, *args: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-m", "aurora", "--data-dir", str(data_dir), *args],
        check=True,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return _json_object(result.stdout)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(url: str, *, method: str = "GET", payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=5) as response:
        return _json_object(response.read().decode("utf-8"))


def _wait_for_health(url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = ""
            if process.stdout is not None:
                output = process.stdout.read()
            raise AssertionError(f"server exited before becoming healthy: {output}")
        try:
            payload = _request_json(url)
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
            continue
        if payload == {"status": "ok"}:
            return
    raise AssertionError(f"server did not become healthy: {last_error}")


@contextlib.contextmanager
def _running_server(data_dir: Path) -> Iterator[str]:
    port = _get_free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aurora",
            "--data-dir",
            str(data_dir),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(f"{base_url}/health", process)
        yield base_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_cli_happy_path_uses_temp_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "cli-data"

    inject = _run_cli(
        data_dir,
        "inject",
        "--payload",
        "I live in Hangzhou.",
        "--session-id",
        "session-a",
        "--turn-id",
        "turn-1",
        "--source",
        "user",
    )
    workspace = _run_cli(
        data_dir,
        "read-workspace",
        "--cue",
        "Where do I live?",
        "--session-id",
        "session-a",
    )
    stats = _run_cli(data_dir, "field-stats")

    assert inject["trace_ids"]
    assert workspace["active_trace_ids"]
    assert workspace["anchor_refs"]
    assert _int_field(stats, "packet_count") == 1
    assert _int_field(stats, "trace_count") >= 1


def test_http_happy_path_uses_temp_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "http-data"

    with _running_server(data_dir) as base_url:
        inject = _request_json(
            f"{base_url}/inject",
            method="POST",
            payload={
                "payload": "I like tea.",
                "session_id": "session-a",
                "turn_id": "turn-1",
                "source": "user",
            },
        )
        workspace = _request_json(
            f"{base_url}/read-workspace",
            method="POST",
            payload={"cue": "What do I like?", "session_id": "session-a"},
        )
        stats = _request_json(f"{base_url}/field-stats")

    assert inject["trace_ids"]
    assert workspace["active_trace_ids"]
    assert workspace["anchor_refs"]
    assert _int_field(stats, "packet_count") == 1
    assert _int_field(stats, "trace_count") >= 1
