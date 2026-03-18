from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from aurora.runtime.engine import AuroraKernel

pytest.importorskip("mcp.server.fastmcp")

if os.environ.get("AURORA_LIVE_TESTS") != "1":
    pytest.skip("set AURORA_LIVE_TESTS=1 to run live Bailian tests", allow_module_level=True)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized = value.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
            normalized = normalized[1:-1]
        values[key.strip()] = normalized
    return values


def _live_env() -> dict[str, str]:
    dotenv_path = _REPO_ROOT / ".env"
    if not dotenv_path.is_file():
        pytest.skip("live Bailian tests require /Users/lewis/Aurora/.env")
    env = os.environ.copy()
    env.update(_load_dotenv(dotenv_path))
    if env.get("AURORA_LLM_PROVIDER", "").strip().lower() != "bailian":
        pytest.skip("live Bailian tests require AURORA_LLM_PROVIDER=bailian")
    required = (
        "AURORA_BAILIAN_LLM_API_KEY",
        "AURORA_BAILIAN_LLM_MODEL",
        "AURORA_BAILIAN_LLM_BASE_URL",
    )
    missing = [name for name in required if not env.get(name, "").strip()]
    if missing:
        pytest.skip(f"live Bailian tests missing config: {', '.join(missing)}")
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["AURORA_API_KEY"] = "secret-token"
    return env


def _pick_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _http_json(
    method: str,
    url: str,
    payload: dict[str, object] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_live_bailian_kernel_turn_state_and_recall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _live_env()
    for key, value in env.items():
        if key.startswith("AURORA_"):
            monkeypatch.setenv(key, value)

    kernel = AuroraKernel.create(data_dir=str(tmp_path / ".aurora"))
    try:
        subject_id = "subject-sdk-real"
        turn1 = kernel.turn(subject_id, "我现在住在杭州，也喜欢爵士乐。")
        turn2 = kernel.turn(subject_id, "最近我在适应新的生活节奏。")
        state = kernel.state(subject_id)
        recall = kernel.recall(subject_id, "杭州 生活", limit=5)
    finally:
        kernel.close()

    assert turn1.response_text.strip()
    assert turn2.response_text.strip()
    assert any("杭州" in atom.text for atom in state.atoms), state
    assert any("杭州" in atom.text for atom in recall.atoms), recall


def test_live_bailian_conversation_recalls_user_facts_in_follow_up_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _live_env()
    for key, value in env.items():
        if key.startswith("AURORA_"):
            monkeypatch.setenv(key, value)

    kernel = AuroraKernel.create(data_dir=str(tmp_path / ".aurora"))
    try:
        subject_id = "subject-dialogue-live"
        kernel.turn(subject_id, "我现在住在杭州，也喜欢爵士乐。")
        location_reply = kernel.turn(subject_id, "你还记得我现在住在哪里吗？请直接回答地点。").response_text
        music_reply = kernel.turn(subject_id, "你还记得我喜欢什么音乐吗？").response_text
    finally:
        kernel.close()

    assert "杭州" in location_reply, location_reply
    assert "爵士乐" in music_reply, music_reply


def test_live_bailian_conversation_prefers_corrected_memory_in_follow_up_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _live_env()
    for key, value in env.items():
        if key.startswith("AURORA_"):
            monkeypatch.setenv(key, value)

    kernel = AuroraKernel.create(data_dir=str(tmp_path / ".aurora"))
    try:
        subject_id = "subject-dialogue-correction-live"
        kernel.turn(subject_id, "我之前住在上海。")
        kernel.turn(subject_id, "更正，我现在住在杭州。")
        reply = kernel.turn(subject_id, "那我现在住在哪里？").response_text
    finally:
        kernel.close()

    assert "杭州" in reply, reply


def test_live_bailian_cli_http_and_mcp_smoke(tmp_path: Path) -> None:
    env = _live_env()
    workdir = tmp_path

    def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "aurora", *args],
            cwd=workdir,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    cli_turn = run_cli("turn", "我现在住在杭州，也喜欢爵士乐。", "--subject-id", "subject-cli-real")
    cli_state = run_cli("state", "--subject-id", "subject-cli-real")
    cli_recall = run_cli("recall", "我现在住在哪里？", "--subject-id", "subject-cli-real", "--limit", "5")
    cli_status = run_cli("status")

    state_obj = json.loads(cli_state.stdout)
    recall_obj = json.loads(cli_recall.stdout)
    status_obj = json.loads(cli_status.stdout)

    assert cli_turn.stdout.strip()
    assert state_obj["subject_id"] == "subject-cli-real"
    assert any("杭州" in atom["text"] for atom in state_obj["atoms"]), state_obj
    assert any("杭州" in atom["text"] for atom in recall_obj["atoms"]), recall_obj
    assert status_obj["status"] == "ok" and status_obj["subjects"] >= 1, status_obj

    api_port = _pick_port()
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "aurora.surface.api:create_app", "--factory", "--host", "127.0.0.1", "--port", str(api_port)],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{api_port}/health", timeout=2.0) as response:
                    if response.status == 200:
                        break
            except Exception:
                if api_proc.poll() is not None:
                    stderr = api_proc.stderr.read() if api_proc.stderr is not None else ""
                    raise AssertionError(stderr)
                time.sleep(0.2)
        else:
            raise AssertionError("uvicorn did not start in time")

        health_status, health_obj = _http_json("GET", f"http://127.0.0.1:{api_port}/health")
        unauthorized_status, unauthorized_obj = _http_json("GET", f"http://127.0.0.1:{api_port}/state/subject-api-real")
        auth_headers = {"Content-Type": "application/json", "Authorization": "Bearer secret-token"}
        turn_status, turn_obj = _http_json(
            "POST",
            f"http://127.0.0.1:{api_port}/turn",
            {"subject_id": "subject-api-real", "text": "我现在住在杭州，也喜欢爵士乐。"},
            headers=auth_headers,
        )
        state_status, state_api = _http_json(
            "GET",
            f"http://127.0.0.1:{api_port}/state/subject-api-real",
            headers={"Authorization": "Bearer secret-token"},
        )
        recall_status, recall_api = _http_json(
            "POST",
            f"http://127.0.0.1:{api_port}/recall",
            {"subject_id": "subject-api-real", "query": "我现在住在哪里？", "limit": 5},
            headers=auth_headers,
        )
    finally:
        api_proc.terminate()
        try:
            api_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api_proc.kill()
            api_proc.wait(timeout=10)

    state_atoms = state_api.get("atoms")
    recall_atoms = recall_api.get("atoms")

    assert health_status == 200 and health_obj["status"] == "ok", health_obj
    assert unauthorized_status == 401 and unauthorized_obj["detail"] == "unauthorized", unauthorized_obj
    assert turn_status == 200 and turn_obj["subject_id"] == "subject-api-real", turn_obj
    assert isinstance(state_atoms, list), state_api
    assert isinstance(recall_atoms, list), recall_api
    assert state_status == 200 and any(isinstance(atom, dict) and "杭州" in str(atom.get("text", "")) for atom in state_atoms), state_api
    assert recall_status == 200 and any(isinstance(atom, dict) and "杭州" in str(atom.get("text", "")) for atom in recall_atoms), recall_api

    mcp_proc = subprocess.Popen(
        [sys.executable, "-m", "aurora.surface.mcp"],
        cwd=workdir,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(2.0)
        if mcp_proc.poll() is not None:
            stderr = mcp_proc.stderr.read() if mcp_proc.stderr is not None else ""
            raise AssertionError(f"MCP exited early: {stderr}")
    finally:
        mcp_proc.terminate()
        try:
            mcp_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mcp_proc.kill()
            mcp_proc.wait(timeout=10)
