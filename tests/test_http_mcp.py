from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from fastapi.testclient import TestClient

import aurora.surfaces.mcp as mcp_module
from aurora.core.types import DecoderOutput, DecoderRequest
from aurora.models.decoder import TransformersLocalDecoder, build_local_decoder
from aurora.runtime import AuroraSystem, AuroraSystemConfig
import aurora.runtime.system as system_module
from aurora.surfaces.http import build_app

from tests.conftest import QueueLLM, SystemFactory


def _mcp_context(system: AuroraSystem) -> Any:
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=mcp_module.AuroraMCPContext(system=system),
        )
    )


def test_snapshot_uses_internal_data_dir_path(system_factory: SystemFactory) -> None:
    system = system_factory()
    client = TestClient(build_app(system))

    response = client.post("/snapshot")
    assert response.status_code == 200
    snapshot_path = response.json()["snapshot_path"]
    assert snapshot_path.startswith(system.config.data_dir)
    assert snapshot_path.endswith(".json")

    assert mcp_module.aurora_snapshot(ctx=_mcp_context(system))["snapshot_path"].startswith(system.config.data_dir)


def test_invalid_source_is_rejected_across_surfaces(system_factory: SystemFactory) -> None:
    system = system_factory()
    client = TestClient(build_app(system))
    payload = {
        "payload": "hello",
        "session_id": "session-a",
        "turn_id": "turn-1",
        "source": "invalid",
        "metadata": {"role": "user"},
    }

    with pytest.raises(ValueError, match="source must be one of"):
        system.inject(payload)

    response = client.post("/inject", json=payload)
    assert response.status_code == 400
    assert "source must be one of" in response.json()["detail"]

    with pytest.raises(ValueError, match="source must be one of"):
        mcp_module.aurora_inject(
            payload="hello",
            session_id="session-a",
            turn_id="turn-1",
            source="invalid",
            metadata={"role": "user"},
            ctx=_mcp_context(system),
        )


def test_http_routes_require_bearer_when_api_key_is_configured(
    system_factory: SystemFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AURORA_API_KEY", "secret-token")
    system = system_factory()
    client = TestClient(build_app(system))

    health = client.get("/health")
    docs = client.get("/docs")
    openapi = client.get("/openapi.json")
    redoc = client.get("/redoc")
    oauth_redirect = client.get("/docs/oauth2-redirect")
    unauthorized = client.post("/inject", json={"payload": "hello", "session_id": "s", "turn_id": "t", "source": "user"})
    wrong_token = client.post(
        "/inject",
        headers={"Authorization": "Bearer wrong-token"},
        json={"payload": "hello", "session_id": "s", "turn_id": "t", "source": "user"},
    )
    authorized = client.post(
        "/inject",
        headers={"Authorization": "Bearer secret-token"},
        json={"payload": "hello", "session_id": "s", "turn_id": "t", "source": "user"},
    )

    assert health.status_code == 200
    assert docs.status_code == 200
    assert openapi.status_code == 200
    assert redoc.status_code == 404
    assert oauth_redirect.status_code == 404
    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert wrong_token.status_code == 401
    assert authorized.status_code == 200


def test_respond_rolls_back_when_generation_fails(system_factory: SystemFactory) -> None:
    system = system_factory()

    class RaisingDecoder:
        def decode(self, request: DecoderRequest) -> DecoderOutput:
            del request
            raise RuntimeError("decode failed")

    system._local_decoder = RaisingDecoder()

    with pytest.raises(RuntimeError, match="decode failed"):
        system.respond(
            {
                "payload": "Please respond.",
                "session_id": "session-a",
                "turn_id": "turn-1",
                "source": "user",
                "metadata": {"role": "user"},
            }
        )

    assert system.field.step == 0
    assert system.field.anchor_store.packets == {}
    assert system.field.anchor_store.anchors == {}
    assert system.field.trace_store.traces == {}
    assert system.field.edge_store.edges == {}


def test_system_does_not_build_local_decoder_when_llm_is_present(
    system_factory: SystemFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal calls
        calls += 1
        raise AssertionError("local decoder should not be built during responder-backed init")

    monkeypatch.setattr(system_module, "build_local_decoder", fail)
    system = system_factory(llm=QueueLLM("ok"))

    assert calls == 0
    system.close()


def test_close_is_idempotent(system_factory: SystemFactory) -> None:
    system = system_factory()

    system.close()
    system.close()


def test_system_config_data_dir_owns_default_runtime_paths(tmp_path: Path) -> None:
    root = tmp_path / "aurora-root"
    system = AuroraSystem(
        AuroraSystemConfig(
            data_dir=str(root),
        )
    )
    try:
        assert system.config.data_dir == str(root)
        assert system.config.db_path == str(root / "aurora.sqlite")
        assert system.config.blob_dir == str(root / "blobs")
        assert system.field.config.data_dir == str(root)
        assert system.field.config.db_path == str(root / "aurora.sqlite")
        assert system.field.config.blob_dir == str(root / "blobs")
    finally:
        system.close()


def test_reloaded_system_reapplies_current_runtime_config(tmp_path: Path) -> None:
    root = tmp_path / "aurora"
    db_path = root / "aurora.sqlite"
    first = AuroraSystem(
        AuroraSystemConfig(
            db_path=str(db_path),
            data_dir=str(root),
            blob_dir=str(root / "blobs"),
            trace_budget=111,
            edge_budget=222,
            packet_chars=64,
            hot_trace_limit=4,
            workspace_k=3,
            settle_steps=2,
        )
    )
    try:
        first.inject({"payload": "hello", "session_id": "session-a", "turn_id": "turn-1", "source": "user"})
    finally:
        first.close()

    second = AuroraSystem(
        AuroraSystemConfig(
            db_path=str(db_path),
            data_dir=str(root),
            blob_dir=str(root / "blobs"),
            trace_budget=999,
            edge_budget=777,
            packet_chars=256,
            hot_trace_limit=9,
            workspace_k=5,
            settle_steps=6,
        )
    )
    try:
        assert second.field.config.trace_budget == 999
        assert second.field.budget_config.max_traces == 999
        assert second.field.config.edge_budget == 777
        assert second.field.budget_config.max_edges == 777
        assert second.field.config.packet_chars == 256
        assert second.field.packetizer.max_chars == 256
        assert second.field.config.frontier_size == 9
        assert second.field.config.workspace_size == 5
        assert second.field.config.settle_steps == 6
    finally:
        second.close()


def test_reloaded_system_rejects_incompatible_encoder_dim(tmp_path: Path) -> None:
    root = tmp_path / "aurora"
    db_path = root / "aurora.sqlite"
    system = AuroraSystem(
        AuroraSystemConfig(
            db_path=str(db_path),
            data_dir=str(root),
            blob_dir=str(root / "blobs"),
            encoder_dim=32,
        )
    )
    try:
        system.inject({"payload": "hello", "session_id": "session-a", "turn_id": "turn-1", "source": "user"})
    finally:
        system.close()

    with pytest.raises(RuntimeError, match="stored field latent_dim 32 does not match configured latent_dim 64"):
        AuroraSystem(
            AuroraSystemConfig(
                db_path=str(db_path),
                data_dir=str(root),
                blob_dir=str(root / "blobs"),
                encoder_dim=64,
            )
        )


def test_field_stats_is_guarded_by_system_lock(system_factory: SystemFactory) -> None:
    system = system_factory()

    class SpyLock:
        def __init__(self) -> None:
            self.enter_count = 0

        def __enter__(self) -> "SpyLock":
            self.enter_count += 1
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
            del exc_type, exc, tb
            return False

    spy = SpyLock()
    setattr(system, "_lock", spy)

    stats = system.field_stats()

    assert stats.packet_count >= 0
    assert spy.enter_count == 1


def test_background_maintenance_exceptions_are_not_swallowed(
    system_factory: SystemFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = system_factory()
    triggered = threading.Event()

    def hook(args: threading.ExceptHookArgs) -> None:
        if isinstance(args.exc_value, RuntimeError) and str(args.exc_value) == "boom":
            triggered.set()

    def fail(*, ms_budget: int | None = None) -> None:
        del ms_budget
        raise RuntimeError("boom")

    monkeypatch.setattr(threading, "excepthook", hook)
    monkeypatch.setattr(system, "maintenance_cycle", fail)

    system.start_background_maintenance(interval=0.01, ms_budget=1)
    try:
        assert triggered.wait(timeout=1.0)
    finally:
        system.stop_background_maintenance()


def test_mcp_lifespan_owns_one_runtime_and_uses_explicit_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AURORA_DATA_DIR", str(tmp_path))

    async def run() -> tuple[str, bool, bool]:
        async with mcp_module._aurora_lifespan(mcp_module.mcp) as state:
            ctx = _mcp_context(state.system)
            mcp_module.aurora_inject(
                payload="hello",
                session_id="session-a",
                turn_id="turn-1",
                source="user",
                ctx=ctx,
            )
            stats = mcp_module.aurora_field_stats(ctx=ctx)
            return state.system.config.data_dir, stats["packet_count"] == 1, state.system is mcp_module._system_from_context(ctx)

    data_dir, packet_count_ok, same_system = asyncio.run(run())

    assert data_dir == str(tmp_path)
    assert packet_count_ok
    assert same_system


def test_transformers_decoder_moves_inputs_to_model_device() -> None:
    class FakeBatch(dict[str, object]):
        def to(self, device: object) -> "FakeBatch":
            self["device"] = device
            return self

    class FakeTokenizer:
        def __call__(self, text: str, return_tensors: str) -> FakeBatch:
            assert text
            assert return_tensors == "pt"
            return FakeBatch(input_ids=[[1, 2, 3]])

        def decode(self, tokens: object, skip_special_tokens: bool = True) -> str:
            assert tokens == [1, 2, 3, 4]
            assert skip_special_tokens is True
            return "decoded"

    class FakeModel:
        device = "cuda:0"

        def generate(self, **inputs: object) -> list[list[int]]:
            assert inputs["device"] == self.device
            return [[1, 2, 3, 4]]

    class FakeTorch:
        @staticmethod
        def no_grad() -> contextlib.AbstractContextManager[None]:
            return contextlib.nullcontext()

    decoder = TransformersLocalDecoder(model_id="test-model")
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    object.__setattr__(decoder, "_model", fake_model)
    object.__setattr__(decoder, "_tokenizer", fake_tokenizer)
    object.__setattr__(decoder, "_torch", FakeTorch())

    result = decoder.decode(
        request=type(
            "Req",
            (),
            {
                "prompt": "prompt",
                "cue": "cue",
                "workspace": type("Workspace", (), {"summary_vector": (0.1, 0.2)})(),
            },
        )()
    )

    assert result.text == "decoded"


def test_serializer_decoder_backend_is_not_supported() -> None:
    with pytest.raises(ValueError, match="unsupported decoder kind: serializer"):
        build_local_decoder("serializer")
