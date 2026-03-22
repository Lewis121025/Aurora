from __future__ import annotations

import contextlib
import threading

import pytest
from fastapi.testclient import TestClient

import aurora.surfaces.mcp as mcp_module
from aurora.core.types import DecoderOutput, DecoderRequest
from aurora.models.decoder import TransformersLocalDecoder, build_local_decoder
from aurora.surfaces.http import build_app

from tests.conftest import SystemFactory


def test_snapshot_uses_internal_data_dir_path(system_factory: SystemFactory) -> None:
    system = system_factory()
    client = TestClient(build_app(system))

    try:
        response = client.post("/snapshot")
        assert response.status_code == 200
        snapshot_path = response.json()["snapshot_path"]
        assert snapshot_path.startswith(system.config.data_dir)
        assert snapshot_path.endswith(".json")

        mcp_module._system = system
        assert mcp_module.aurora_snapshot()["snapshot_path"].startswith(system.config.data_dir)
    finally:
        mcp_module._system = None


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

    mcp_module._system = system
    try:
        with pytest.raises(ValueError, match="source must be one of"):
            mcp_module.aurora_inject(
                payload="hello",
                session_id="session-a",
                turn_id="turn-1",
                source="invalid",
                metadata={"role": "user"},
            )
    finally:
        mcp_module._system = None


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


def test_close_is_idempotent(system_factory: SystemFactory) -> None:
    system = system_factory()

    system.close()
    system.close()


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
