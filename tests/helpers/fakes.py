from __future__ import annotations

from datetime import datetime

from aurora.core_math.contracts import (
    CollapseRequest,
    CollapseResult,
    InputEnvelope,
    IntegrityEnvelope,
    WakeEnvelope,
)
from aurora.core_math.state import SEALED_STATE_VERSION, utc_now
from aurora.substrate_core import AuroraSubstrateCore
from aurora.host_runtime.provider import CollapseProvider


class FakeSubstrateClient:
    def __init__(self) -> None:
        self.core = AuroraSubstrateCore()

    def boot(self, now: datetime | None = None) -> bytes:
        return self.core.boot(now=now or utc_now())

    def on_input(self, sealed_state: bytes, envelope: InputEnvelope):
        return self.core.on_input(sealed_state, envelope)

    def on_wake(self, sealed_state: bytes, envelope: WakeEnvelope):
        return self.core.on_wake(sealed_state, envelope)

    def health_snapshot(self, sealed_state: bytes, last_error: str | None, provider_healthy: bool):
        return self.core.health_snapshot(sealed_state, last_error, provider_healthy)

    def integrity_report(self, sealed_state: bytes, config_fingerprint: str) -> IntegrityEnvelope:
        from aurora.host_runtime.substrate_client import LocalOpaqueSubstrateClient

        report = LocalOpaqueSubstrateClient(self.core).integrity_report(
            sealed_state,
            config_fingerprint=config_fingerprint,
        )
        assert isinstance(report, IntegrityEnvelope)
        return report


class FakeCollapseProvider(CollapseProvider):
    def __init__(self) -> None:
        self.requests: list[CollapseRequest] = []
        self._healthy = True

    def collapse(self, request: CollapseRequest) -> CollapseResult:
        self.requests.append(request)
        if not request.emit_reply:
            return CollapseResult(output_text=None, provider_name="fake")
        text = request.user_text
        if request.released_traces:
            text = f"{text} :: {request.released_traces[0].text[:60]}"
        return CollapseResult(output_text=text, provider_name="fake")

    def is_healthy(self) -> bool:
        return self._healthy

    def ensure_ready(self) -> None:
        return

    def provider_identity(self) -> str:
        return "fake"


def assert_no_internal_leaks(request: CollapseRequest) -> None:
    payload = repr(request).lower()
    assert "latent" not in payload
    assert "metric" not in payload
    assert "free_energy" not in payload
    assert SEALED_STATE_VERSION not in payload
