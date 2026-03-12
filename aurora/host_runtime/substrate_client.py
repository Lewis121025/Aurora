from __future__ import annotations
from datetime import datetime
from typing import Protocol

from aurora.core_math.contracts import (
    FeedbackEnvelope,
    HealthEnvelope,
    InputEnvelope,
    IntegrityEnvelope,
    SubstrateInputResult,
    SubstrateWakeResult,
    WakeEnvelope,
)
from aurora.substrate_core.engine import AuroraSubstrateCore


class SubstrateClient(Protocol):
    def boot(self, now: datetime | None = None) -> bytes: ...
    def on_input(self, sealed_state: bytes, envelope: InputEnvelope) -> SubstrateInputResult: ...
    def on_wake(self, sealed_state: bytes, envelope: WakeEnvelope) -> SubstrateWakeResult: ...
    def on_feedback(self, sealed_state: bytes, envelope: FeedbackEnvelope) -> bytes: ...
    def health_snapshot(self, sealed_state: bytes, last_error: str | None, provider_healthy: bool) -> HealthEnvelope: ...
    def integrity_report(self, sealed_state: bytes, config_fingerprint: str) -> IntegrityEnvelope: ...


class LocalOpaqueSubstrateClient:
    def __init__(self, core: AuroraSubstrateCore | None = None) -> None:
        self.core = core or AuroraSubstrateCore()

    def boot(self, now: datetime | None = None) -> bytes:
        return self.core.boot(now=now)

    def on_input(self, sealed_state: bytes, envelope: InputEnvelope) -> SubstrateInputResult:
        return self.core.on_input(sealed_state, envelope)

    def on_wake(self, sealed_state: bytes, envelope: WakeEnvelope) -> SubstrateWakeResult:
        return self.core.on_wake(sealed_state, envelope)

    def on_feedback(self, sealed_state: bytes, envelope: FeedbackEnvelope) -> bytes:
        return self.core.on_feedback(sealed_state, envelope)

    def health_snapshot(self, sealed_state: bytes, last_error: str | None, provider_healthy: bool) -> HealthEnvelope:
        return self.core.health_snapshot(sealed_state, last_error, provider_healthy)

    def integrity_report(self, sealed_state: bytes, config_fingerprint: str) -> IntegrityEnvelope:
        return self.core.integrity_report(sealed_state, config_fingerprint)
