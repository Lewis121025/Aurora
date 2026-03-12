from __future__ import annotations

from datetime import datetime

from aurora.core_math.contracts import (
    HealthEnvelope,
    InputEnvelope,
    InputOutcome,
    IntegrityEnvelope,
    WakeEnvelope,
)
from aurora.core_math.state import isoformat_utc, utc_now
from aurora.host_runtime.config import AuroraSettings
from aurora.host_runtime.errors import CollapseProviderError
from aurora.host_runtime.provider import CollapseProvider, OpenAICompatibleCollapseProvider
from aurora.host_runtime.storage import SealedBlobStore
from aurora.host_runtime.substrate_client import LocalOpaqueSubstrateClient, SubstrateClient


class AuroraRuntime:
    def __init__(
        self,
        settings: AuroraSettings | None = None,
        *,
        substrate_client: SubstrateClient | None = None,
        provider: CollapseProvider | None = None,
    ) -> None:
        self.settings = settings or AuroraSettings.from_env()
        self.substrate_client = substrate_client or LocalOpaqueSubstrateClient()
        self.provider = provider or OpenAICompatibleCollapseProvider(
            base_url=self.settings.provider_base_url,
            model=self.settings.provider_model,
            api_key=self.settings.provider_api_key,
            provider_name=self.settings.provider_name,
            timeout_s=self.settings.provider_timeout_s,
        )
        self.provider.ensure_ready()
        self.store = SealedBlobStore(
            self.settings.data_dir,
            self.settings.sealed_state_filename,
            self.settings.alarm_filename,
        )
        self._last_error: str | None = None
        if not self.store.exists():
            self.store.save(self.substrate_client.boot(now=utc_now()))

    def handle_input(
        self,
        user_text: str,
        *,
        language: str = "auto",
        when: datetime | None = None,
    ) -> InputOutcome:
        sealed = self.store.load()
        envelope = InputEnvelope(
            user_text=user_text,
            timestamp=isoformat_utc(when or utc_now()),
            language=language,
        )
        result = self.substrate_client.on_input(sealed, envelope)
        self.store.save(result.sealed_state)
        self.store.write_alarm(result.next_wake_at)

        if result.collapse_request.emit_reply:
            try:
                collapse = self.provider.collapse(result.collapse_request)
                output_text = collapse.output_text
                outcome = "emitted"
            except CollapseProviderError as exc:
                self._last_error = str(exc)
                raise
        else:
            output_text = None
            outcome = "silence"
        self._last_error = None
        return InputOutcome(
            event_id=result.event_id,
            output_text=output_text,
            outcome=outcome,
            next_wake_at=result.next_wake_at,
        )

    def process_wake(self, when: datetime | None = None) -> str | None:
        sealed = self.store.load()
        result = self.substrate_client.on_wake(
            sealed,
            WakeEnvelope(timestamp=isoformat_utc(when or utc_now())),
        )
        self.store.save(result.sealed_state)
        self.store.write_alarm(result.next_wake_at)
        self._last_error = None
        return result.next_wake_at

    def health(self) -> HealthEnvelope:
        sealed = self.store.load()
        return self.substrate_client.health_snapshot(
            sealed,
            last_error=self._last_error,
            provider_healthy=self.provider.is_healthy(),
        )

    def integrity(self) -> IntegrityEnvelope:
        sealed = self.store.load()
        return self.substrate_client.integrity_report(
            sealed,
            config_fingerprint=self.settings.provider_fingerprint(),
        )
