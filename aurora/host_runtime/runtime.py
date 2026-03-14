from __future__ import annotations
from datetime import datetime

from aurora.core_math.contracts import (
    FeedbackEnvelope,
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
        self, user_text: str, *, language: str = "auto", when: datetime | None = None
    ) -> InputOutcome:
        sealed = self.store.load()
        ts = when or utc_now()

        result = self.substrate_client.on_input(
            sealed, InputEnvelope(user_text=user_text, timestamp=isoformat_utc(ts), language=language)
        )

        if result.collapse_request.emit_reply:
            try:
                collapse = self.provider.collapse(result.collapse_request)
                output_text = collapse.output_text
                outcome = "emitted"
            except CollapseProviderError as exc:
                self._last_error = str(exc)
                self.store.save(result.sealed_state)
                self.store.write_alarm(result.next_wake_at)
                raise
        else:
            output_text = None
            outcome = "silence"

        fb_env = FeedbackEnvelope(
            event_id=result.event_id,
            output_text=output_text,
            timestamp=isoformat_utc(ts),
            is_internal=False,
        )
        final_sealed = self.substrate_client.on_feedback(result.sealed_state, fb_env)

        self.store.save(final_sealed)
        self.store.write_alarm(result.next_wake_at)
        self._last_error = None
        return InputOutcome(
            event_id=result.event_id,
            output_text=output_text,
            outcome=outcome,
            next_wake_at=result.next_wake_at,
        )

    def process_wake(self, when: datetime | None = None) -> str | None:
        sealed = self.store.load()
        ts = when or utc_now()

        result = self.substrate_client.on_wake(sealed, WakeEnvelope(timestamp=isoformat_utc(ts)))
        current_sealed = result.sealed_state

        if result.dream_request and result.event_id:
            try:
                collapse = self.provider.collapse(result.dream_request)
                if collapse.output_text:
                    fb_env = FeedbackEnvelope(
                        event_id=result.event_id,
                        output_text=collapse.output_text,
                        timestamp=isoformat_utc(ts),
                        is_internal=result.dream_request.is_internal_dream,
                        is_compression=result.dream_request.is_internal_dream_compression,
                        consumed_nodes=result.consumed_nodes,
                    )
                    current_sealed = self.substrate_client.on_feedback(current_sealed, fb_env)
            except CollapseProviderError as exc:
                self._last_error = str(exc)

        self.store.save(current_sealed)
        self.store.write_alarm(result.next_wake_at)
        return result.next_wake_at

    def health(self) -> HealthEnvelope:
        return self.substrate_client.health_snapshot(
            self.store.load(), self._last_error, self.provider.is_healthy()
        )

    def integrity(self) -> IntegrityEnvelope:
        return self.substrate_client.integrity_report(
            self.store.load(), self.settings.provider_fingerprint()
        )
