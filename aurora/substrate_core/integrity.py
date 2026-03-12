from __future__ import annotations

from datetime import datetime, timezone

from aurora.core_math.contracts import IntegrityEnvelope
from aurora.core_math.state import SEALED_STATE_VERSION


def build_integrity_report(
    *,
    sealed_state_version: str = SEALED_STATE_VERSION,
    config_fingerprint: str,
    runtime_boundary: str = "process-opaque",
    substrate_transport: str = "in-process",
) -> IntegrityEnvelope:
    return IntegrityEnvelope(
        version=SEALED_STATE_VERSION,
        runtime_boundary=runtime_boundary,
        substrate_transport=substrate_transport,
        sealed_state_version=sealed_state_version,
        config_fingerprint=config_fingerprint,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
