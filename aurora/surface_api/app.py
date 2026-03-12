from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from aurora.host_runtime.runtime import AuroraRuntime
from aurora.host_runtime.errors import CollapseProviderError
from aurora.surface_api.schemas import (
    HealthResponse,
    IntegrityResponse,
    InputRequest,
    InputResponse,
)


def build_app(runtime_factory: Callable[[], AuroraRuntime] | None = None) -> FastAPI:
    runtime_singleton: AuroraRuntime | None = None

    def get_runtime() -> AuroraRuntime:
        nonlocal runtime_singleton
        if runtime_singleton is None:
            factory = runtime_factory or AuroraRuntime
            runtime_singleton = factory()
        return runtime_singleton

    app = FastAPI(
        title="Aurora Seed v1",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.post("/v1/input", response_model=InputResponse)
    def input_turn(payload: InputRequest) -> InputResponse:
        try:
            outcome = get_runtime().handle_input(payload.user_text, language=payload.language)
        except CollapseProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return InputResponse(
            event_id=outcome.event_id,
            output_text=outcome.output_text,
            outcome=outcome.outcome,
            next_wake_at=outcome.next_wake_at,
        )

    @app.get("/v1/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        health = get_runtime().health()
        return HealthResponse(
            version=health.version,
            substrate_alive=health.substrate_alive,
            sealed_state_version=health.sealed_state_version,
            anchor_count=health.anchor_count,
            next_wake_at=health.next_wake_at,
            last_error=health.last_error,
            provider_healthy=health.provider_healthy,
        )

    @app.get("/v1/integrity", response_model=IntegrityResponse)
    def integrity() -> IntegrityResponse:
        report = get_runtime().integrity()
        return IntegrityResponse(
            version=report.version,
            runtime_boundary=report.runtime_boundary,
            substrate_transport=report.substrate_transport,
            sealed_state_version=report.sealed_state_version,
            config_fingerprint=report.config_fingerprint,
            generated_at=report.generated_at,
        )

    return app


app = build_app()
