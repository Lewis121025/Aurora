from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aurora.runtime.engine import AuroraEngine
from aurora.runtime.projections import StateSummary
from aurora.surface.schemas import (
    HealthResponse,
    PhaseResponse,
    StateResponse,
    TurnRequest,
    TurnResponse,
)

_OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json"})


def build_app(engine: AuroraEngine) -> FastAPI:
    app = FastAPI(title="Aurora Surface")
    runtime = engine
    api_key = os.environ.get("AURORA_API_KEY")

    @app.middleware("http")
    async def _auth(request: Request, call_next):  # type: ignore[no-untyped-def]
        if api_key and request.url.path not in _OPEN_PATHS:
            provided = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if provided != api_key:
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        summary = runtime.health_summary()
        return HealthResponse(
            status=str(summary["status"]),
            phase=str(summary["phase"]),
            turns=int(summary["turns"]),
            transitions=int(summary["transitions"]),
        )

    @app.get("/state", response_model=StateResponse)
    def state() -> StateResponse:
        summary: StateSummary = runtime.state_summary()
        return StateResponse(
            phase=str(summary["phase"]),
            sleep_need=summary["sleep_need"],
            active_relation_ids=summary["active_relation_ids"],
            pending_sleep_relation_ids=summary["pending_sleep_relation_ids"],
            active_knot_ids=summary["active_knot_ids"],
            anchor_thread_ids=summary["anchor_thread_ids"],
            turns=summary["turns"],
            memory_fragments=summary["memory_fragments"],
            memory_traces=summary["memory_traces"],
            memory_associations=summary["memory_associations"],
            memory_threads=summary["memory_threads"],
            memory_knots=summary["memory_knots"],
            relation_formations=summary["relation_formations"],
            relation_moments=summary["relation_moments"],
            sleep_cycles=summary["sleep_cycles"],
            transitions=summary["transitions"],
        )

    @app.post("/turn", response_model=TurnResponse)
    def turn(request: TurnRequest) -> TurnResponse:
        output = runtime.handle_turn(session_id=request.session_id, text=request.text)
        return TurnResponse(
            turn_id=output.turn_id,
            response_text=output.response_text,
            aurora_move=output.aurora_move,
            dominant_channels=output.dominant_channels,
        )

    @app.post("/doze", response_model=PhaseResponse)
    def doze() -> PhaseResponse:
        output = runtime.doze()
        return PhaseResponse(phase=output.phase, transition_id=output.transition_id)

    @app.post("/sleep", response_model=PhaseResponse)
    def sleep() -> PhaseResponse:
        output = runtime.sleep()
        return PhaseResponse(phase=output.phase, transition_id=output.transition_id)

    return app


def create_app() -> FastAPI:
    return build_app(AuroraEngine.create())
