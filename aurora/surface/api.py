from __future__ import annotations

from fastapi import FastAPI

from aurora.runtime.engine import AuroraEngine
from aurora.runtime.projections import StateSummary
from aurora.surface.schemas import (
    HealthResponse,
    PhaseResponse,
    StateResponse,
    TurnRequest,
    TurnResponse,
)


def build_app(engine: AuroraEngine | None = None) -> FastAPI:
    app = FastAPI(title="Aurora Surface")
    runtime = engine or AuroraEngine.create()

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


app = build_app()
