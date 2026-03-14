from __future__ import annotations

from fastapi import FastAPI

from aurora.runtime.engine import AuroraEngine, StateSummary
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
            current_relation_id=(
                None
                if summary["current_relation_id"] is None
                else str(summary["current_relation_id"])
            ),
            active_thread_ids=summary["active_thread_ids"],
            active_knot_ids=summary["active_knot_ids"],
            last_transition_at=summary["last_transition_at"],
            turns=summary["turns"],
            memory_fragments=summary["memory_fragments"],
            memory_traces=summary["memory_traces"],
            memory_associations=summary["memory_associations"],
            threads=summary["threads"],
            knots=summary["knots"],
            relation_moments=summary["relation_moments"],
            trust=summary["trust"],
            boundary_tension=summary["boundary_tension"],
            sleep_cycles=summary["sleep_cycles"],
            last_reweave_delta=summary["last_reweave_delta"],
            transitions=summary["transitions"],
        )

    @app.post("/turn", response_model=TurnResponse)
    def turn(request: TurnRequest) -> TurnResponse:
        output = runtime.handle_turn(session_id=request.session_id, text=request.text)
        return TurnResponse(
            turn_id=output.turn_id,
            response_text=output.response_text,
            touch_channels=output.touch_channels,
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


app = build_app(engine=AuroraEngine.create(data_dir=".aurora_v2"))
