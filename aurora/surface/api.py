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
            continuity_pressure=summary["continuity_pressure"],
            sleep_pressure=summary["sleep_pressure"],
            coherence_pressure=summary["coherence_pressure"],
            softness=summary["softness"],
            boundary_tension=summary["boundary_tension"],
            active_relation_id=summary["active_relation_id"],
            recent_chapter_bias=summary["recent_chapter_bias"],
            turns=summary["turns"],
            memory_fragments=summary["memory_fragments"],
            memory_traces=summary["memory_traces"],
            memory_associations=summary["memory_associations"],
            memory_chapters=summary["memory_chapters"],
            relation_count=summary["relation_count"],
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
