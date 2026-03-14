from __future__ import annotations

from fastapi import FastAPI

from aurora.runtime.engine import AuroraEngine
from aurora.surface.schemas import (
    HealthResponse,
    PhaseResponse,
    StateResponse,
    TurnRequest,
    TurnResponse,
)


def build_app(engine: AuroraEngine | None = None) -> FastAPI:
    app = FastAPI(title="Aurora Next Surface")
    runtime = engine or AuroraEngine.create()

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        summary = runtime.health_summary()
        return HealthResponse(
            status=str(summary["status"]),
            phase=str(summary["phase"]),
            turns=int(summary["turns"]),
            transitions=int(summary["transitions"]),
        )

    @app.get("/v1/state", response_model=StateResponse)
    def state() -> StateResponse:
        summary = runtime.state_summary()
        return StateResponse(
            phase=str(summary["phase"]),
            updated_at=float(summary["updated_at"]),
            self_view=float(summary["self_view"]),
            world_view=float(summary["world_view"]),
            openness=float(summary["openness"]),
            turns=int(summary["turns"]),
            memory_fragments=int(summary["memory_fragments"]),
            memory_traces=int(summary["memory_traces"]),
            memory_associations=int(summary["memory_associations"]),
            avg_salience=float(summary["avg_salience"]),
            avg_narrative_weight=float(summary["avg_narrative_weight"]),
            narrative_pressure=float(summary["narrative_pressure"]),
            sleep_cycles=int(summary["sleep_cycles"]),
            last_reweave_delta=float(summary["last_reweave_delta"]),
            relation_moments=int(summary["relation_moments"]),
            relation_tone=str(summary["relation_tone"]),
            relation_strength=float(summary["relation_strength"]),
            transitions=int(summary["transitions"]),
        )

    @app.post("/v1/turn", response_model=TurnResponse)
    def turn(request: TurnRequest) -> TurnResponse:
        output = runtime.handle_turn(session_id=request.session_id, text=request.text)
        return TurnResponse(
            turn_id=output.turn_id,
            response_text=output.response_text,
            touch_modes=output.touch_modes,
        )

    @app.post("/v1/doze", response_model=PhaseResponse)
    def doze() -> PhaseResponse:
        output = runtime.doze()
        return PhaseResponse(phase=output.phase, transition_id=output.transition_id)

    @app.post("/v1/sleep", response_model=PhaseResponse)
    def sleep() -> PhaseResponse:
        output = runtime.sleep()
        return PhaseResponse(phase=output.phase, transition_id=output.transition_id)

    return app


app = build_app()
