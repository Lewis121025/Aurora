"""HTTP surface for Aurora."""

from __future__ import annotations

from typing import Any, cast

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from aurora.runtime.system import AuroraSystem, to_dict


class InjectRequest(BaseModel):
    payload: str = Field(min_length=1)
    session_id: str = ""
    turn_id: str | None = None
    source: str = "user"
    payload_type: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: int | None = None


class ReadWorkspaceRequest(BaseModel):
    cue: str = Field(min_length=1)
    session_id: str = ""
    k: int | None = None


class MaintenanceCycleRequest(BaseModel):
    ms_budget: int | None = None


class RespondRequest(BaseModel):
    cue: str = Field(min_length=1)
    session_id: str = ""
    turn_id: str | None = None
    source: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: int | None = None


def build_app(system: AuroraSystem) -> FastAPI:
    app = FastAPI(title="Aurora", version="v2")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/inject")
    def inject_route(payload: InjectRequest = Body(...)) -> dict[str, Any]:
        try:
            return cast(
                dict[str, Any],
                to_dict(
                system.inject(
                    {
                        "payload": payload.payload,
                        "session_id": payload.session_id,
                        "turn_id": payload.turn_id,
                        "source": payload.source,
                        "payload_type": payload.payload_type,
                        "ts": payload.ts,
                        "metadata": payload.metadata,
                    }
                )
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/read-workspace")
    def read_workspace_route(payload: ReadWorkspaceRequest = Body(...)) -> dict[str, Any]:
        try:
            return cast(
                dict[str, Any],
                to_dict(
                system.read_workspace(
                    {"payload": payload.cue, "session_id": payload.session_id},
                    k=payload.k,
                )
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/maintenance-cycle")
    def maintenance_cycle_route(payload: MaintenanceCycleRequest = Body(...)) -> dict[str, Any]:
        return cast(dict[str, Any], to_dict(system.maintenance_cycle(ms_budget=payload.ms_budget)))

    @app.post("/respond")
    def respond_route(payload: RespondRequest = Body(...)) -> dict[str, Any]:
        try:
            return cast(
                dict[str, Any],
                to_dict(
                system.respond(
                    {
                        "payload": payload.cue,
                        "session_id": payload.session_id,
                        "turn_id": payload.turn_id,
                        "source": payload.source,
                        "ts": payload.ts,
                        "metadata": payload.metadata,
                    }
                )
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/snapshot")
    def snapshot_route() -> dict[str, Any]:
        return cast(dict[str, Any], to_dict(system.snapshot()))

    @app.get("/field-stats")
    def field_stats_route() -> dict[str, Any]:
        return cast(dict[str, Any], to_dict(system.field_stats()))

    return app


__all__ = [
    "InjectRequest",
    "MaintenanceCycleRequest",
    "ReadWorkspaceRequest",
    "RespondRequest",
    "build_app",
]
