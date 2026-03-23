"""HTTP surface for Aurora."""

from __future__ import annotations

import os
import secrets
from typing import Any, cast

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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


def _load_api_key() -> str | None:
    value = os.getenv("AURORA_API_KEY", "").strip()
    return value or None


_BEARER = HTTPBearer(auto_error=False)


def _auth_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _validate_api_key(
    api_key: str,
    credentials: HTTPAuthorizationCredentials | None,
) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _auth_error()
    if not secrets.compare_digest(credentials.credentials, api_key):
        raise _auth_error()


def build_app(system: AuroraSystem) -> FastAPI:
    app = FastAPI(
        title="Aurora",
        version="v2",
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
    )
    api_key = _load_api_key()

    def require_api_key(
        credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
    ) -> None:
        if api_key is not None:
            _validate_api_key(api_key, credentials)

    protected = APIRouter(dependencies=([Depends(require_api_key)] if api_key is not None else []))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @protected.post("/inject")
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

    @protected.post("/read-workspace")
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

    @protected.post("/maintenance-cycle")
    def maintenance_cycle_route(payload: MaintenanceCycleRequest = Body(...)) -> dict[str, Any]:
        return cast(dict[str, Any], to_dict(system.maintenance_cycle(ms_budget=payload.ms_budget)))

    @protected.post("/respond")
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

    @protected.post("/snapshot")
    def snapshot_route() -> dict[str, Any]:
        return cast(dict[str, Any], to_dict(system.snapshot()))

    @protected.get("/field-stats")
    def field_stats_route() -> dict[str, Any]:
        return cast(dict[str, Any], to_dict(system.field_stats()))

    app.include_router(protected)
    return app


__all__ = [
    "InjectRequest",
    "MaintenanceCycleRequest",
    "ReadWorkspaceRequest",
    "RespondRequest",
    "build_app",
]
