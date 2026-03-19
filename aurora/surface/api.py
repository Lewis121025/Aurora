"""Aurora FastAPI surface."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Awaitable, Callable, Protocol, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from aurora.runtime.contracts import IngestOutput, RecallResult, SubjectMemoryState, TurnOutput
from aurora.runtime.engine import AuroraKernel

_OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json"})


def _require_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    return normalized


class TurnRequest(BaseModel):
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    now_ts: float | None = None

    @field_validator("subject_id", "session_id", "text")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class FinalizeSessionRequest(BaseModel):
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    ended_at: float | None = None

    @field_validator("subject_id", "session_id")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class RecallRequest(BaseModel):
    subject_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=20)

    @field_validator("subject_id", "query")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class SurfaceStore(Protocol):
    def subject_count(self) -> int: ...


class SurfaceKernel(Protocol):
    @property
    def store(self) -> SurfaceStore: ...

    def turn(
        self,
        subject_id: str,
        session_id: str,
        text: str,
        now_ts: float | None = None,
    ) -> TurnOutput: ...

    def finalize_session(
        self,
        subject_id: str,
        session_id: str,
        ended_at: float | None = None,
    ) -> IngestOutput: ...

    def state(self, subject_id: str) -> SubjectMemoryState: ...

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        limit: int = 8,
    ) -> RecallResult: ...


def build_app(engine: SurfaceKernel) -> FastAPI:
    """Build the Aurora API app."""
    app = FastAPI(title="Aurora", version="5.0")
    api_key = os.environ.get("AURORA_API_KEY")

    @app.middleware("http")
    async def _auth(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if api_key and request.url.path not in _OPEN_PATHS:
            provided = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if provided != api_key:
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, int | str]:
        return {"status": "ok", "subjects": engine.store.subject_count()}

    @app.post("/turn")
    def turn(payload: TurnRequest) -> dict[str, object]:
        return asdict(engine.turn(payload.subject_id, payload.session_id, payload.text, now_ts=payload.now_ts))

    @app.post("/finalize-session")
    def finalize_session(payload: FinalizeSessionRequest) -> dict[str, object]:
        return asdict(
            engine.finalize_session(
                payload.subject_id,
                payload.session_id,
                ended_at=payload.ended_at,
            )
        )

    @app.get("/state/{subject_id}")
    def state(subject_id: str) -> dict[str, object]:
        return asdict(engine.state(_require_non_empty(subject_id)))

    @app.post("/recall")
    def recall(payload: RecallRequest) -> dict[str, object]:
        return asdict(
            engine.recall(
                payload.subject_id,
                payload.query,
                limit=payload.limit,
            )
        )

    return app


def create_app() -> FastAPI:
    """Create a FastAPI app instance."""
    return build_app(cast(SurfaceKernel, AuroraKernel.create()))
