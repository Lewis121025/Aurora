"""FastAPI surface for AuroraSystem."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncIterator, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from aurora.system import AuroraSystem, event_ingest_to_dict, recall_result_to_dict, response_output_to_dict

_OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json"})


def _require_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    return normalized


def _payload_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return cast(dict[str, object], asdict(cast(Any, value)))


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    source: str = Field(default="dialogue", min_length=1)
    now_ts: float | None = None

    @field_validator("text", "source")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


class BatchEvent(BaseModel):
    text: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    source: str | None = None
    now_ts: float | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _require_non_empty(value)


class IngestBatchRequest(BaseModel):
    events: list[BatchEvent]
    source: str = Field(default="dialogue", min_length=1)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _require_non_empty(value)


class RetrieveRequest(BaseModel):
    cue: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=32)
    propagation_steps: int = Field(default=3, ge=1, le=8)

    @field_validator("cue")
    @classmethod
    def validate_cue(cls, value: str) -> str:
        return _require_non_empty(value)


class CurrentStateRequest(BaseModel):
    top_k: int = Field(default=10, ge=1, le=32)


class ReplayRequest(BaseModel):
    budget: int = Field(default=8, ge=1, le=64)


class RespondRequest(BaseModel):
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    source: str = Field(default="dialogue", min_length=1)
    top_k: int = Field(default=8, ge=1, le=32)
    propagation_steps: int = Field(default=3, ge=1, le=8)
    now_ts: float | None = None

    @field_validator("session_id", "text", "source")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)


def build_app(
    system: AuroraSystem,
    *,
    lifespan: Any = None,
) -> FastAPI:
    app = FastAPI(title="Aurora", version="6.0", lifespan=lifespan)
    api_key = os.environ.get("AURORA_API_KEY")

    @app.middleware("http")
    async def auth(request: Request, call_next: Any) -> Response:
        if api_key and request.url.path not in _OPEN_PATHS:
            provided = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if provided != api_key:
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return cast(Response, await call_next(request))

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "snapshots": system.stats()["snapshots"]}

    @app.post("/ingest")
    def ingest(payload: IngestRequest) -> dict[str, object]:
        return event_ingest_to_dict(
            system.ingest(payload.text, metadata=dict(payload.metadata), source=payload.source, now_ts=payload.now_ts)
        )

    @app.post("/ingest-batch")
    def ingest_batch(payload: IngestBatchRequest) -> dict[str, object]:
        return _payload_dict(
            system.ingest_batch(
                [
                    {
                        "text": event.text,
                        "metadata": dict(event.metadata),
                        "source": event.source or payload.source,
                        "now_ts": event.now_ts,
                    }
                    for event in payload.events
                ],
                source=payload.source,
            )
        )

    @app.post("/retrieve")
    def retrieve(payload: RetrieveRequest) -> dict[str, object]:
        return recall_result_to_dict(
            system.retrieve(payload.cue, top_k=payload.top_k, propagation_steps=payload.propagation_steps)
        )

    @app.post("/current-state")
    def current_state(payload: CurrentStateRequest) -> dict[str, object]:
        return recall_result_to_dict(system.current_state(top_k=payload.top_k))

    @app.post("/replay")
    def replay(payload: ReplayRequest) -> dict[str, object]:
        return _payload_dict(system.replay(budget=payload.budget))

    @app.post("/respond")
    def respond(payload: RespondRequest) -> dict[str, object]:
        return response_output_to_dict(
            system.respond(
                payload.session_id,
                payload.text,
                metadata=dict(payload.metadata),
                source=payload.source,
                top_k=payload.top_k,
                propagation_steps=payload.propagation_steps,
                now_ts=payload.now_ts,
            )
        )

    @app.get("/stats")
    def stats() -> dict[str, object]:
        return _payload_dict(system.stats())

    @app.get("/operations")
    def operations(limit: int = 50) -> dict[str, object]:
        return {"operations": system.operation_history(limit=limit)}

    @app.get("/atoms/{atom_id}")
    def atom(atom_id: str) -> dict[str, object]:
        return _payload_dict(system.get_atom(_require_non_empty(atom_id)))

    return app


def create_app() -> FastAPI:
    system = AuroraSystem.create()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            system.close()

    return build_app(system, lifespan=lifespan)
