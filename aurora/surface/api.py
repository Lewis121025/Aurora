"""Aurora v3 FastAPI surface."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aurora.runtime.engine import AuroraKernel

_OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json"})


class TurnRequest(BaseModel):
    relation_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    now_ts: float | None = None


class RecallRequest(BaseModel):
    relation_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


def build_app(engine: AuroraKernel) -> FastAPI:
    """Build Aurora v3 API."""
    app = FastAPI(title="Aurora", version="3.0")
    api_key = os.environ.get("AURORA_API_KEY")

    @app.middleware("http")
    async def _auth(request: Request, call_next):  # type: ignore[no-untyped-def]
        if api_key and request.url.path not in _OPEN_PATHS:
            provided = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if provided != api_key:
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "relations": engine.store.relation_count()}

    @app.post("/turn")
    def turn(payload: TurnRequest) -> dict[str, Any]:
        return asdict(engine.turn(payload.relation_id, payload.text, now_ts=payload.now_ts))

    @app.get("/snapshot/{relation_id}")
    def snapshot(relation_id: str) -> dict[str, Any]:
        return asdict(engine.snapshot(relation_id))

    @app.post("/recall")
    def recall(payload: RecallRequest) -> dict[str, Any]:
        return asdict(engine.recall(payload.relation_id, payload.query, limit=payload.limit))

    return app


def create_app() -> FastAPI:
    """Create a FastAPI app instance."""
    return build_app(AuroraKernel.create())
