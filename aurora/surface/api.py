"""Aurora v2 FastAPI surface."""

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
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    now_ts: float | None = None


class CompileRequest(BaseModel):
    session_id: str | None = None
    now_ts: float | None = None


class RecallRequest(BaseModel):
    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


def build_app(engine: AuroraKernel) -> FastAPI:
    """构建 Aurora v2 API。"""
    app = FastAPI(title="Aurora", version="2.0")
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
        return asdict(engine.turn(payload.session_id, payload.text, now_ts=payload.now_ts))

    @app.post("/compile")
    def compile_pending(payload: CompileRequest) -> dict[str, Any]:
        return asdict(engine.compile_pending(session_id=payload.session_id, now_ts=payload.now_ts))

    @app.get("/snapshot/{session_id}")
    def snapshot(session_id: str) -> dict[str, Any]:
        return asdict(engine.snapshot(session_id))

    @app.post("/recall")
    def recall(payload: RecallRequest) -> dict[str, Any]:
        return asdict(engine.recall(payload.session_id, payload.query, limit=payload.limit))

    return app


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    return build_app(AuroraKernel.create())
