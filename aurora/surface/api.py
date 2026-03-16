"""HTTP API 模块。

基于 FastAPI 提供 Aurora 接口：
- GET /health: 健康检查
- POST /turn: 执行认知循环
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aurora.runtime.engine import AuroraEngine

_OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json"})


def build_app(engine: AuroraEngine) -> FastAPI:
    """构建 FastAPI 应用。"""
    app = FastAPI(title="Aurora")
    api_key = os.environ.get("AURORA_API_KEY")

    @app.middleware("http")
    async def _auth(request: Request, call_next):  # type: ignore[no-untyped-def]
        if api_key and request.url.path not in _OPEN_PATHS:
            provided = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if provided != api_key:
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "relations": len(engine.relational_states)}

    @app.post("/turn")
    def turn(session_id: str, text: str) -> dict:
        output = engine.handle_turn(session_id=session_id, text=text)
        return {
            "turn_id": output.turn_id,
            "response_text": output.response_text,
            "aurora_move": output.aurora_move,
        }

    return app


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    return build_app(AuroraEngine.create())
