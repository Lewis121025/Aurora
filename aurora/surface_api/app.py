from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from aurora.memory import AuroraMemory
from aurora.surface_api.schemas import (
    AddMemoryRequest,
    AddMemoryResponse,
    DeleteMemoryRequest,
    GetAllResponse,
    HealthResponse,
    MemoryResult,
    SearchRequest,
    SearchResponse,
    UpdateMemoryRequest,
    UpdateMemoryResponse,
)


def build_app(memory_factory: Callable[[], AuroraMemory] | None = None) -> FastAPI:
    memory_singleton: AuroraMemory | None = None

    def get_memory() -> AuroraMemory:
        nonlocal memory_singleton
        if memory_singleton is None:
            factory = memory_factory or AuroraMemory
            memory_singleton = factory()
        return memory_singleton

    app = FastAPI(
        title="Aurora Memory",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.post("/v1/memories", response_model=AddMemoryResponse)
    def add_memory(payload: AddMemoryRequest) -> AddMemoryResponse:
        try:
            result = get_memory().add(payload.text, user_id=payload.user_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return AddMemoryResponse(**result)

    @app.post("/v1/memories/search", response_model=SearchResponse)
    def search_memories(payload: SearchRequest) -> SearchResponse:
        try:
            results = get_memory().search(
                payload.query, user_id=payload.user_id, limit=payload.limit
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return SearchResponse(results=[MemoryResult(**r) for r in results])

    @app.get("/v1/memories", response_model=GetAllResponse)
    def list_memories(user_id: str | None = None) -> GetAllResponse:
        try:
            memories = get_memory().get_all(user_id=user_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return GetAllResponse(memories=[MemoryResult(**m) for m in memories])

    @app.put("/v1/memories/{memory_id}", response_model=UpdateMemoryResponse)
    def update_memory(memory_id: str, payload: UpdateMemoryRequest) -> UpdateMemoryResponse:
        try:
            result = get_memory().update(memory_id, payload.data, user_id=payload.user_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return UpdateMemoryResponse(**result)

    @app.delete("/v1/memories")
    def delete_memory(payload: DeleteMemoryRequest) -> dict:
        try:
            get_memory().delete(payload.memory_id, user_id=payload.user_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "deleted", "memory_id": payload.memory_id}

    @app.delete("/v1/memories/all")
    def delete_all_memories(user_id: str | None = None) -> dict:
        try:
            get_memory().delete_all(user_id=user_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "all_deleted"}

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        h = get_memory().health()
        return HealthResponse(**h)

    return app


app = build_app()
