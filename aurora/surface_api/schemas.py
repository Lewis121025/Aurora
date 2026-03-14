from __future__ import annotations

from pydantic import BaseModel, Field


class AddMemoryRequest(BaseModel):
    text: str = Field(min_length=1)
    user_id: str | None = None


class AddMemoryResponse(BaseModel):
    memory_id: str
    user_id: str
    text: str
    created_at: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class MemoryResult(BaseModel):
    memory_id: str
    text: str
    source: str
    timestamp: str
    score: float


class SearchResponse(BaseModel):
    results: list[MemoryResult]


class GetAllResponse(BaseModel):
    memories: list[MemoryResult]


class UpdateMemoryRequest(BaseModel):
    data: str = Field(min_length=1)
    user_id: str | None = None


class UpdateMemoryResponse(BaseModel):
    memory_id: str
    text: str
    updated_at: str


class DeleteMemoryRequest(BaseModel):
    memory_id: str = Field(min_length=1)
    user_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    user_id: str
    graph: dict
    core: dict
