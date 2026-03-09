"""
Aurora REST API
================

Aurora Soul-Memory V3 REST API。
"""

from aurora.interfaces.api.schemas import (
    IdentityResponse,
    IngestRequest,
    IngestResponse,
    MemoryStatsResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
    RespondRequest,
)


def __getattr__(name):
    if name == "app":
        from aurora.interfaces.api.app import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "app",
    "IngestRequest",
    "IngestResponse",
    "IdentityResponse",
    "MemoryStatsResponse",
    "QueryHit",
    "QueryRequest",
    "QueryResponse",
    "RespondRequest",
]
