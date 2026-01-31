"""
Aurora REST API
================

FastAPI-based REST API for the AURORA memory system.

Usage:
    # Run with uvicorn
    uvicorn aurora.api.app:app --host 0.0.0.0 --port 8000
    
    # Or import directly
    from aurora.api import app

Endpoints:
    POST /ingest    - Ingest a new interaction
    POST /query     - Query memory
    GET  /narrative - Get self-narrative
    GET  /stats     - Get memory statistics
    POST /evolve    - Trigger evolution
"""

# Schemas are always available (no FastAPI dependency)
from aurora.api.schemas import (
    QueryHitV1,
    QueryHit,
    IngestRequestV1,
    IngestResponseV1,
    QueryRequestV1,
    QueryResponseV1,
)

# App requires FastAPI - lazy import
def __getattr__(name):
    if name == "app":
        from aurora.api.app import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "app",
    # Schemas
    "QueryHitV1",
    "QueryHit",
    "IngestRequestV1",
    "IngestResponseV1",
    "QueryRequestV1",
    "QueryResponseV1",
]
