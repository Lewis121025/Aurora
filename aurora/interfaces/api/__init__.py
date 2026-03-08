"""
Aurora REST API
================

AURORA 记忆系统的FastAPI REST API。

用法:
    # 使用uvicorn运行
    uvicorn aurora.interfaces.api.app:app --host 0.0.0.0 --port 8000

    # 或直接导入
    from aurora.interfaces.api import app

端点:
    POST /v1/memory/ingest
    POST /v1/memory/query
    GET  /v1/memory/coherence
    GET  /v1/memory/self-narrative
    GET  /v1/memory/stats
    POST /v1/memory/evolve
"""

# 模式总是可用的 (无FastAPI依赖)
from aurora.interfaces.api.schemas import (
    QueryHitV1,
    QueryHit,
    IngestRequestV1,
    IngestResponseV1,
    QueryRequestV1,
    QueryResponseV1,
)

# App需要FastAPI - 延迟导入
def __getattr__(name):
    if name == "app":
        from aurora.interfaces.api.app import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "app",
    # 模式
    "QueryHitV1",
    "QueryHit",
    "IngestRequestV1",
    "IngestResponseV1",
    "QueryRequestV1",
    "QueryResponseV1",
]
