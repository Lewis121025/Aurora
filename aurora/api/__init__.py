"""
Aurora REST API
================

AURORA 记忆系统的FastAPI REST API。

用法:
    # 使用uvicorn运行
    uvicorn aurora.api.app:app --host 0.0.0.0 --port 8000

    # 或直接导入
    from aurora.api import app

端点:
    POST /ingest    - 摄入新的交互
    POST /query     - 查询记忆
    GET  /narrative - 获取自我叙事
    GET  /stats     - 获取记忆统计
    POST /evolve    - 触发演化
"""

# 模式总是可用的 (无FastAPI依赖)
from aurora.api.schemas import (
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
        from aurora.api.app import app
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
