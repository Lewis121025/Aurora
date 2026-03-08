from __future__ import annotations

import os
import time
from typing import Any, Dict

from aurora.runtime.settings import AuroraSettings
from aurora.runtime.runtime import AuroraRuntime
from aurora.version import __version__
from aurora.interfaces.api.schemas import (
    # Request models
    IngestRequest,
    QueryRequest,
    FeedbackRequest,
    EvolveRequest,
    CausalChainRequest,
    # Response models
    IngestResponse,
    QueryResponse,
    QueryHit,
    CoherenceResponse,
    SelfNarrativeResponse,
    CausalChainResponse,
    CausalChainItemV1,
    MemoryStatsResponse,
    HealthResponse,
)

try:
    from fastapi import FastAPI
except Exception as e:  # pragma: no cover
    raise RuntimeError("FastAPI未安装。使用以下命令安装: pip install -e '.[api]'") from e


settings = AuroraSettings(data_dir=os.environ.get("AURORA_DATA_DIR", "./data"))
runtime = AuroraRuntime(settings=settings)

app = FastAPI(title="AURORA Memory API", version=__version__)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=time.time(),
    )


@app.post("/v1/memory/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """摄入新的交互到记忆中"""
    # 如果未提供则生成event_id
    event_id = req.event_id or f"evt_{int(time.time() * 1000)}"
    r = runtime.ingest_interaction(
        event_id=event_id,
        session_id=req.session_id,
        user_message=req.user_message,
        agent_message=req.agent_message,
        actors=req.actors,
        context=req.context,
        ts=req.ts,
    )
    return IngestResponse(**r.__dict__)


@app.post("/v1/memory/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """按语义相似性查询记忆"""
    r = runtime.query(text=req.text, k=req.k)
    return QueryResponse(
        query=r.query,
        attractor_path_len=r.attractor_path_len,
        hits=[QueryHit(**h.__dict__) for h in r.hits],
    )


@app.post("/v1/memory/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """提供关于检索质量的反馈"""
    runtime.feedback(query_text=req.query_text, chosen_id=req.chosen_id, success=req.success)
    return {"ok": True}


# -----------------------------------------------------------------------------
# 扩展API: 一致性、自我叙事、因果
# -----------------------------------------------------------------------------


@app.get("/v1/memory/coherence", response_model=CoherenceResponse)
def check_coherence() -> CoherenceResponse:
    """检查当前单用户记忆的一致性"""
    result = runtime.check_coherence()
    return CoherenceResponse(
        overall_score=result.overall_score,
        conflict_count=result.conflict_count,
        unfinished_story_count=result.unfinished_story_count,
        recommendations=result.recommendations,
    )


@app.get("/v1/memory/self-narrative", response_model=SelfNarrativeResponse)
def get_self_narrative() -> SelfNarrativeResponse:
    """获取当前单用户代理的自我叙事"""
    data = runtime.get_self_narrative()
    return SelfNarrativeResponse(**data)


@app.post("/v1/memory/causal-chain", response_model=CausalChainResponse)
def get_causal_chain(req: CausalChainRequest) -> CausalChainResponse:
    """获取节点的因果链"""
    chain = runtime.get_causal_chain(req.node_id, req.direction)
    return CausalChainResponse(
        chain=[CausalChainItemV1(**item) for item in chain]
    )


@app.post("/v1/memory/evolve")
def evolve(req: EvolveRequest) -> Dict[str, Any]:
    """触发当前单用户记忆的演化"""
    runtime.evolve()
    return {
        "ok": True,
        "stories": len(runtime.mem.stories),
        "themes": len(runtime.mem.themes),
    }


@app.get("/v1/memory/stats", response_model=MemoryStatsResponse)
def get_stats() -> MemoryStatsResponse:
    """获取当前单用户记忆统计"""
    coherence = runtime.check_coherence()
    narrative = runtime.get_self_narrative()

    return MemoryStatsResponse(
        plot_count=len(runtime.mem.plots),
        story_count=len(runtime.mem.stories),
        theme_count=len(runtime.mem.themes),
        coherence_score=coherence.overall_score,
        self_narrative_coherence=narrative["coherence_score"],
    )
