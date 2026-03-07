from __future__ import annotations

import os
import time
from typing import Any, Dict

from aurora.config import AuroraSettings
from aurora.hub import AuroraHub
from aurora.api.schemas import (
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
    from fastapi import FastAPI, HTTPException
except Exception as e:  # pragma: no cover
    raise RuntimeError("FastAPI未安装。使用以下命令安装: pip install -e '.[api]'") from e


settings = AuroraSettings(data_dir=os.environ.get("AURORA_DATA_DIR", "./data"))
hub = AuroraHub(settings=settings)

app = FastAPI(title="AURORA Memory API", version="0.1.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=time.time(),
    )


@app.post("/v1/memory/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """摄入新的交互到记忆中"""
    t = hub.tenant(req.user_id)
    # 如果未提供则生成event_id
    event_id = req.event_id or f"evt_{int(time.time() * 1000)}"
    r = t.ingest_interaction(
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
    t = hub.tenant(req.user_id)
    r = t.query(text=req.text, k=req.k)
    return QueryResponse(
        query=r.query,
        attractor_path_len=r.attractor_path_len,
        hits=[QueryHit(**h.__dict__) for h in r.hits],
    )


@app.post("/v1/memory/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """提供关于检索质量的反馈"""
    t = hub.tenant(req.user_id)
    t.feedback(query_text=req.query_text, chosen_id=req.chosen_id, success=req.success)
    return {"ok": True}


# -----------------------------------------------------------------------------
# 扩展API: 一致性、自我叙事、因果
# -----------------------------------------------------------------------------


@app.get("/v1/memory/coherence/{user_id}", response_model=CoherenceResponse)
def check_coherence(user_id: str) -> CoherenceResponse:
    """检查用户的记忆一致性"""
    t = hub.tenant(user_id)
    result = t.check_coherence()
    return CoherenceResponse(
        overall_score=result.overall_score,
        conflict_count=result.conflict_count,
        unfinished_story_count=result.unfinished_story_count,
        recommendations=result.recommendations,
    )


@app.get("/v1/memory/self-narrative/{user_id}", response_model=SelfNarrativeResponse)
def get_self_narrative(user_id: str) -> SelfNarrativeResponse:
    """获取用户代理的自我叙事"""
    t = hub.tenant(user_id)
    data = t.get_self_narrative()
    return SelfNarrativeResponse(**data)


@app.post("/v1/memory/causal-chain", response_model=CausalChainResponse)
def get_causal_chain(req: CausalChainRequest) -> CausalChainResponse:
    """获取节点的因果链"""
    t = hub.tenant(req.user_id)
    chain = t.get_causal_chain(req.node_id, req.direction)
    return CausalChainResponse(
        chain=[CausalChainItemV1(**item) for item in chain]
    )


@app.post("/v1/memory/evolve")
def evolve(req: EvolveRequest) -> Dict[str, Any]:
    """触发用户记忆的演化"""
    t = hub.tenant(req.user_id)
    t.evolve()
    return {
        "ok": True,
        "stories": len(t.mem.stories),
        "themes": len(t.mem.themes),
    }


@app.get("/v1/memory/stats/{user_id}", response_model=MemoryStatsResponse)
def get_stats(user_id: str) -> MemoryStatsResponse:
    """获取用户的记忆统计"""
    t = hub.tenant(user_id)
    coherence = t.check_coherence()
    narrative = t.get_self_narrative()

    return MemoryStatsResponse(
        plot_count=len(t.mem.plots),
        story_count=len(t.mem.stories),
        theme_count=len(t.mem.themes),
        coherence_score=coherence.overall_score,
        self_narrative_coherence=narrative["coherence_score"],
    )
