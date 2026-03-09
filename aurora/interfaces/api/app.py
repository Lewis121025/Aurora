from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Dict

from aurora.runtime.settings import AuroraSettings
from aurora.runtime.runtime import AuroraRuntime
from aurora.version import __version__
from aurora.interfaces.api.schemas import (
    IngestRequest,
    QueryRequest,
    RespondRequest,
    FeedbackRequest,
    EvolveRequest,
    CausalChainRequest,
    ChatTimings,
    ChatTurnResponse,
    IngestResponse,
    QueryResponse,
    QueryHit,
    RetrievalTraceSummary,
    StructuredMemoryContext,
    EvidenceRef,
    CoherenceResponse,
    SelfNarrativeResponse,
    CausalChainResponse,
    CausalChainItem,
    MemoryStatsResponse,
    HealthResponse,
)

try:
    from fastapi import FastAPI
except Exception as e:  # pragma: no cover
    raise RuntimeError("FastAPI未安装。使用以下命令安装: pip install -e '.[api]'") from e

app = FastAPI(title="AURORA Memory API", version=__version__)


@lru_cache(maxsize=1)
def get_runtime() -> AuroraRuntime:
    settings = AuroraSettings(data_dir=os.environ.get("AURORA_DATA_DIR", "./data"))
    return AuroraRuntime(settings=settings)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=time.time(),
    )


@app.post("/v2/memory/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """摄入新的交互到记忆中"""
    # 如果未提供则生成event_id
    event_id = req.event_id or f"evt_{int(time.time() * 1000)}"
    runtime = get_runtime()
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


@app.post("/v2/memory/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """按语义相似性查询记忆"""
    runtime = get_runtime()
    r = runtime.query(text=req.text, k=req.k)
    return QueryResponse(
        query=r.query,
        attractor_path_len=r.attractor_path_len,
        hits=[QueryHit(**h.__dict__) for h in r.hits],
    )


@app.post("/v2/memory/respond", response_model=ChatTurnResponse)
def respond(req: RespondRequest) -> ChatTurnResponse:
    """基于结构化记忆上下文生成当前回复。"""
    runtime = get_runtime()
    result = runtime.respond(
        session_id=req.session_id,
        user_message=req.user_message,
        event_id=req.event_id,
        context=req.context,
        actors=req.actors,
        k=req.k,
        ts=req.ts,
    )
    return ChatTurnResponse(
        reply=result.reply,
        event_id=result.event_id,
        memory_context=StructuredMemoryContext(
            known_facts=result.memory_context.known_facts,
            preferences=result.memory_context.preferences,
            relationship_state=result.memory_context.relationship_state,
            active_narratives=result.memory_context.active_narratives,
            temporal_context=result.memory_context.temporal_context,
            system_intuition=result.memory_context.system_intuition,
            cautions=result.memory_context.cautions,
            evidence_refs=[
                EvidenceRef(
                    id=ref.id,
                    kind=ref.kind,
                    score=ref.score,
                    role=ref.role,
                )
                for ref in result.memory_context.evidence_refs
            ],
        ),
        rendered_memory_brief=result.rendered_memory_brief,
        system_prompt=result.system_prompt,
        user_prompt=result.user_prompt,
        retrieval_trace_summary=RetrievalTraceSummary(
            query=result.retrieval_trace_summary.query,
            query_type=result.retrieval_trace_summary.query_type,
            attractor_path_len=result.retrieval_trace_summary.attractor_path_len,
            hit_count=result.retrieval_trace_summary.hit_count,
            timeline_count=result.retrieval_trace_summary.timeline_count,
            standalone_count=result.retrieval_trace_summary.standalone_count,
            abstain=result.retrieval_trace_summary.abstain,
            abstention_reason=result.retrieval_trace_summary.abstention_reason,
            asker_id=result.retrieval_trace_summary.asker_id,
            activated_identity=result.retrieval_trace_summary.activated_identity,
        ),
        ingest_result=IngestResponse(**result.ingest_result.__dict__),
        timings=ChatTimings(
            retrieval_ms=result.timings.retrieval_ms,
            generation_ms=result.timings.generation_ms,
            ingest_ms=result.timings.ingest_ms,
            total_ms=result.timings.total_ms,
        ),
        llm_error=result.llm_error,
    )


@app.post("/v2/memory/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """提供关于检索质量的反馈"""
    runtime = get_runtime()
    runtime.feedback(query_text=req.query_text, chosen_id=req.chosen_id, success=req.success)
    return {"ok": True}


# -----------------------------------------------------------------------------
# 扩展API: 一致性、自我叙事、因果
# -----------------------------------------------------------------------------


@app.get("/v2/memory/coherence", response_model=CoherenceResponse)
def check_coherence() -> CoherenceResponse:
    """检查当前单用户记忆的一致性"""
    runtime = get_runtime()
    result = runtime.check_coherence()
    return CoherenceResponse(
        overall_score=result.overall_score,
        conflict_count=result.conflict_count,
        unfinished_story_count=result.unfinished_story_count,
        recommendations=result.recommendations,
    )


@app.get("/v2/memory/self-narrative", response_model=SelfNarrativeResponse)
def get_self_narrative() -> SelfNarrativeResponse:
    """获取当前单用户代理的自我叙事"""
    runtime = get_runtime()
    data = runtime.get_self_narrative()
    return SelfNarrativeResponse(**data)


@app.post("/v2/memory/causal-chain", response_model=CausalChainResponse)
def get_causal_chain(req: CausalChainRequest) -> CausalChainResponse:
    """获取节点的因果链"""
    runtime = get_runtime()
    chain = runtime.get_causal_chain(req.node_id, req.direction)
    return CausalChainResponse(
        chain=[CausalChainItem(**item) for item in chain]
    )


@app.post("/v2/memory/evolve")
def evolve(req: EvolveRequest) -> Dict[str, Any]:
    """触发当前单用户记忆的演化"""
    runtime = get_runtime()
    runtime.evolve()
    return {
        "ok": True,
        "stories": len(runtime.mem.stories),
        "themes": len(runtime.mem.themes),
    }


@app.get("/v2/memory/stats", response_model=MemoryStatsResponse)
def get_stats() -> MemoryStatsResponse:
    """获取当前单用户记忆统计"""
    runtime = get_runtime()
    coherence = runtime.check_coherence()
    narrative = runtime.get_self_narrative()

    return MemoryStatsResponse(
        plot_count=len(runtime.mem.plots),
        story_count=len(runtime.mem.stories),
        theme_count=len(runtime.mem.themes),
        coherence_score=coherence.overall_score,
        self_narrative_coherence=narrative["coherence_score"],
    )
