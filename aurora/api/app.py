from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from aurora.config import AuroraSettings
from aurora.hub import AuroraHub

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise RuntimeError("FastAPI is not installed. Install with: pip install -e '.[api]'") from e


settings = AuroraSettings(data_dir=os.environ.get("AURORA_DATA_DIR", "./data"))
hub = AuroraHub(settings=settings)

app = FastAPI(title="AURORA Memory API", version="0.1.0")


class IngestRequest(BaseModel):
    event_id: str
    user_id: str
    session_id: str
    user_message: str
    agent_message: str
    actors: Optional[List[str]] = None
    context: Optional[str] = None
    ts: Optional[float] = None


class IngestResponse(BaseModel):
    event_id: str
    plot_id: str
    story_id: Optional[str]
    encoded: bool
    tension: float
    surprise: float
    pred_error: float
    redundancy: float


class QueryRequest(BaseModel):
    user_id: str
    text: str
    k: int = Field(default=8, ge=1, le=50)


class QueryHitModel(BaseModel):
    id: str
    kind: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    query: str
    attractor_path_len: int
    hits: List[QueryHitModel]


class FeedbackRequest(BaseModel):
    user_id: str
    query_text: str
    chosen_id: str
    success: bool


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "ts": time.time()}


@app.post("/v1/memory/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    t = hub.tenant(req.user_id)
    r = t.ingest_interaction(
        event_id=req.event_id,
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
    t = hub.tenant(req.user_id)
    r = t.query(text=req.text, k=req.k)
    return QueryResponse(
        query=r.query,
        attractor_path_len=r.attractor_path_len,
        hits=[QueryHitModel(**h.__dict__) for h in r.hits],
    )


@app.post("/v1/memory/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    t = hub.tenant(req.user_id)
    t.feedback(query_text=req.query_text, chosen_id=req.chosen_id, success=req.success)
    return {"ok": True}


# -----------------------------------------------------------------------------
# Extended APIs: Coherence, Self-Narrative, Causal
# -----------------------------------------------------------------------------

class CoherenceResponse(BaseModel):
    overall_score: float
    conflict_count: int
    unfinished_story_count: int
    recommendations: List[str]


@app.get("/v1/memory/coherence/{user_id}", response_model=CoherenceResponse)
def check_coherence(user_id: str) -> CoherenceResponse:
    """Check memory coherence for a user"""
    t = hub.tenant(user_id)
    result = t.check_coherence()
    return CoherenceResponse(
        overall_score=result.overall_score,
        conflict_count=result.conflict_count,
        unfinished_story_count=result.unfinished_story_count,
        recommendations=result.recommendations,
    )


class SelfNarrativeResponse(BaseModel):
    identity_statement: str
    identity_narrative: str
    capability_narrative: str
    core_values: List[str]
    coherence_score: float
    capabilities: Dict[str, Dict[str, Any]]
    relationships: Dict[str, Dict[str, Any]]
    unresolved_tensions: List[str]
    full_narrative: str


@app.get("/v1/memory/self-narrative/{user_id}", response_model=SelfNarrativeResponse)
def get_self_narrative(user_id: str) -> SelfNarrativeResponse:
    """Get self-narrative for a user's agent"""
    t = hub.tenant(user_id)
    data = t.get_self_narrative()
    return SelfNarrativeResponse(**data)


class CausalChainRequest(BaseModel):
    user_id: str
    node_id: str
    direction: str = Field(default="ancestors", pattern="^(ancestors|descendants)$")


class CausalChainItem(BaseModel):
    node_id: str
    strength: float


class CausalChainResponse(BaseModel):
    chain: List[CausalChainItem]


@app.post("/v1/memory/causal-chain", response_model=CausalChainResponse)
def get_causal_chain(req: CausalChainRequest) -> CausalChainResponse:
    """Get causal chain for a node"""
    t = hub.tenant(req.user_id)
    chain = t.get_causal_chain(req.node_id, req.direction)
    return CausalChainResponse(
        chain=[CausalChainItem(**item) for item in chain]
    )


class EvolveRequest(BaseModel):
    user_id: str


@app.post("/v1/memory/evolve")
def evolve(req: EvolveRequest) -> Dict[str, Any]:
    """Trigger evolution for a user's memory"""
    t = hub.tenant(req.user_id)
    t.evolve()
    return {
        "ok": True,
        "stories": len(t.mem.stories),
        "themes": len(t.mem.themes),
    }


class MemoryStatsResponse(BaseModel):
    plot_count: int
    story_count: int
    theme_count: int
    coherence_score: float
    self_narrative_coherence: float


@app.get("/v1/memory/stats/{user_id}", response_model=MemoryStatsResponse)
def get_stats(user_id: str) -> MemoryStatsResponse:
    """Get memory statistics for a user"""
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
