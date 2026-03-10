from __future__ import annotations

import os
import time
from functools import lru_cache

from aurora.interfaces.api.schemas import (
    ChatTimings,
    ChatTurnResponse,
    EvolveRequest,
    EvidenceRef,
    FeedbackRequest,
    HealthResponse,
    IdentityResponse,
    IdentitySnapshotResponse,
    IngestRequest,
    IngestResponse,
    MemoryStatsResponse,
    NarrativeSummaryResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
    RespondRequest,
    RetrievalTraceSummary,
    StructuredMemoryContext,
)
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings, DEFAULT_DATA_DIR
from aurora.system.version import __version__

try:
    from fastapi import FastAPI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("FastAPI未安装。使用以下命令安装: pip install -e '.[api]'") from exc

app = FastAPI(title="Aurora Soul API", version=__version__)


@lru_cache(maxsize=1)
def get_runtime() -> AuroraRuntime:
    settings = AuroraSettings(data_dir=os.environ.get("AURORA_DATA_DIR", DEFAULT_DATA_DIR))
    return AuroraRuntime(settings=settings)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="healthy", version=__version__, timestamp=time.time())


def _identity_response(payload: dict[str, object]) -> IdentitySnapshotResponse:
    return IdentitySnapshotResponse(**payload)


def _summary_response(payload: dict[str, object]) -> NarrativeSummaryResponse:
    return NarrativeSummaryResponse(**payload)


@app.post("/v4/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    event_id = req.event_id or f"evt_{int(time.time() * 1000)}"
    result = get_runtime().ingest_interaction(
        event_id=event_id,
        session_id=req.session_id,
        user_message=req.user_message,
        agent_message=req.agent_message,
        actors=req.actors,
        context=req.context,
        ts=req.ts,
    )
    return IngestResponse(**result.__dict__)


@app.post("/v4/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = get_runtime().query(text=req.text, k=req.k)
    return QueryResponse(
        query=result.query,
        attractor_path_len=result.attractor_path_len,
        hits=[QueryHit(**hit.__dict__) for hit in result.hits],
    )


@app.post("/v4/respond", response_model=ChatTurnResponse)
def respond(req: RespondRequest) -> ChatTurnResponse:
    result = get_runtime().respond(
        session_id=req.session_id,
        user_message=req.user_message,
        event_id=req.event_id,
        context=req.context,
        actors=req.actors,
        k=req.k,
        ts=req.ts,
    )
    identity = result.memory_context.identity
    summary = result.memory_context.narrative_summary
    return ChatTurnResponse(
        reply=result.reply,
        event_id=result.event_id,
        memory_context=StructuredMemoryContext(
            mode=result.memory_context.mode,
            narrative_pressure=result.memory_context.narrative_pressure,
            intuition=result.memory_context.intuition,
            identity=_identity_response(identity.to_state_dict()),
            narrative_summary=_summary_response(summary.to_state_dict()),
            retrieval_hits=result.memory_context.retrieval_hits,
            evidence_refs=[
                EvidenceRef(**ref.__dict__) for ref in result.memory_context.evidence_refs
            ],
        ),
        rendered_memory_brief=result.rendered_memory_brief,
        system_prompt=result.system_prompt,
        user_prompt=result.user_prompt,
        retrieval_trace_summary=RetrievalTraceSummary(**result.retrieval_trace_summary.__dict__),
        ingest_result=IngestResponse(**result.ingest_result.__dict__),
        timings=ChatTimings(**result.timings.__dict__),
        llm_error=result.llm_error,
    )


@app.post("/v4/feedback")
def feedback(req: FeedbackRequest) -> dict[str, bool]:
    get_runtime().feedback(query_text=req.query_text, chosen_id=req.chosen_id, success=req.success)
    return {"ok": True}


@app.post("/v4/evolve")
def evolve(req: EvolveRequest) -> dict[str, object]:
    dream_plots = get_runtime().evolve(dreams=req.dreams)
    return {"ok": True, "dreams": len(dream_plots)}


@app.get("/v4/identity", response_model=IdentityResponse)
def identity() -> IdentityResponse:
    report = get_runtime().get_identity()
    return IdentityResponse(
        identity=_identity_response(report["identity"]),
        narrative_summary=_summary_response(report["narrative_summary"]),
    )


@app.get("/v4/stats", response_model=MemoryStatsResponse)
def stats() -> MemoryStatsResponse:
    return MemoryStatsResponse(**get_runtime().get_stats())
