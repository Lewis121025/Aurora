from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from aurora.interfaces.api.schemas import (
    AcceptedInteractionRequest,
    ChatReplyAcceptedResponse,
    ChatTimings,
    EventStatusResponse,
    EvidenceRef,
    FeedbackRequest,
    HealthResponse,
    IdentityResponse,
    IdentitySnapshotResponse,
    JobStatusResponse,
    MemoryStatsResponse,
    MessagePayload,
    NarrativeSummaryResponse,
    PersistenceReceiptResponse,
    QueryHitResponse,
    QueryRequest,
    QueryResponse,
    ReplyRequest,
    RetrievalTraceSummary,
    StructuredMemoryContext,
)
from aurora.interfaces.messages import message_payload, to_domain_messages
from aurora.runtime.results import StructuredMemoryContext as RuntimeStructuredMemoryContext
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings, DEFAULT_DATA_DIR
from aurora.soul.models import Message
from aurora.system.version import __version__

try:
    from fastapi import FastAPI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("FastAPI未安装。使用以下命令安装: pip install -e '.[api]'") from exc


@lru_cache(maxsize=1)
def get_runtime() -> AuroraRuntime:
    settings = AuroraSettings(data_dir=os.environ.get("AURORA_DATA_DIR", DEFAULT_DATA_DIR))
    return AuroraRuntime(settings=settings)


def shutdown_runtime() -> None:
    cache_info = getattr(get_runtime, "cache_info", None)
    if callable(cache_info) and cache_info().currsize == 0:
        return
    runtime = get_runtime()
    runtime.close()
    cache_clear = getattr(get_runtime, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


@asynccontextmanager
async def app_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    shutdown_runtime()


app = FastAPI(title="Aurora Soul API", version=__version__, lifespan=app_lifespan)


def _identity_response(payload: dict[str, object]) -> IdentitySnapshotResponse:
    return IdentitySnapshotResponse.model_validate(payload)


def _summary_response(payload: dict[str, object]) -> NarrativeSummaryResponse:
    return NarrativeSummaryResponse.model_validate(payload)


def _context_response(context: RuntimeStructuredMemoryContext) -> StructuredMemoryContext:
    identity = context.identity
    summary = context.narrative_summary
    if identity is None or summary is None:
        raise ValueError("Structured memory context is missing identity or narrative summary")
    return StructuredMemoryContext(
        mode=context.mode,
        narrative_pressure=context.narrative_pressure,
        intuition=context.intuition,
        identity=_identity_response(identity.to_state_dict()),
        narrative_summary=_summary_response(summary.to_state_dict()),
        retrieval_hits=context.retrieval_hits,
        overlay_hits=context.overlay_hits,
        evidence_refs=[EvidenceRef(**ref.__dict__) for ref in context.evidence_refs],
    )


def _message_response(message: Message) -> MessagePayload:
    return message_payload(message)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="healthy", version=__version__, timestamp=time.time())


@app.post("/v6/interactions", response_model=PersistenceReceiptResponse)
def interactions(req: AcceptedInteractionRequest) -> PersistenceReceiptResponse:
    event_id = req.event_id or f"evt_turn_{int(time.time() * 1000)}"
    receipt = get_runtime().accept_interaction(
        event_id=event_id,
        session_id=req.session_id,
        messages=_to_domain_messages(req.messages),
        ts=req.ts,
    )
    return PersistenceReceiptResponse(**receipt.__dict__)


@app.post("/v6/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = get_runtime().query(
        messages=_to_domain_messages(req.messages),
        k=req.k,
        session_id=req.session_id,
    )
    return QueryResponse(
        query=result.query,
        attractor_path_len=result.attractor_path_len,
        overlay_hit_count=result.overlay_hit_count,
        hits=[QueryHitResponse(**hit.__dict__) for hit in result.hits],
    )


@app.post("/v6/chat/replies", response_model=ChatReplyAcceptedResponse)
def chat_replies(req: ReplyRequest) -> ChatReplyAcceptedResponse:
    result = get_runtime().respond(
        session_id=req.session_id,
        user_messages=_to_domain_messages(req.messages),
        event_id=req.event_id,
        k=req.k,
        ts=req.ts,
    )
    return ChatReplyAcceptedResponse(
        reply_message=_message_response(result.reply_message),
        event_id=result.event_id,
        memory_context=_context_response(result.memory_context),
        rendered_memory_brief=result.rendered_memory_brief,
        retrieval_trace_summary=RetrievalTraceSummary(**result.retrieval_trace_summary.__dict__),
        persistence=PersistenceReceiptResponse(**result.persistence.__dict__),
        timings=ChatTimings(**result.timings.__dict__),
        llm_error=result.llm_error,
    )


@app.get("/v6/events/{event_id}", response_model=EventStatusResponse)
def event_status(event_id: str) -> EventStatusResponse:
    return EventStatusResponse(**get_runtime().get_event_status(event_id))


@app.get("/v6/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    return JobStatusResponse(**get_runtime().get_job_status(job_id))


@app.post("/v6/feedback")
def feedback(req: FeedbackRequest) -> dict[str, bool]:
    get_runtime().feedback(query_text=req.query_text, chosen_id=req.chosen_id, success=req.success)
    return {"ok": True}


@app.get("/v6/identity", response_model=IdentityResponse)
def identity() -> IdentityResponse:
    report = get_runtime().get_identity()
    identity_payload = report["identity"]
    summary_payload = report["narrative_summary"]
    if not isinstance(identity_payload, dict) or not isinstance(summary_payload, dict):
        raise ValueError("Runtime identity payload is malformed")
    return IdentityResponse(
        identity=_identity_response(identity_payload),
        narrative_summary=_summary_response(summary_payload),
    )


@app.get("/v6/stats", response_model=MemoryStatsResponse)
def stats() -> MemoryStatsResponse:
    return MemoryStatsResponse(**get_runtime().get_stats())
