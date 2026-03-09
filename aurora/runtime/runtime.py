from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from aurora.system.errors import ConfigurationError
from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.storage.doc_store import Document, SQLiteDocStore
from aurora.integrations.storage.event_log import Event, SQLiteEventLog
from aurora.integrations.storage.snapshot import Snapshot, SnapshotStore
from aurora.privacy.pii import redact
from aurora.runtime.bootstrap import (
    check_embedding_api_keys,
    create_embedding_provider,
    create_llm_provider,
    create_meaning_extractor,
    create_memory,
)
from aurora.runtime.response_context import ResponseContextBuilder
from aurora.runtime.results import (
    ChatTimings,
    ChatTurnResult,
    IngestResult,
    QueryHit,
    QueryResult,
    StructuredMemoryContext,
)
from aurora.runtime.settings import AuroraSettings
from aurora.soul.engine import AuroraSoul

logger = logging.getLogger(__name__)

RUNTIME_SCHEMA_VERSION = "aurora-runtime-v3"


class AuroraRuntime:
    def __init__(self, *, settings: AuroraSettings, llm: Optional[LLMProvider] = None):
        self.settings = settings
        self.llm = llm or create_llm_provider(settings)

        check_embedding_api_keys()
        self._lock = threading.RLock()

        os.makedirs(self.settings.data_dir, exist_ok=True)
        self.event_log = SQLiteEventLog(os.path.join(self.settings.data_dir, self.settings.event_log_filename))
        self.doc_store = SQLiteDocStore(os.path.join(self.settings.data_dir, "docs.sqlite3"))
        self.snapshots = SnapshotStore(os.path.join(self.settings.data_dir, self.settings.snapshot_dirname))

        self.last_seq = 0
        self.mem = self._load_or_init()
        self.response_contexts = ResponseContextBuilder(memory=self.mem)

        self._ensure_v3_doc_store()
        self._replay()

    def _load_or_init(self) -> AuroraSoul:
        try:
            latest = self.snapshots.latest()
        except ValueError as exc:
            raise ConfigurationError(str(exc).replace("V2", "Aurora Soul V3")) from exc
        if latest is not None:
            _seq, snap = latest
            self.last_seq = snap.last_seq
            embedder = create_embedding_provider(self.settings)
            extractor = create_meaning_extractor(settings=self.settings, llm=self.llm)
            try:
                return AuroraSoul.from_state_dict(snap.state, embedder=embedder, extractor=extractor)
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
        return create_memory(settings=self.settings, llm=self.llm)

    def _ensure_v3_doc_store(self) -> None:
        for kind in ("plot", "ingest_result"):
            for doc in self.doc_store.iter_kind(kind=kind, limit=5):
                if doc.body.get("runtime_schema_version") != RUNTIME_SCHEMA_VERSION:
                    raise ConfigurationError(
                        "Detected legacy documents in docs.sqlite3. Aurora Soul requires a fresh data directory."
                    )

    def _ensure_v3_event_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get("runtime_schema_version") != RUNTIME_SCHEMA_VERSION:
            raise ConfigurationError(
                "Detected legacy event payloads. Aurora Soul requires a fresh data directory."
            )

    def _replay(self) -> None:
        for seq, event in self.event_log.iter_events(after_seq=self.last_seq):
            if event.type != "interaction":
                continue
            self._ensure_v3_event_payload(event.payload)
            self._apply_interaction(
                event_id=event.id,
                user_message=str(event.payload.get("user_message", "")),
                agent_message=str(event.payload.get("agent_message", "")),
                context=event.payload.get("context"),
                actors=event.payload.get("actors"),
                ts=event.ts,
                persist=False,
            )
            self.last_seq = seq

    def ingest_interaction(
        self,
        *,
        event_id: str,
        session_id: str,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]] = None,
        context: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> IngestResult:
        event_ts = ts or time.time()
        with self._lock:
            existing_seq = self.event_log.get_seq_by_id(event_id)
            if existing_seq is not None:
                return self._load_existing_ingest_result(event_id)

            if self.settings.pii_redaction_enabled:
                user_message = redact(user_message).redacted_text
                agent_message = redact(agent_message).redacted_text

            payload = self._build_event_payload(
                user_message=user_message,
                agent_message=agent_message,
                actors=actors,
                context=context,
            )
            seq = self.event_log.append(
                Event(
                    id=event_id,
                    ts=event_ts,
                    session_id=session_id,
                    type="interaction",
                    payload=payload,
                )
            )
            result = self._apply_interaction(
                event_id=event_id,
                user_message=user_message,
                agent_message=agent_message,
                context=context,
                actors=actors,
                ts=event_ts,
                persist=True,
            )
            self.last_seq = max(self.last_seq, seq)
            if self.settings.snapshot_every_events > 0 and self.last_seq % self.settings.snapshot_every_events == 0:
                self._snapshot()
            return result

    def _build_event_payload(
        self,
        *,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "user_message": user_message,
            "agent_message": agent_message,
            "actors": list(actors) if actors else ["user", "agent"],
            "context": context,
        }

    def _interaction_text(self, *, user_message: str, agent_message: str) -> str:
        return f"USER: {user_message}\nAURORA: {agent_message}"

    def _apply_interaction(
        self,
        *,
        event_id: str,
        user_message: str,
        agent_message: str,
        context: Optional[str],
        actors: Optional[Sequence[str]],
        ts: float,
        persist: bool,
    ) -> IngestResult:
        plot = self.mem.ingest(
            self._interaction_text(user_message=user_message, agent_message=agent_message),
            actors=actors or ("user", "agent"),
            context_text=context or user_message,
            source="wake",
            ts=ts,
        )
        result = IngestResult(
            event_id=event_id,
            plot_id=plot.id,
            story_id=plot.story_id,
            phase=self.mem.identity.phase,
            source=plot.source,
            tension=float(plot.tension),
            contradiction=float(plot.contradiction),
            active_energy=float(self.mem.identity.active_energy),
            repressed_energy=float(self.mem.identity.repressed_energy),
        )
        if persist:
            self._persist_interaction_artifacts(event_id=event_id, ts=ts, plot=plot, result=result)
        return result

    def _persist_interaction_artifacts(self, *, event_id: str, ts: float, plot: Any, result: IngestResult) -> None:
        self.doc_store.upsert(
            Document(
                id=plot.id,
                kind="plot",
                ts=ts,
                body={
                    "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
                    "plot_id": plot.id,
                    "story_id": plot.story_id,
                    "text": plot.text,
                    "source": plot.source,
                    "phase": self.mem.identity.phase,
                    "tags": list(plot.frame.tags),
                    "tension": float(plot.tension),
                    "contradiction": float(plot.contradiction),
                },
            )
        )
        self.doc_store.upsert(
            Document(
                id=f"ingest:{event_id}",
                kind="ingest_result",
                ts=ts,
                body={
                    "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
                    "event_id": result.event_id,
                    "plot_id": result.plot_id,
                    "story_id": result.story_id,
                    "phase": result.phase,
                    "source": result.source,
                    "tension": result.tension,
                    "contradiction": result.contradiction,
                    "active_energy": result.active_energy,
                    "repressed_energy": result.repressed_energy,
                },
            )
        )

    def _snapshot(self) -> None:
        snap = Snapshot(last_seq=self.last_seq, state=self.mem.to_state_dict())
        self.snapshots.save(snap)

    def _load_existing_ingest_result(self, event_id: str) -> IngestResult:
        doc = self.doc_store.get(f"ingest:{event_id}")
        if doc is None:
            return IngestResult(
                event_id=event_id,
                plot_id="",
                story_id=None,
                phase=self.mem.identity.phase,
                source="wake",
                tension=0.0,
                contradiction=0.0,
                active_energy=float(self.mem.identity.active_energy),
                repressed_energy=float(self.mem.identity.repressed_energy),
            )
        body = doc.body
        return IngestResult(
            event_id=event_id,
            plot_id=str(body.get("plot_id", "")),
            story_id=body.get("story_id"),
            phase=str(body.get("phase", self.mem.identity.phase)),
            source=str(body.get("source", "wake")),
            tension=float(body.get("tension", 0.0)),
            contradiction=float(body.get("contradiction", 0.0)),
            active_energy=float(body.get("active_energy", self.mem.identity.active_energy)),
            repressed_energy=float(body.get("repressed_energy", self.mem.identity.repressed_energy)),
        )

    def build_response_context(self, *, user_message: str, k: int = 6) -> tuple[StructuredMemoryContext, Any]:
        with self._lock:
            trace = self.mem.query(user_message, k=k)
            context = self.response_contexts.build(trace=trace, max_items=k)
            summary = self.response_contexts.summarize_trace(trace)
        return context, summary

    def respond(
        self,
        *,
        session_id: str,
        user_message: str,
        event_id: Optional[str] = None,
        context: Optional[str] = None,
        actors: Optional[Sequence[str]] = None,
        k: int = 6,
        ts: Optional[float] = None,
    ) -> ChatTurnResult:
        total_started = time.perf_counter()
        resolved_event_id = event_id or f"evt_resp_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        retrieval_started = time.perf_counter()
        memory_context, trace_summary = self.build_response_context(user_message=user_message, k=k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        prompt = self.response_contexts.build_prompt(
            user_message=user_message,
            rendered_memory_brief=self.response_contexts.render_memory_brief(memory_context),
        )

        generation_started = time.perf_counter()
        llm_error: Optional[str] = None
        reply = self._generate_reply(prompt=prompt, memory_context=memory_context, user_message=user_message)
        if reply.startswith("__LLM_ERROR__:"):
            llm_error = reply.removeprefix("__LLM_ERROR__:")
            reply = self._build_response_fallback(memory_context=memory_context, user_message=user_message)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0

        ingest_started = time.perf_counter()
        ingest_result = self.ingest_interaction(
            event_id=resolved_event_id,
            session_id=session_id,
            user_message=user_message,
            agent_message=reply,
            actors=actors,
            context=context,
            ts=ts,
        )
        ingest_ms = (time.perf_counter() - ingest_started) * 1000.0

        total_ms = (time.perf_counter() - total_started) * 1000.0
        return ChatTurnResult(
            reply=reply,
            event_id=resolved_event_id,
            memory_context=memory_context,
            rendered_memory_brief=prompt.rendered_memory_brief,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            retrieval_trace_summary=trace_summary,
            ingest_result=ingest_result,
            timings=ChatTimings(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                ingest_ms=ingest_ms,
                total_ms=total_ms,
            ),
            llm_error=llm_error,
        )

    def _generate_reply(self, *, prompt: Any, memory_context: StructuredMemoryContext, user_message: str) -> str:
        try:
            return self.llm.complete(
                prompt.user_prompt,
                system=prompt.system_prompt,
                temperature=0.3,
                max_tokens=400,
                timeout_s=self.settings.llm_timeout,
                max_retries=max(1, min(2, int(self.settings.llm_max_retries))),
            ).strip()
        except Exception as exc:
            return f"__LLM_ERROR__:{exc}"

    @staticmethod
    def _build_response_fallback(*, memory_context: StructuredMemoryContext, user_message: str) -> str:
        if memory_context.narrative_summary is not None:
            return (
                "这轮语言模型调用失败了。"
                f"我现在只稳稳抓住了一个内部状态：{memory_context.narrative_summary.core_statement}"
            )
        return (
            "这轮语言模型调用失败了，而且当前没有足够稳定的 soul-memory 线索来回答。"
            f"我会先把这轮对话记住：{user_message[:80]}"
        )

    def query(self, *, text: str, k: int = 8) -> QueryResult:
        with self._lock:
            trace = self.mem.query(text, k=k)
            hits: List[QueryHit] = []
            for node_id, score, kind in trace.ranked:
                snippet = self._snippet_for(node_id=node_id, kind=kind)
                metadata: Optional[Dict[str, str]] = None
                if kind == "plot":
                    plot = self.mem.plots.get(node_id)
                    if plot is not None:
                        metadata = {"source": plot.source, "story_id": plot.story_id or ""}
                hits.append(QueryHit(id=node_id, kind=kind, score=float(score), snippet=snippet, metadata=metadata))
        return QueryResult(query=text, attractor_path_len=len(trace.attractor_path), hits=hits)

    def _snippet_for(self, *, node_id: str, kind: str) -> str:
        if kind == "plot":
            plot = self.mem.plots.get(node_id)
            return "" if plot is None else plot.text[:240]
        if kind == "story":
            story = self.mem.stories.get(node_id)
            if story is None:
                return ""
            return f"plots={len(story.plot_ids)} status={story.status} unresolved={story.unresolved_energy:.3f}"
        theme = self.mem.themes.get(node_id)
        if theme is None:
            return ""
        return (theme.name or theme.description)[:240]

    def feedback(self, *, query_text: str, chosen_id: str, success: bool) -> None:
        with self._lock:
            self.mem.feedback_retrieval(query_text=query_text, chosen_id=chosen_id, success=success)

    def evolve(self, *, dreams: Optional[int] = None) -> List[Any]:
        with self._lock:
            return self.mem.evolve(dreams=dreams or self.settings.dreams_per_evolve)

    def get_identity(self) -> Dict[str, Any]:
        with self._lock:
            identity = self.mem.snapshot_identity()
            summary = self.mem.narrative_summary()
            return {
                "identity": identity.to_state_dict(),
                "narrative_summary": summary.to_state_dict(),
            }

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "plot_count": len(self.mem.plots),
                "story_count": len(self.mem.stories),
                "theme_count": len(self.mem.themes),
                "phase": self.mem.identity.phase,
                "pressure": self.mem.identity.narrative_pressure(),
                "dream_count": self.mem.identity.dream_count,
                "repair_count": self.mem.identity.repair_count,
                "active_energy": self.mem.identity.active_energy,
                "repressed_energy": self.mem.identity.repressed_energy,
            }
