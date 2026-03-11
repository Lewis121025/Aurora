"""Aurora V6 runtime: event-sourced CQRS with multimodal interaction payloads."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.storage.runtime_store import (
    SQLiteRuntimeStore,
    StoredEvent,
    StoredJob,
)
from aurora.integrations.storage.snapshot import Snapshot, SnapshotStore
from aurora.runtime.bootstrap import (
    check_embedding_api_keys,
    create_content_embedding_provider,
    create_llm_provider,
    create_meaning_provider,
    create_memory,
    create_narrative_provider,
    create_query_analyzer,
    create_text_embedding_provider,
)
from aurora.runtime.response_context import ResponseContextBuilder
from aurora.runtime.results import (
    ChatStreamEvent,
    ChatTimings,
    ChatTurnResult,
    PersistenceReceipt,
    QueryHit,
    QueryResult,
)
from aurora.runtime.settings import AuroraSettings
from aurora.soul.engine import AuroraSoul
from aurora.soul.models import Message, TextPart, messages_to_text
from aurora.system.errors import ConfigurationError

logger = logging.getLogger(__name__)

RUNTIME_SCHEMA_VERSION = "aurora-runtime-v6"
SYSTEM_SESSION_ID = "__aurora_system__"


class AuroraRuntime:
    def __init__(self, *, settings: AuroraSettings, llm: Optional[LLMProvider] = None):
        self.settings = settings
        self.llm = llm if llm is not None else create_llm_provider(settings)

        check_embedding_api_keys()
        self._lock = threading.RLock()
        self._closed = False
        self._stop = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._worker_threads: List[threading.Thread] = []
        self._loaded_snapshot_seq = 0
        self._last_projected_seq = 0

        os.makedirs(self.settings.data_dir, exist_ok=True)
        self.store = SQLiteRuntimeStore(
            os.path.join(self.settings.data_dir, self.settings.runtime_db_filename)
        )
        self.snapshots = SnapshotStore(
            os.path.join(self.settings.data_dir, self.settings.snapshot_dirname)
        )
        self.mem = self._load_or_init()
        self.response_contexts = ResponseContextBuilder(memory=self.mem)

        self._replay_projected_events()
        self._start_workers()
        self._start_scheduler()

    def _load_or_init(self) -> AuroraSoul:
        try:
            latest = self.snapshots.latest()
        except ValueError as exc:
            raise ConfigurationError(str(exc).replace("V2", "Aurora Soul V6")) from exc
        if latest is None:
            return create_memory(settings=self.settings, llm=self.llm)
        _seq, snap = latest
        self._loaded_snapshot_seq = snap.last_seq
        self._last_projected_seq = snap.last_seq
        event_embedder = create_content_embedding_provider(self.settings)
        axis_embedder = create_text_embedding_provider(self.settings)
        meaning_provider = create_meaning_provider(settings=self.settings, llm=self.llm)
        narrator = create_narrative_provider(settings=self.settings, llm=self.llm)
        query_analyzer = create_query_analyzer(settings=self.settings, llm=self.llm)
        try:
            return AuroraSoul.from_state_dict(
                snap.state,
                event_embedder=event_embedder,
                axis_embedder=axis_embedder,
                meaning_provider=meaning_provider,
                narrator=narrator,
                query_analyzer=query_analyzer,
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc

    def _snapshot(self) -> None:
        self.snapshots.save(Snapshot(last_seq=self._last_projected_seq, state=self.mem.to_state_dict()))

    def _maybe_snapshot(self) -> None:
        threshold = int(self.settings.snapshot_every_projected_events)
        if threshold <= 0:
            return
        if self._last_projected_seq <= 0:
            return
        if self._last_projected_seq % threshold == 0:
            self._snapshot()

    def _replay_projected_events(self) -> None:
        target_seq = int(self.store.get_runtime_state().get("last_projected_seq", 0))
        if target_seq <= self._loaded_snapshot_seq:
            self._last_projected_seq = max(self._last_projected_seq, target_seq)
            return
        for event in self.store.iter_projected_events(after_seq=self._loaded_snapshot_seq):
            with self._lock:
                self._apply_event_to_memory(event)
            self._last_projected_seq = max(self._last_projected_seq, event.seq)

    def _start_workers(self) -> None:
        worker_count = max(1, int(self.settings.worker_count))
        for index in range(worker_count):
            worker_id = f"aurora-worker-{index + 1}"
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=worker_id,
                daemon=True,
            )
            thread.start()
            self._worker_threads.append(thread)

    def _start_scheduler(self) -> None:
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="aurora-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _scheduler_loop(self) -> None:
        next_evolve_ts = time.time() + max(1.0, float(self.settings.evolve_every_seconds))
        next_fade_ts = time.time() + max(1.0, float(self.settings.fade_every_seconds))
        sleep_s = max(0.05, float(self.settings.job_poll_interval_ms) / 1000.0)
        while not self._stop.wait(sleep_s):
            now = time.time()
            if now >= next_evolve_ts:
                bucket = int(now // max(1.0, float(self.settings.evolve_every_seconds)))
                self.store.enqueue_job(
                    job_type="run_evolve",
                    payload={"dreams": int(self.settings.dreams_per_evolve)},
                    max_attempts=self.settings.job_retry_limit,
                    dedupe_key=f"run_evolve:{bucket}",
                )
                next_evolve_ts = now + max(1.0, float(self.settings.evolve_every_seconds))
            if now >= next_fade_ts:
                bucket = int(now // max(1.0, float(self.settings.fade_every_seconds)))
                self.store.enqueue_job(
                    job_type="run_fading",
                    payload={
                        "cold_age_s": float(self.settings.fade_cold_age_s),
                        "mass_threshold": float(self.settings.fade_mass_threshold),
                        "group_min_size": int(self.settings.fade_group_min_size),
                    },
                    max_attempts=self.settings.job_retry_limit,
                    dedupe_key=f"run_fading:{bucket}",
                )
                next_fade_ts = now + max(1.0, float(self.settings.fade_every_seconds))

    def _worker_loop(self, worker_id: str) -> None:
        poll_s = max(0.05, float(self.settings.job_poll_interval_ms) / 1000.0)
        while not self._stop.wait(poll_s):
            jobs = self.store.claim_jobs(
                worker_id=worker_id,
                limit=1,
                lease_seconds=float(self.settings.job_lease_seconds),
            )
            if not jobs:
                continue
            for job in jobs:
                try:
                    self._process_job(job)
                except Exception as exc:  # pragma: no cover
                    logger.exception("worker failed processing job %s", job.job_id)
                    if job.event_id:
                        self.store.mark_projection_failed(event_id=job.event_id, error=str(exc))
                    self.store.fail_job(job_id=job.job_id, error=str(exc))

    def _process_job(self, job: StoredJob) -> None:
        if job.job_type.startswith("project_"):
            self._process_projection_job(job)
            return
        if job.job_type == "run_evolve":
            self._process_run_evolve(job)
            return
        if job.job_type == "run_fading":
            self._process_run_fading(job)
            return
        raise ValueError(f"Unknown job type: {job.job_type}")

    def _process_projection_job(self, job: StoredJob) -> None:
        if job.event_id is None:
            raise RuntimeError(f"Projection job {job.job_id} missing event_id")
        projection = self.store.get_projection_status(job.event_id)
        if projection is not None and projection.status == "projected":
            self.store.complete_job(job.job_id)
            return
        event = self.store.get_event(job.event_id)
        if event is None:
            raise RuntimeError(f"Missing event {job.event_id}")
        self.store.mark_projecting(job.event_id)
        with self._lock:
            node_id, node_kind = self._apply_event_to_memory(event)
            self._last_projected_seq = max(self._last_projected_seq, event.seq)
            self._maybe_snapshot()
        self.store.mark_projected(
            event_id=job.event_id,
            event_seq=event.seq,
            node_id=node_id,
            node_kind=node_kind,
        )
        self.store.complete_job(job.job_id)

    def _process_run_evolve(self, job: StoredJob) -> None:
        dreams = int(job.payload.get("dreams", self.settings.dreams_per_evolve))
        with self._lock:
            repair_events = self.mem.plan_repair_events(limit=1)
            dream_events = self.mem.plan_dream_events(n=dreams)
        for payload in repair_events:
            event_id = f"evt_repair_{uuid.uuid4().hex}"
            payload = {**payload, "runtime_schema_version": RUNTIME_SCHEMA_VERSION, "search_text": payload["text"]}
            self.store.append_event_and_enqueue(
                event_id=event_id,
                event_type="repair",
                session_id=SYSTEM_SESSION_ID,
                ts=float(payload.get("ts", time.time())),
                payload=payload,
                search_text=str(payload["text"]),
                job_type="project_repair",
                job_payload={},
                max_attempts=self.settings.job_retry_limit,
                dedupe_key=event_id,
            )
        for payload in dream_events:
            event_id = f"evt_dream_{uuid.uuid4().hex}"
            payload = {**payload, "runtime_schema_version": RUNTIME_SCHEMA_VERSION, "search_text": payload["text"]}
            self.store.append_event_and_enqueue(
                event_id=event_id,
                event_type="dream",
                session_id=SYSTEM_SESSION_ID,
                ts=float(payload.get("ts", time.time())),
                payload=payload,
                search_text=str(payload["text"]),
                job_type="project_dream",
                job_payload={},
                max_attempts=self.settings.job_retry_limit,
                dedupe_key=event_id,
            )
        self.store.update_runtime_state(last_evolve_ts=time.time())
        self.store.complete_job(job.job_id)

    def _process_run_fading(self, job: StoredJob) -> None:
        with self._lock:
            plans = self.mem.plan_compaction_events(
                cold_age_s=float(job.payload.get("cold_age_s", self.settings.fade_cold_age_s)),
                mass_threshold=float(job.payload.get("mass_threshold", self.settings.fade_mass_threshold)),
                group_min_size=int(job.payload.get("group_min_size", self.settings.fade_group_min_size)),
            )
        for payload in plans:
            event_id = f"evt_compaction_{uuid.uuid4().hex}"
            payload = {**payload, "runtime_schema_version": RUNTIME_SCHEMA_VERSION, "search_text": payload["text"]}
            self.store.append_event_and_enqueue(
                event_id=event_id,
                event_type="compaction",
                session_id=SYSTEM_SESSION_ID,
                ts=float(payload.get("created_ts", time.time())),
                payload=payload,
                search_text=str(payload["text"]),
                job_type="project_compaction",
                job_payload={},
                max_attempts=self.settings.job_retry_limit,
                dedupe_key=event_id,
            )
        self.store.update_runtime_state(last_fade_ts=time.time())
        self.store.complete_job(job.job_id)

    def _apply_event_to_memory(self, event: StoredEvent) -> Tuple[Optional[str], Optional[str]]:
        payload = dict(event.payload)
        version = str(payload.get("runtime_schema_version", ""))
        if version != RUNTIME_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported runtime event schema version: {version!r}. "
                f"Supported version is {RUNTIME_SCHEMA_VERSION!r}."
            )
        payload.setdefault("ts", event.ts)
        node_id = self.mem.apply_projected_event(
            event_type=event.event_type,
            event_id=event.event_id,
            payload=payload,
        )
        node_kind = {"compaction": "summary", "interaction": "plot", "dream": "plot", "repair": "plot"}[
            event.event_type
        ]
        return node_id, node_kind

    def accept_interaction(
        self,
        *,
        event_id: str,
        session_id: str,
        messages: Sequence[Message],
        ts: Optional[float] = None,
    ) -> PersistenceReceipt:
        event_ts = float(ts or time.time())
        search_text = messages_to_text(messages, include_image_uris=True)
        payload = {
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "messages": [message.to_state_dict() for message in messages],
            "ts": event_ts,
            "search_text": search_text,
        }
        event, job = self.store.append_event_and_enqueue(
            event_id=event_id,
            event_type="interaction",
            session_id=session_id,
            ts=event_ts,
            payload=payload,
            search_text=search_text,
            job_type="project_interaction",
            job_payload={},
            max_attempts=self.settings.job_retry_limit,
            dedupe_key=event_id,
        )
        return PersistenceReceipt(
            event_id=event.event_id,
            job_id=job.job_id,
            status="accepted",
            accepted_at=event.ts,
            projection_status="accepted",
        )

    def _graph_query_hits(self, *, messages: Sequence[Message], k: int) -> Tuple[List[QueryHit], Any]:
        with self._lock:
            trace = self.mem.query(messages, k=max(k * 2, 8))
            hits: List[QueryHit] = []
            for node_id, score, kind in trace.ranked:
                snippet, metadata = self._snippet_for(node_id=node_id, kind=kind)
                hits.append(
                    QueryHit(
                        id=node_id,
                        kind=kind,  # type: ignore[arg-type]
                        score=float(score),
                        snippet=snippet,
                        metadata=metadata,
                    )
                )
        return hits, trace

    def _merge_hits(
        self,
        *,
        graph_hits: List[QueryHit],
        overlay_hits: List[QueryHit],
        limit: int,
    ) -> List[QueryHit]:
        graph_max = max((hit.score for hit in graph_hits), default=1.0) or 1.0
        represented_event_ids = {
            event_id
            for hit in graph_hits
            for event_id in self._graph_hit_event_ids(hit)
        }
        merged: List[QueryHit] = []
        for hit in graph_hits:
            merged.append(
                QueryHit(
                    id=hit.id,
                    kind=hit.kind,
                    score=min(1.5, hit.score / graph_max),
                    snippet=hit.snippet,
                    metadata=hit.metadata,
                )
            )
        for hit in overlay_hits:
            if hit.id in represented_event_ids:
                continue
            merged.append(hit)
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:limit]

    def _graph_hit_event_ids(self, hit: QueryHit) -> List[str]:
        if hit.kind == "plot":
            plot = self.mem.plots.get(hit.id)
            return [plot.event_id] if plot is not None and plot.event_id else []
        if hit.kind == "summary":
            summary = self.mem.summaries.get(hit.id)
            return list(summary.source_event_ids) if summary is not None else []
        return []

    def _overlay_query_hits(
        self,
        *,
        query_text: str,
        k: int,
        session_id: Optional[str],
    ) -> List[QueryHit]:
        hits = self.store.search_overlay_events(
            query_text=query_text,
            limit=max(k * 2, self.settings.overlay_search_limit),
            session_id=session_id,
        )
        return [
            QueryHit(
                id=hit.event_id,
                kind="event",
                score=float(hit.score),
                snippet=hit.snippet,
                metadata={"projection_status": hit.projection_status, "session_id": hit.session_id},
            )
            for hit in hits
        ]

    def query(
        self,
        *,
        messages: Sequence[Message],
        k: int = 8,
        session_id: Optional[str] = None,
    ) -> QueryResult:
        query_text = self.mem.meaning_provider.project(messages)
        graph_hits, trace = self._graph_query_hits(messages=messages, k=k)
        overlay_hits = self._overlay_query_hits(query_text=query_text, k=k, session_id=session_id)
        merged = self._merge_hits(graph_hits=graph_hits, overlay_hits=overlay_hits, limit=k)
        return QueryResult(
            query=query_text,
            attractor_path_len=len(trace.attractor_path),
            overlay_hit_count=len(overlay_hits),
            hits=merged,
        )

    def build_response_context(
        self,
        *,
        session_id: str,
        user_messages: Sequence[Message],
        k: int = 6,
    ) -> tuple[Any, Any]:
        query_text = self.mem.meaning_provider.project(user_messages)
        graph_hits, trace = self._graph_query_hits(messages=user_messages, k=k)
        overlay_hits = self._overlay_query_hits(
            query_text=query_text,
            k=k,
            session_id=session_id,
        )
        merged = self._merge_hits(graph_hits=graph_hits, overlay_hits=overlay_hits, limit=k)
        context = self.response_contexts.build(hits=merged, max_items=k)
        summary = self.response_contexts.trace_summary(
            query=query_text,
            attractor_path_len=len(trace.attractor_path),
            graph_kinds=[kind for _, _, kind in trace.ranked],
            overlay_hit_count=len(overlay_hits),
            query_type=trace.query_type.name if trace.query_type is not None else None,
            time_relation=trace.time_range.relation if trace.time_range is not None else None,
            time_start=trace.time_range.start if trace.time_range is not None else None,
            time_end=trace.time_range.end if trace.time_range is not None else None,
            time_anchor_event=trace.time_range.anchor_event if trace.time_range is not None else None,
        )
        return context, summary

    def _response_llm_timeout_s(self) -> float:
        return min(float(self.settings.llm_timeout), 10.0)

    def _response_llm_max_retries(self) -> int:
        return 1

    def respond_stream(
        self,
        *,
        session_id: str,
        user_messages: Sequence[Message],
        event_id: Optional[str] = None,
        k: int = 6,
        ts: Optional[float] = None,
    ) -> Iterator[ChatStreamEvent]:
        total_started = time.perf_counter()
        resolved_event_id = event_id or f"evt_turn_{uuid.uuid4().hex}"

        yield ChatStreamEvent(kind="status", stage="retrieval", text="正在检索记忆")
        retrieval_started = time.perf_counter()
        memory_context, trace_summary = self.build_response_context(
            session_id=session_id,
            user_messages=user_messages,
            k=k,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        prompt = self.response_contexts.build_prompt(
            user_messages=list(user_messages),
            rendered_memory_brief=self.response_contexts.render_memory_brief(memory_context),
        )

        yield ChatStreamEvent(kind="status", stage="generation", text="正在生成回复")
        generation_started = time.perf_counter()
        reply_parts: List[str] = []
        assert self.llm is not None
        for chunk in self.llm.stream_complete(
            prompt.messages,
            temperature=0.3,
            max_tokens=400,
            timeout_s=self._response_llm_timeout_s(),
            max_retries=self._response_llm_max_retries(),
        ):
            if not chunk:
                continue
            reply_parts.append(chunk)
            yield ChatStreamEvent(kind="reply_delta", stage="generation", text=chunk)
        reply = "".join(reply_parts).strip()
        if not reply:
            raise ValueError("LLM returned empty stream response")
        reply_message = Message(role="assistant", parts=(TextPart(text=reply),))
        generation_ms = (time.perf_counter() - generation_started) * 1000.0

        yield ChatStreamEvent(kind="status", stage="persist_accept", text="已接收，后台整合中")
        persist_started = time.perf_counter()
        receipt = self.accept_interaction(
            event_id=resolved_event_id,
            session_id=session_id,
            messages=[*user_messages, reply_message],
            ts=ts,
        )
        persist_ms = (time.perf_counter() - persist_started) * 1000.0

        total_ms = (time.perf_counter() - total_started) * 1000.0
        yield ChatStreamEvent(
            kind="done",
            stage="done",
            result=ChatTurnResult(
                reply_message=reply_message,
                event_id=resolved_event_id,
                memory_context=memory_context,
                rendered_memory_brief=prompt.rendered_memory_brief,
                retrieval_trace_summary=trace_summary,
                persistence=receipt,
                timings=ChatTimings(
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    persist_ms=persist_ms,
                    total_ms=total_ms,
                ),
                llm_error=None,
            ),
        )

    def respond(
        self,
        *,
        session_id: str,
        user_messages: Sequence[Message],
        event_id: Optional[str] = None,
        k: int = 6,
        ts: Optional[float] = None,
    ) -> ChatTurnResult:
        total_started = time.perf_counter()
        resolved_event_id = event_id or f"evt_turn_{uuid.uuid4().hex}"
        retrieval_started = time.perf_counter()
        memory_context, trace_summary = self.build_response_context(
            session_id=session_id,
            user_messages=user_messages,
            k=k,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        prompt = self.response_contexts.build_prompt(
            user_messages=list(user_messages),
            rendered_memory_brief=self.response_contexts.render_memory_brief(memory_context),
        )
        generation_started = time.perf_counter()
        reply_message = self._generate_reply(prompt=prompt)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        persist_started = time.perf_counter()
        receipt = self.accept_interaction(
            event_id=resolved_event_id,
            session_id=session_id,
            messages=[*user_messages, reply_message],
            ts=ts,
        )
        persist_ms = (time.perf_counter() - persist_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return ChatTurnResult(
            reply_message=reply_message,
            event_id=resolved_event_id,
            memory_context=memory_context,
            rendered_memory_brief=prompt.rendered_memory_brief,
            retrieval_trace_summary=trace_summary,
            persistence=receipt,
            timings=ChatTimings(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                persist_ms=persist_ms,
                total_ms=total_ms,
            ),
            llm_error=None,
        )

    def _generate_reply(self, *, prompt: Any) -> Message:
        assert self.llm is not None
        reply_message = self.llm.complete(
            prompt.messages,
            temperature=0.3,
            max_tokens=400,
            timeout_s=self._response_llm_timeout_s(),
            max_retries=self._response_llm_max_retries(),
        )
        if not messages_to_text((reply_message,)).strip():
            raise ValueError("LLM returned an empty reply message")
        return reply_message

    def get_identity(self) -> Dict[str, Any]:
        with self._lock:
            identity = self.mem.snapshot_identity()
            summary = self.mem.narrative_summary()
            return {
                "identity": identity.to_state_dict(),
                "narrative_summary": summary.to_state_dict(),
            }

    def get_event_status(self, event_id: str) -> Dict[str, Any]:
        event = self.store.get_event(event_id)
        if event is None:
            raise KeyError(event_id)
        projection = self.store.get_projection_status(event_id)
        job = self.store.get_job_by_event(event_id)
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "session_id": event.session_id,
            "accepted_at": event.ts,
            "projection": None if projection is None else projection.__dict__,
            "job": None if job is None else job.__dict__,
        }

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job.__dict__

    def get_stats(self) -> Dict[str, Any]:
        runtime_state = self.store.get_runtime_state()
        with self._lock:
            return {
                "plot_count": len(self.mem.plots),
                "summary_count": len(self.mem.summaries),
                "story_count": len(self.mem.stories),
                "theme_count": len(self.mem.themes),
                "architecture_mode": self.mem.cfg.architecture_mode,
                "current_mode": self.mem.identity.current_mode_label,
                "pressure": self.mem.identity.narrative_pressure(),
                "dream_count": self.mem.identity.dream_count,
                "repair_count": self.mem.identity.repair_count,
                "active_energy": self.mem.identity.active_energy,
                "repressed_energy": self.mem.identity.repressed_energy,
                "graph_metrics": dict(self.mem.graph_metrics),
                "queue_depth": self.store.queue_depth(),
                "oldest_pending_age_s": self.store.oldest_pending_age(),
                "last_projected_seq": int(runtime_state.get("last_projected_seq", 0)),
                "last_fade_ts": runtime_state.get("last_fade_ts"),
                "last_evolve_ts": runtime_state.get("last_evolve_ts"),
            }

    def recent_events(self, *, limit: int = 20) -> List[StoredEvent]:
        events = list(self.store.iter_events())
        return events[-max(1, limit) :]

    def feedback(self, *, query_text: str, chosen_id: str, success: bool) -> None:
        with self._lock:
            if chosen_id.startswith("evt_"):
                return
            self.mem.feedback_retrieval(query_text=query_text, chosen_id=chosen_id, success=success)

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.store.queue_depth() == 0:
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=2.0)
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        self.store.close()

    def _snippet_for(self, *, node_id: str, kind: str) -> Tuple[str, Optional[Dict[str, str]]]:
        if kind == "plot":
            plot = self.mem.plots.get(node_id)
            if plot is None:
                return "", None
            metadata = {"source": plot.source, "story_id": plot.story_id or ""}
            if plot.event_id:
                metadata["event_id"] = plot.event_id
            return plot.semantic_text[:240], metadata
        if kind == "summary":
            summary = self.mem.summaries.get(node_id)
            if summary is None:
                return "", None
            metadata = {
                "source_event_count": str(len(summary.source_event_ids)),
                "source_plot_count": str(len(summary.source_plot_ids)),
            }
            return summary.text[:240], metadata
        if kind == "story":
            story = self.mem.stories.get(node_id)
            if story is None:
                return "", None
            return f"plots={len(story.plot_ids)} status={story.status} unresolved={story.unresolved_energy:.3f}", None
        theme = self.mem.themes.get(node_id)
        if theme is None:
            return "", None
        return (theme.name or theme.description)[:240], None
