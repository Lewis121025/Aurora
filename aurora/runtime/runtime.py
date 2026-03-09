from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from aurora.core.causal import CausalEdgeBelief, CausalMemoryGraph
from aurora.core.coherence import CoherenceGuardian
from aurora.core.memory import AuroraMemory
from aurora.core.models.plot import Plot
from aurora.core.models.trace import QueryHit
from aurora.exceptions import ConfigurationError
from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.storage.doc_store import Document, SQLiteDocStore
from aurora.integrations.storage.event_log import Event, SQLiteEventLog
from aurora.integrations.storage.snapshot import Snapshot, SnapshotStore
from aurora.privacy.pii import redact
from aurora.runtime.bootstrap import (
    check_embedding_api_keys,
    create_embedding_provider,
    create_llm_provider,
    create_memory,
)
from aurora.runtime.interaction_pipeline import (
    InteractionPreparer,
    PreparedInteraction,
    RUNTIME_SCHEMA_VERSION,
    build_ingest_result_body,
    build_plot_document_body,
)
from aurora.runtime.plot_extractor import (
    PLOT_EXTRACTION_MAX_RETRIES,
    PLOT_EXTRACTION_TIMEOUT_S,
    PlotExtractor,
)
from aurora.runtime.response_context import ResponseContextBuilder
from aurora.runtime.results import (
    ChatTimings,
    ChatTurnResult,
    CoherenceResult,
    IngestResult,
    QueryResult,
    RetrievalTraceSummary,
    StructuredMemoryContext,
)
from aurora.runtime.settings import AuroraSettings
from aurora.utils.logging import log_event

logger = logging.getLogger(__name__)


class AuroraRuntime:
    """单用户聊天场景下的运行时编排器。"""

    def __init__(self, *, settings: AuroraSettings, llm: Optional[LLMProvider] = None):
        self.settings = settings
        self.llm: LLMProvider = llm or create_llm_provider(settings)

        check_embedding_api_keys()
        self._lock = threading.RLock()

        os.makedirs(self.settings.data_dir, exist_ok=True)

        self.event_log = SQLiteEventLog(os.path.join(self.settings.data_dir, self.settings.event_log_filename))
        self.doc_store = SQLiteDocStore(os.path.join(self.settings.data_dir, "docs.sqlite3"))
        self.snapshots = SnapshotStore(os.path.join(self.settings.data_dir, self.settings.snapshot_dirname))
        self._ensure_v2_doc_store()

        self.last_seq: int = 0
        self.mem: AuroraMemory = self._load_or_init()
        self.plot_extractor = PlotExtractor(llm=self.llm, doc_store=self.doc_store)
        self.interactions = InteractionPreparer(embedder=self.mem.embedder, extractor=self.plot_extractor)
        self.coherence_guardian = CoherenceGuardian(self.mem.metric)
        self.response_contexts = ResponseContextBuilder(
            memory=self.mem,
            doc_store=self.doc_store,
            llm=self.llm,
            compile_timeout_s=min(self.settings.llm_timeout, 8.0),
            compile_max_retries=1,
        )
        self.causal_beliefs: Dict[tuple, CausalEdgeBelief] = {}

        self._replay()

    def _load_or_init(self) -> AuroraMemory:
        try:
            latest = self.snapshots.latest()
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if latest is not None:
            _seq, snap = latest
            self.last_seq = snap.last_seq
            return AuroraMemory.from_state_dict(
                snap.state,
                embedder=create_embedding_provider(self.settings),
            )

        return create_memory(settings=self.settings)

    def _replay(self) -> None:
        for seq, ev in self.event_log.iter_events(after_seq=self.last_seq):
            if ev.type != "interaction":
                continue
            payload = ev.payload
            if payload.get("runtime_schema_version") != RUNTIME_SCHEMA_VERSION:
                raise ConfigurationError(
                    "Detected legacy event payloads. Aurora V2 requires a fresh data directory."
                )
            prepared = self.interactions.prepare_replay(
                event_id=ev.id,
                payload=payload,
                user_message=payload.get("user_message", ""),
                agent_message=payload.get("agent_message", ""),
                actors=payload.get("actors"),
                context=payload.get("context"),
            )
            self._apply_interaction(
                event_id=ev.id,
                user_message=payload.get("user_message", ""),
                agent_message=payload.get("agent_message", ""),
                context=payload.get("context"),
                ts=ev.ts,
                persist=False,
                prepared=prepared,
            )
            canonical_payload = self.interactions.build_event_payload(
                user_message=payload.get("user_message", ""),
                agent_message=payload.get("agent_message", ""),
                actors=payload.get("actors"),
                context=payload.get("context"),
                prepared=prepared,
            )
            if payload != canonical_payload:
                self.event_log.update_payload(ev.id, canonical_payload)
            self.last_seq = seq

    def _ensure_v2_doc_store(self) -> None:
        for kind in ("plot", "ingest_result"):
            for doc in self.doc_store.iter_kind(kind=kind, limit=5):
                if doc.body.get("runtime_schema_version") != RUNTIME_SCHEMA_VERSION:
                    raise ConfigurationError(
                        "Detected legacy documents in docs.sqlite3. Aurora V2 requires a fresh data directory."
                    )

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
        logger: Optional[Any] = None,
    ) -> IngestResult:
        ts = ts or time.time()

        with self._lock:
            existing_seq = self.event_log.get_seq_by_id(event_id)
            if existing_seq is not None:
                return self._load_existing_ingest_result(event_id)

            if self.settings.pii_redaction_enabled:
                user_message = redact(user_message).redacted_text
                agent_message = redact(agent_message).redacted_text

            prepared = self.interactions.prepare_live(
                user_message=user_message,
                agent_message=agent_message,
                actors=actors,
                context=context,
            )

            seq = self.event_log.append(
                Event(
                    id=event_id,
                    ts=ts,
                    session_id=session_id,
                    type="interaction",
                    payload=self.interactions.build_event_payload(
                        user_message=user_message,
                        agent_message=agent_message,
                        actors=actors,
                        context=context,
                        prepared=prepared,
                    ),
                )
            )

            res = self._apply_interaction(
                event_id=event_id,
                user_message=user_message,
                agent_message=agent_message,
                context=context,
                ts=ts,
                persist=True,
                prepared=prepared,
            )
            self.last_seq = max(self.last_seq, seq)

            if self.settings.snapshot_every_events > 0 and self.last_seq % self.settings.snapshot_every_events == 0:
                self._snapshot(logger=logger)

            if logger:
                log_event(logger, "aurora_ingest", event_id=event_id, plot_id=res.plot_id)

            return res

    def build_response_context(
        self,
        *,
        user_message: str,
        k: int = 6,
        asker_id: str = "user",
    ) -> tuple[StructuredMemoryContext, RetrievalTraceSummary]:
        with self._lock:
            trace = self.mem.query_with_timeline(text=user_message, k=k, asker_id=asker_id)
            memory_context = self.response_contexts.build(trace=trace, max_items=k)
            trace_summary = self.response_contexts.summarize_trace(trace)
        return memory_context, trace_summary

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
        logger: Optional[Any] = None,
    ) -> ChatTurnResult:
        total_started = time.perf_counter()
        resolved_event_id = event_id or f"evt_resp_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        retrieval_started = time.perf_counter()
        memory_context, trace_summary = self.build_response_context(
            user_message=user_message,
            k=k,
            asker_id="user",
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        prompt = self.response_contexts.build_prompt(
            user_message=user_message,
            rendered_memory_brief=self.response_contexts.render_memory_brief(memory_context),
        )

        generation_started = time.perf_counter()
        llm_error: Optional[str] = None
        reply = self._generate_reply(
            prompt=prompt,
            memory_context=memory_context,
            user_message=user_message,
        )
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
            logger=logger,
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

    def query(self, *, text: str, k: int = 8) -> QueryResult:
        with self._lock:
            trace = self.mem.query(text, k=k)

        hits: List[QueryHit] = []
        for nid, score, kind in trace.ranked:
            snippet = ""
            if kind == "plot":
                plot = self.mem.plots.get(nid)
                snippet = (plot.text[:240] + "...") if plot else ""
            else:
                doc = self.doc_store.get(nid)
                if doc:
                    snippet = (doc.body.get("summary", "") or doc.body.get("description", ""))[:240]
            hits.append(QueryHit(id=nid, kind=kind, score=float(score), snippet=snippet))

        return QueryResult(query=text, attractor_path_len=len(trace.attractor_path), hits=hits)

    def feedback(self, *, query_text: str, chosen_id: str, success: bool) -> None:
        with self._lock:
            self.mem.feedback_retrieval(query_text=query_text, chosen_id=chosen_id, success=success)

    def evolve(self, *, logger: Optional[Any] = None) -> None:
        with self._lock:
            self.mem.evolve()
            self.mem.self_narrative_engine.update_from_themes(list(self.mem.themes.values()))

        if logger:
            log_event(logger, "aurora_evolve", stories=len(self.mem.stories), themes=len(self.mem.themes))

    def check_coherence(self, *, logger: Optional[Any] = None) -> CoherenceResult:
        with self._lock:
            report = self.coherence_guardian.full_check(
                graph=self.mem.graph,
                plots=self.mem.plots,
                stories=self.mem.stories,
                themes=self.mem.themes,
                causal_beliefs=self.causal_beliefs,
            )

        result = CoherenceResult(
            overall_score=report.overall_score,
            conflict_count=len(report.conflicts),
            unfinished_story_count=len(report.unfinished_stories),
            recommendations=[r.action_description for r in report.recommended_actions[:5]],
        )

        if logger:
            log_event(logger, "aurora_coherence_check", score=result.overall_score, conflicts=result.conflict_count)

        return result

    def get_self_narrative(self) -> Dict[str, Any]:
        with self._lock:
            narrative = self.mem.self_narrative_engine.narrative
            return {
                "profile_id": narrative.profile_id,
                "identity_statement": narrative.identity_statement,
                "identity_narrative": narrative.identity_narrative,
                "seed_narrative": narrative.seed_narrative,
                "capability_narrative": narrative.capability_narrative,
                "core_values": narrative.core_values,
                "coherence_score": narrative.coherence_score,
                "capabilities": {
                    name: {"probability": cap.capability_probability(), "description": cap.description}
                    for name, cap in narrative.capabilities.items()
                },
                "trait_beliefs": {
                    name: {"probability": belief.probability(), "description": belief.description}
                    for name, belief in narrative.trait_beliefs.items()
                },
                "relationships": {
                    entity_id: {
                        "trust": rel.trust(),
                        "familiarity": rel.familiarity(),
                        "interaction_count": rel.interaction_count,
                    }
                    for entity_id, rel in narrative.relationships.items()
                },
                "subconscious": {
                    "dark_matter_count": len(self.mem.subconscious_state.dark_matter_pool),
                    "repressed_count": len(self.mem.subconscious_state.repressed_plot_ids),
                    "last_intuition": list(self.mem.subconscious_state.last_intuition),
                },
                "unresolved_tensions": narrative.unresolved_tensions,
                "full_narrative": narrative.to_full_narrative(),
            }

    def get_causal_chain(self, node_id: str, direction: str = "ancestors") -> List[Dict[str, Any]]:
        with self._lock:
            causal_graph = CausalMemoryGraph(self.mem.metric)
            causal_graph.g = self.mem.graph.g.copy()
            causal_graph.causal_beliefs = self.causal_beliefs
            chain = causal_graph.get_causal_ancestors(node_id) if direction == "ancestors" else causal_graph.get_causal_descendants(node_id)
            return [{"node_id": nid, "strength": strength} for nid, strength in chain]

    def record_feedback_with_learning(
        self,
        *,
        query_text: str,
        chosen_id: str,
        success: bool,
        entity_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.mem.feedback_retrieval(query_text=query_text, chosen_id=chosen_id, success=success)
            plot = self.mem.plots.get(chosen_id)
            if plot:
                self.mem.self_narrative_engine.update_from_interaction(
                    plot=plot,
                    success=success,
                    entity_id=entity_id or "self",
                )

    def _generate_reply(
        self,
        *,
        prompt: Any,
        memory_context: StructuredMemoryContext,
        user_message: str,
    ) -> str:
        try:
            reply = self.llm.complete(
                prompt.user_prompt,
                system=prompt.system_prompt,
                temperature=0.3,
                max_tokens=400,
                timeout_s=self.settings.llm_timeout,
                max_retries=max(1, min(2, int(self.settings.llm_max_retries))),
            ).strip()
        except Exception as exc:
            return f"__LLM_ERROR__:{exc}"

        if reply:
            return reply
        return self._build_response_fallback(memory_context=memory_context, user_message=user_message)

    @staticmethod
    def _build_response_fallback(*, memory_context: StructuredMemoryContext, user_message: str) -> str:
        if memory_context.known_facts:
            return (
                "这轮语言模型调用失败了。"
                f"我当前能确认的记忆是：{memory_context.known_facts[0]}"
            )
        if memory_context.preferences:
            return (
                "这轮语言模型调用失败了。"
                f"我目前只确认到一个偏好相关记忆：{memory_context.preferences[0]}"
            )
        return (
            "这轮语言模型调用失败了，而且当前没有足够稳定的记忆证据来回答这句话。"
            f"我会先把这轮对话记住：{user_message[:80]}"
        )

    def _apply_interaction(
        self,
        *,
        event_id: str,
        user_message: str,
        agent_message: str,
        context: Optional[str],
        ts: float,
        persist: bool,
        prepared: PreparedInteraction,
    ) -> IngestResult:
        plot = self.mem.ingest(
            prepared.interaction_text,
            actors=prepared.resolved_actors,
            context_text=context,
            event_id=event_id,
            interaction_embedding=prepared.interaction_embedding,
            context_embedding=prepared.context_embedding,
            ts=ts,
        )
        memory_layer = "explicit" if plot.exposure == "explicit" else "shadow"

        if persist:
            self._persist_interaction_artifacts(
                event_id=event_id,
                ts=ts,
                plot=plot,
                memory_layer=memory_layer,
                prepared=prepared,
                user_message=user_message,
                agent_message=agent_message,
            )

        return IngestResult(
            event_id=event_id,
            plot_id=plot.id,
            story_id=plot.story_id,
            memory_layer=memory_layer,
            tension=float(plot.tension),
            surprise=float(plot.surprise),
            pred_error=float(plot.pred_error),
            redundancy=float(plot.redundancy),
        )

    def _snapshot(self, *, logger: Optional[Any] = None) -> None:
        snap = Snapshot(last_seq=self.last_seq, state=self.mem.to_state_dict())
        path = self.snapshots.save(snap)
        if logger:
            log_event(logger, "aurora_snapshot", last_seq=self.last_seq, path=path)

    def _load_existing_ingest_result(self, event_id: str) -> IngestResult:
        doc = self.doc_store.get(f"ingest:{event_id}")
        if doc is None:
            return IngestResult(
                event_id=event_id,
                plot_id="",
                story_id=None,
                memory_layer="shadow",
                tension=0.0,
                surprise=0.0,
                pred_error=0.0,
                redundancy=0.0,
            )

        body = doc.body
        return IngestResult(
            event_id=event_id,
            plot_id=body["plot_id"],
            story_id=body.get("story_id"),
            memory_layer=body.get("memory_layer", "explicit"),
            tension=float(body.get("tension", 0.0)),
            surprise=float(body.get("surprise", 0.0)),
            pred_error=float(body.get("pred_error", 0.0)),
            redundancy=float(body.get("redundancy", 0.0)),
        )

    def _persist_interaction_artifacts(
        self,
        *,
        event_id: str,
        ts: float,
        plot: Plot,
        memory_layer: str,
        prepared: PreparedInteraction,
        user_message: str,
        agent_message: str,
    ) -> None:
        self.doc_store.upsert(
            Document(
                id=plot.id,
                kind="plot",
                ts=ts,
                body=build_plot_document_body(
                    prepared=prepared,
                    plot=plot,
                    user_message=user_message,
                    agent_message=agent_message,
                ),
            )
        )
        self.doc_store.upsert(
            Document(
                id=f"ingest:{event_id}",
                kind="ingest_result",
                ts=ts,
                body=build_ingest_result_body(event_id=event_id, plot=plot, memory_layer=memory_layer),
            )
        )
