from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aurora.core.causal import CausalEdgeBelief, CausalMemoryGraph
from aurora.core.coherence import CoherenceGuardian
from aurora.core.memory import AuroraMemory
from aurora.core.models.trace import QueryHit
from aurora.core.self_narrative import SelfNarrativeEngine
from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.schemas import PlotExtraction
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
from aurora.runtime.results import CoherenceResult, IngestResult, QueryResult
from aurora.runtime.settings import AuroraSettings
from aurora.utils.logging import log_event

logger = logging.getLogger(__name__)

PLOT_EXTRACTION_TIMEOUT_S = 8.0
PLOT_EXTRACTION_MAX_RETRIES = 1


@dataclass(frozen=True)
class PreparedInteraction:
    extraction: PlotExtraction
    interaction_text: str
    resolved_actors: Tuple[str, ...]
    interaction_embedding: np.ndarray
    context_embedding: Optional[np.ndarray]


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

        self.last_seq: int = 0
        self.mem: AuroraMemory = self._load_or_init()
        self.coherence_guardian = CoherenceGuardian(self.mem.metric)
        self.self_narrative_engine = SelfNarrativeEngine(self.mem.metric)
        self.causal_beliefs: Dict[tuple, CausalEdgeBelief] = {}

        self._replay()

    def _load_or_init(self) -> AuroraMemory:
        latest = self.snapshots.latest()
        if latest is not None:
            _seq, snap = latest
            self.last_seq = snap.last_seq
            state = snap.state
            state.embedder = create_embedding_provider(self.settings)
            return state

        return create_memory(settings=self.settings)

    def _replay(self) -> None:
        for seq, ev in self.event_log.iter_events(after_seq=self.last_seq):
            if ev.type != "interaction":
                continue
            payload = ev.payload
            prepared = self._prepare_replay_interaction(
                event_id=ev.id,
                user_message=payload.get("user_message", ""),
                agent_message=payload.get("agent_message", ""),
                actors=payload.get("actors"),
                context=payload.get("context"),
                payload=payload,
            )
            self._apply_interaction(
                event_id=ev.id,
                user_message=payload.get("user_message", ""),
                agent_message=payload.get("agent_message", ""),
                actors=payload.get("actors"),
                context=payload.get("context"),
                ts=ev.ts,
                persist=False,
                prepared=prepared,
            )
            canonical_payload = self._build_event_payload(
                user_message=payload.get("user_message", ""),
                agent_message=payload.get("agent_message", ""),
                actors=payload.get("actors"),
                context=payload.get("context"),
                prepared=prepared,
            )
            if payload != canonical_payload:
                self.event_log.update_payload(ev.id, canonical_payload)
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
        logger: Optional[Any] = None,
    ) -> IngestResult:
        ts = ts or time.time()

        with self._lock:
            existing_seq = self.event_log.get_seq_by_id(event_id)
            if existing_seq is not None:
                doc = self.doc_store.get(f"ingest:{event_id}")
                if doc:
                    body = doc.body
                    return IngestResult(
                        event_id=event_id,
                        plot_id=body["plot_id"],
                        story_id=body.get("story_id"),
                        encoded=bool(body.get("encoded", True)),
                        tension=float(body.get("tension", 0.0)),
                        surprise=float(body.get("surprise", 0.0)),
                        pred_error=float(body.get("pred_error", 0.0)),
                        redundancy=float(body.get("redundancy", 0.0)),
                    )
                return IngestResult(event_id=event_id, plot_id="", story_id=None, encoded=False, tension=0.0, surprise=0.0, pred_error=0.0, redundancy=0.0)

            if self.settings.pii_redaction_enabled:
                user_message = redact(user_message).redacted_text
                agent_message = redact(agent_message).redacted_text

            prepared = self._prepare_live_interaction(
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
                    payload=self._build_event_payload(
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
                actors=actors,
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
            self.self_narrative_engine.update_from_themes(list(self.mem.themes.values()))

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
            narrative = self.self_narrative_engine.narrative
            return {
                "identity_statement": narrative.identity_statement,
                "identity_narrative": narrative.identity_narrative,
                "capability_narrative": narrative.capability_narrative,
                "core_values": narrative.core_values,
                "coherence_score": narrative.coherence_score,
                "capabilities": {
                    name: {"probability": cap.capability_probability(), "description": cap.description}
                    for name, cap in narrative.capabilities.items()
                },
                "relationships": {
                    entity_id: {
                        "trust": rel.trust(),
                        "familiarity": rel.familiarity(),
                        "interaction_count": rel.interaction_count,
                    }
                    for entity_id, rel in narrative.relationships.items()
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
                self.self_narrative_engine.update_from_interaction(plot=plot, success=success, entity_id=entity_id or "self")

    def _apply_interaction(
        self,
        *,
        event_id: str,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
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
        encoded = plot.id in self.mem.plots

        if persist:
            self.doc_store.upsert(
                Document(
                    id=plot.id,
                    kind="plot",
                    ts=ts,
                    body={
                        "schema_version": prepared.extraction.schema_version,
                        "actors": prepared.extraction.actors,
                        "action": prepared.extraction.action,
                        "context": prepared.extraction.context,
                        "outcome": prepared.extraction.outcome,
                        "goal": prepared.extraction.goal,
                        "obstacles": prepared.extraction.obstacles,
                        "decision": prepared.extraction.decision,
                        "emotion_valence": prepared.extraction.emotion_valence,
                        "emotion_arousal": prepared.extraction.emotion_arousal,
                        "claims": [claim.model_dump() for claim in prepared.extraction.claims],
                        "interaction_text": prepared.interaction_text,
                        "resolved_actors": list(prepared.resolved_actors),
                        "context_embedding": prepared.context_embedding.tolist() if prepared.context_embedding is not None else None,
                        "plot_state": plot.to_state_dict(),
                        "raw": {"user_message": user_message, "agent_message": agent_message},
                    },
                )
            )
            self.doc_store.upsert(
                Document(
                    id=f"ingest:{event_id}",
                    kind="ingest_result",
                    ts=ts,
                    body={
                        "event_id": event_id,
                        "plot_id": plot.id,
                        "story_id": plot.story_id,
                        "encoded": encoded,
                        "tension": plot.tension,
                        "surprise": plot.surprise,
                        "pred_error": plot.pred_error,
                        "redundancy": plot.redundancy,
                    },
                )
            )

        return IngestResult(
            event_id=event_id,
            plot_id=plot.id,
            story_id=plot.story_id,
            encoded=encoded,
            tension=float(plot.tension),
            surprise=float(plot.surprise),
            pred_error=float(plot.pred_error),
            redundancy=float(plot.redundancy),
        )

    def _prepare_live_interaction(
        self,
        *,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
    ) -> PreparedInteraction:
        extraction = self._extract_plot(user_message=user_message, agent_message=agent_message, context=context, actors=actors)
        return self._prepare_interaction(
            user_message=user_message,
            agent_message=agent_message,
            actors=actors,
            context=context,
            extraction=extraction,
            interaction_embedding=None,
            context_embedding=None,
        )

    def _prepare_replay_interaction(
        self,
        *,
        event_id: str,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
        payload: Dict[str, Any],
    ) -> PreparedInteraction:
        extraction = self._load_plot_extraction(
            event_id=event_id,
            payload=payload,
            user_message=user_message,
            agent_message=agent_message,
            actors=actors,
            context=context,
        )
        return self._prepare_interaction(
            user_message=user_message,
            agent_message=agent_message,
            actors=payload.get("resolved_actors") or actors,
            context=context,
            extraction=extraction,
            interaction_embedding=self._deserialize_embedding(payload.get("interaction_embedding")),
            context_embedding=self._deserialize_embedding(payload.get("context_embedding")),
            interaction_text=payload.get("interaction_text"),
        )

    def _prepare_interaction(
        self,
        *,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
        extraction: PlotExtraction,
        interaction_embedding: Optional[np.ndarray],
        context_embedding: Optional[np.ndarray],
        interaction_text: Optional[str] = None,
    ) -> PreparedInteraction:
        resolved_actors = self._resolve_actors(actors=actors, extraction=extraction)
        canonical_text = interaction_text or self._build_interaction_text(
            user_message=user_message,
            agent_message=agent_message,
            extraction=extraction,
        )
        text_embedding = interaction_embedding
        if text_embedding is None:
            text_embedding = self.mem.embedder.embed(canonical_text)
        ctx_embedding = context_embedding
        if ctx_embedding is None and context:
            ctx_embedding = self.mem.embedder.embed(context)
        return PreparedInteraction(
            extraction=extraction,
            interaction_text=canonical_text,
            resolved_actors=resolved_actors,
            interaction_embedding=np.asarray(text_embedding, dtype=np.float32),
            context_embedding=np.asarray(ctx_embedding, dtype=np.float32) if ctx_embedding is not None else None,
        )

    def _build_event_payload(
        self,
        *,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
        prepared: PreparedInteraction,
    ) -> Dict[str, Any]:
        return {
            "user_message": user_message,
            "agent_message": agent_message,
            "actors": list(actors) if actors else None,
            "context": context,
            "plot_extraction": prepared.extraction.model_dump(),
            "interaction_text": prepared.interaction_text,
            "resolved_actors": list(prepared.resolved_actors),
            "interaction_embedding": prepared.interaction_embedding.tolist(),
            "context_embedding": prepared.context_embedding.tolist() if prepared.context_embedding is not None else None,
        }

    def _load_plot_extraction(
        self,
        *,
        event_id: str,
        payload: Dict[str, Any],
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
    ) -> PlotExtraction:
        stored = payload.get("plot_extraction")
        if isinstance(stored, dict):
            try:
                return PlotExtraction.model_validate(stored)
            except Exception as exc:
                logger.warning("Invalid stored plot extraction for event %s: %s", event_id, exc)

        plot_doc = self._load_plot_doc(event_id)
        if plot_doc is not None:
            try:
                return PlotExtraction.model_validate(plot_doc.body)
            except Exception as exc:
                logger.warning("Invalid plot document extraction for event %s: %s", event_id, exc)

        return self._build_minimal_plot_extraction(
            user_message=user_message,
            agent_message=agent_message,
            actors=actors,
            context=context,
        )

    def _load_plot_doc(self, event_id: str) -> Optional[Document]:
        ingest_doc = self.doc_store.get(f"ingest:{event_id}")
        if ingest_doc is None:
            return None
        plot_id = ingest_doc.body.get("plot_id")
        if not plot_id:
            return None
        return self.doc_store.get(str(plot_id))

    def _resolve_actors(self, *, actors: Optional[Sequence[str]], extraction: PlotExtraction) -> Tuple[str, ...]:
        if actors:
            return tuple(str(actor) for actor in actors)
        if extraction.actors:
            return tuple(str(actor) for actor in extraction.actors)
        return ("user", "agent")

    def _build_interaction_text(
        self,
        *,
        user_message: str,
        agent_message: str,
        extraction: PlotExtraction,
    ) -> str:
        return f"USER: {user_message}\nAGENT: {agent_message}\nOUTCOME: {extraction.outcome}".strip()

    def _deserialize_embedding(self, raw: Any) -> Optional[np.ndarray]:
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim != 1 or arr.size == 0:
            return None
        return arr

    def _build_minimal_plot_extraction(
        self,
        *,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
    ) -> PlotExtraction:
        action_source = user_message or agent_message or "interaction"
        fallback_actors = list(actors) if actors else ["user", "agent"]
        return PlotExtraction(
            action=action_source[:120],
            actors=fallback_actors,
            context=context or "",
        )

    def _extract_plot(
        self,
        *,
        user_message: str,
        agent_message: str,
        context: Optional[str],
        actors: Optional[Sequence[str]],
    ) -> PlotExtraction:
        from aurora.integrations.llm import prompts

        instruction = prompts.instruction("PlotExtraction")
        user_prompt = prompts.render(
            prompts.PLOT_EXTRACTION_USER,
            instruction=instruction,
            user_message=user_message,
            agent_message=agent_message,
            context=context or "",
        )
        try:
            return self.llm.complete_json(
                system=prompts.PLOT_EXTRACTION_SYSTEM,
                user=user_prompt,
                schema=PlotExtraction,
                temperature=0.2,
                timeout_s=PLOT_EXTRACTION_TIMEOUT_S,
                max_retries=PLOT_EXTRACTION_MAX_RETRIES,
            )
        except Exception as exc:
            logger.debug("LLM plot extraction failed, using minimal fallback: %s", exc)
            return self._build_minimal_plot_extraction(
                user_message=user_message,
                agent_message=agent_message,
                actors=actors,
                context=context,
            )

    def _snapshot(self, *, logger: Optional[Any] = None) -> None:
        snap = Snapshot(last_seq=self.last_seq, state=self.mem)
        path = self.snapshots.save(snap)
        if logger:
            log_event(logger, "aurora_snapshot", last_seq=self.last_seq, path=path)
