from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from aurora.core.models.plot import Plot
from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.llm.schemas import PlotExtraction

from .plot_extractor import PlotExtractor


@dataclass(frozen=True)
class PreparedInteraction:
    extraction: PlotExtraction
    interaction_text: str
    resolved_actors: Tuple[str, ...]
    interaction_embedding: np.ndarray
    context_embedding: Optional[np.ndarray]


class InteractionPreparer:
    """Prepare canonical interaction inputs for live ingest and replay."""

    def __init__(self, *, embedder: EmbeddingProvider, extractor: PlotExtractor):
        self._embedder = embedder
        self._extractor = extractor

    def prepare_live(
        self,
        *,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
    ) -> PreparedInteraction:
        extraction = self._extractor.extract_live(
            user_message=user_message,
            agent_message=agent_message,
            actors=actors,
            context=context,
        )
        return self._prepare(
            user_message=user_message,
            agent_message=agent_message,
            actors=actors,
            context=context,
            extraction=extraction,
            interaction_embedding=None,
            context_embedding=None,
            interaction_text=None,
        )

    def prepare_replay(
        self,
        *,
        event_id: str,
        payload: Dict[str, Any],
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
    ) -> PreparedInteraction:
        extraction = self._extractor.recover_for_replay(
            event_id=event_id,
            payload=payload,
            user_message=user_message,
            agent_message=agent_message,
            actors=actors,
            context=context,
        )
        return self._prepare(
            user_message=user_message,
            agent_message=agent_message,
            actors=payload.get("resolved_actors") or actors,
            context=context,
            extraction=extraction,
            interaction_embedding=self._deserialize_embedding(payload.get("interaction_embedding")),
            context_embedding=self._deserialize_embedding(payload.get("context_embedding")),
            interaction_text=payload.get("interaction_text"),
        )

    def build_event_payload(
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

    def _prepare(
        self,
        *,
        user_message: str,
        agent_message: str,
        actors: Optional[Sequence[str]],
        context: Optional[str],
        extraction: PlotExtraction,
        interaction_embedding: Optional[np.ndarray],
        context_embedding: Optional[np.ndarray],
        interaction_text: Optional[str],
    ) -> PreparedInteraction:
        resolved_actors = self._resolve_actors(actors=actors, extraction=extraction)
        canonical_text = interaction_text or self._build_interaction_text(
            user_message=user_message,
            agent_message=agent_message,
            extraction=extraction,
        )
        text_embedding = interaction_embedding
        if text_embedding is None:
            text_embedding = self._embedder.embed(canonical_text)

        ctx_embedding = context_embedding
        if ctx_embedding is None and context:
            ctx_embedding = self._embedder.embed(context)

        return PreparedInteraction(
            extraction=extraction,
            interaction_text=canonical_text,
            resolved_actors=resolved_actors,
            interaction_embedding=np.asarray(text_embedding, dtype=np.float32),
            context_embedding=np.asarray(ctx_embedding, dtype=np.float32) if ctx_embedding is not None else None,
        )

    @staticmethod
    def _resolve_actors(*, actors: Optional[Sequence[str]], extraction: PlotExtraction) -> Tuple[str, ...]:
        if actors:
            return tuple(str(actor) for actor in actors)
        if extraction.actors:
            return tuple(str(actor) for actor in extraction.actors)
        return ("user", "agent")

    @staticmethod
    def _build_interaction_text(
        *,
        user_message: str,
        agent_message: str,
        extraction: PlotExtraction,
    ) -> str:
        return f"USER: {user_message}\nAGENT: {agent_message}\nOUTCOME: {extraction.outcome}".strip()

    @staticmethod
    def _deserialize_embedding(raw: Any) -> Optional[np.ndarray]:
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim != 1 or arr.size == 0:
            return None
        return arr


def build_plot_document_body(
    *,
    prepared: PreparedInteraction,
    plot: Plot,
    user_message: str,
    agent_message: str,
) -> Dict[str, Any]:
    return {
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
    }


def build_ingest_result_body(*, event_id: str, plot: Plot, encoded: bool) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "plot_id": plot.id,
        "story_id": plot.story_id,
        "encoded": encoded,
        "tension": plot.tension,
        "surprise": plot.surprise,
        "pred_error": plot.pred_error,
        "redundancy": plot.redundancy,
    }
