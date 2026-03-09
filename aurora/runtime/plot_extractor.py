from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.schemas import PlotExtraction
from aurora.integrations.storage.doc_store import Document, SQLiteDocStore

logger = logging.getLogger(__name__)

PLOT_EXTRACTION_TIMEOUT_S = 8.0
PLOT_EXTRACTION_MAX_RETRIES = 1


class PlotExtractor:
    """Encapsulate plot extraction and replay recovery policy."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        doc_store: SQLiteDocStore,
        timeout_s: float = PLOT_EXTRACTION_TIMEOUT_S,
        max_retries: int = PLOT_EXTRACTION_MAX_RETRIES,
    ):
        self._llm = llm
        self._doc_store = doc_store
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def extract_live(
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
            return self._llm.complete_json(
                system=prompts.PLOT_EXTRACTION_SYSTEM,
                user=user_prompt,
                schema=PlotExtraction,
                temperature=0.2,
                timeout_s=self._timeout_s,
                max_retries=self._max_retries,
            )
        except Exception as exc:
            logger.debug("LLM plot extraction failed, using minimal fallback: %s", exc)
            return self.build_minimal(
                user_message=user_message,
                agent_message=agent_message,
                actors=actors,
                context=context,
            )

    def recover_for_replay(
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

        return self.build_minimal(
            user_message=user_message,
            agent_message=agent_message,
            actors=actors,
            context=context,
        )

    def build_minimal(
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

    def _load_plot_doc(self, event_id: str) -> Optional[Document]:
        ingest_doc = self._doc_store.get(f"ingest:{event_id}")
        if ingest_doc is None:
            return None
        plot_id = ingest_doc.body.get("plot_id")
        if not plot_id:
            return None
        return self._doc_store.get(str(plot_id))
