from __future__ import annotations

import numpy as np

from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.llm.schemas import PlotExtraction
from aurora.runtime.interaction_pipeline import (
    InteractionPreparer,
    PreparedInteraction,
    build_ingest_result_body,
    build_plot_document_body,
)


class ExplodingEmbedding(EmbeddingProvider):
    def embed(self, text: str) -> np.ndarray:
        raise AssertionError("embed() should not be called")


class StaticExtractor:
    def __init__(self, extraction: PlotExtraction):
        self.extraction = extraction

    def extract_live(self, **kwargs) -> PlotExtraction:
        return self.extraction

    def recover_for_replay(self, **kwargs) -> PlotExtraction:
        return self.extraction


def test_prepare_replay_reuses_stored_embeddings_without_embed_calls():
    extraction = PlotExtraction(
        action="remember preference",
        actors=["user", "agent"],
        outcome="user likes narrative memory",
    )
    preparer = InteractionPreparer(
        embedder=ExplodingEmbedding(),
        extractor=StaticExtractor(extraction),
    )
    interaction_embedding = np.arange(1, 9, dtype=np.float32)
    context_embedding = np.arange(9, 17, dtype=np.float32)
    payload = {
        "resolved_actors": ["user", "agent"],
        "interaction_text": "USER: hi\nAGENT: hello\nOUTCOME: user likes narrative memory",
        "interaction_embedding": interaction_embedding.tolist(),
        "context_embedding": context_embedding.tolist(),
        "plot_extraction": extraction.model_dump(),
    }

    prepared = preparer.prepare_replay(
        event_id="evt_1",
        payload=payload,
        user_message="hi",
        agent_message="hello",
        actors=None,
        context="memory",
    )

    assert isinstance(prepared, PreparedInteraction)
    assert prepared.resolved_actors == ("user", "agent")
    assert np.allclose(prepared.interaction_embedding, interaction_embedding)
    assert np.allclose(prepared.context_embedding, context_embedding)


def test_document_body_builders_capture_canonical_runtime_state(sample_plot):
    extraction = PlotExtraction(
        action="remember preference",
        actors=["user", "agent"],
        outcome="user likes narrative memory",
        goal="retain preference",
    )
    prepared = PreparedInteraction(
        extraction=extraction,
        interaction_text="USER: hi\nAGENT: hello\nOUTCOME: user likes narrative memory",
        resolved_actors=("user", "agent"),
        interaction_embedding=np.ones(64, dtype=np.float32),
        context_embedding=np.zeros(64, dtype=np.float32),
    )

    plot_body = build_plot_document_body(
        prepared=prepared,
        plot=sample_plot,
        user_message="hi",
        agent_message="hello",
    )
    ingest_body = build_ingest_result_body(event_id="evt_1", plot=sample_plot, memory_layer="explicit")

    assert plot_body["action"] == "remember preference"
    assert plot_body["raw"] == {"user_message": "hi", "agent_message": "hello"}
    assert plot_body["resolved_actors"] == ["user", "agent"]
    assert plot_body["plot_state"]["id"] == sample_plot.id
    assert plot_body["runtime_schema_version"] == "aurora-runtime-v2"
    assert ingest_body["event_id"] == "evt_1"
    assert ingest_body["plot_id"] == sample_plot.id
    assert ingest_body["memory_layer"] == "explicit"
    assert ingest_body["runtime_schema_version"] == "aurora-runtime-v2"
