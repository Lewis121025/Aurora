from __future__ import annotations

from pathlib import Path

from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.storage.doc_store import Document, SQLiteDocStore
from aurora.runtime.plot_extractor import PlotExtractor


class ExplodingLLM(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        raise AssertionError("complete() should not be called")

    def complete_json(self, *, system: str, user: str, schema, **kwargs):
        raise AssertionError("complete_json() should not be called")


def test_recover_for_replay_uses_doc_store_when_event_payload_is_missing(temp_data_dir: Path):
    store = SQLiteDocStore(str(temp_data_dir / "docs.sqlite3"))
    store.upsert(
        Document(
            id="plot-1",
            kind="plot",
            ts=1.0,
            body={
                "schema_version": "1.0",
                "actors": ["user", "agent"],
                "action": "remember preference",
                "context": "memory",
                "outcome": "user likes narrative memory",
                "goal": "",
                "obstacles": [],
                "decision": "",
                "emotion_valence": 0.0,
                "emotion_arousal": 0.2,
                "claims": [],
            },
        )
    )
    store.upsert(
        Document(
            id="ingest:evt_1",
            kind="ingest_result",
            ts=1.0,
            body={"plot_id": "plot-1"},
        )
    )
    extractor = PlotExtractor(llm=ExplodingLLM(), doc_store=store)

    extraction = extractor.recover_for_replay(
        event_id="evt_1",
        payload={},
        user_message="hi",
        agent_message="hello",
        actors=None,
        context="memory",
    )

    assert extraction.action == "remember preference"
    assert extraction.outcome == "user likes narrative memory"


def test_recover_for_replay_falls_back_to_minimal_extraction(temp_data_dir: Path):
    store = SQLiteDocStore(str(temp_data_dir / "docs.sqlite3"))
    extractor = PlotExtractor(llm=ExplodingLLM(), doc_store=store)

    extraction = extractor.recover_for_replay(
        event_id="evt_missing",
        payload={},
        user_message="please remember this",
        agent_message="done",
        actors=["user", "assistant"],
        context="memory",
    )

    assert extraction.action == "please remember this"
    assert extraction.actors == ["user", "assistant"]
    assert extraction.context == "memory"
