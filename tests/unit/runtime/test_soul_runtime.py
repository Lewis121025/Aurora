from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora.integrations.storage.doc_store import Document, SQLiteDocStore
from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings
from aurora.system.errors import ConfigurationError


class CountingLLM(LLMProvider):
    def __init__(self) -> None:
        self.complete_calls = 0
        self.complete_json_calls = 0

    def complete(
        self,
        prompt: str,
        *,
        system=None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        stop=None,
        metadata=None,
        max_retries=None,
    ) -> str:
        self.complete_calls += 1
        return "我会陪你一起把这个问题展开。"

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema,
        temperature: float = 0.2,
        timeout_s: float = 30.0,
        metadata=None,
        max_retries=None,
    ):
        self.complete_json_calls += 1
        raise AssertionError("heuristic v4 runtime path should not need auxiliary JSON LLM calls")


def test_runtime_respond_uses_single_generation_call(tmp_path: Path) -> None:
    llm = CountingLLM()
    runtime = AuroraRuntime(
        settings=AuroraSettings(
            data_dir=str(tmp_path),
            embedding_provider="hash",
            axis_embedding_provider="hash",
            meaning_provider="heuristic",
            narrative_provider="heuristic",
        ),
        llm=llm,
    )

    result = runtime.respond(session_id="s1", user_message="你会什么")

    assert llm.complete_calls == 1
    assert llm.complete_json_calls == 0
    assert result.memory_context.mode == result.memory_context.identity.current_mode
    assert result.memory_context.identity is not None
    assert result.ingest_result.mode == result.memory_context.identity.current_mode


def test_runtime_rejects_legacy_snapshot(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True)
    (snapshots_dir / "snapshot_1.json").write_text(
        json.dumps({"last_seq": 1, "state": {"schema_version": "v3"}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        AuroraRuntime(
            settings=AuroraSettings(
                data_dir=str(tmp_path),
                embedding_provider="hash",
                axis_embedding_provider="hash",
                meaning_provider="heuristic",
                narrative_provider="heuristic",
            )
        )


def test_runtime_rejects_buried_legacy_docs(tmp_path: Path) -> None:
    store = SQLiteDocStore(str(tmp_path / "docs.sqlite3"))
    store.upsert(
        Document(
            id="plot_legacy",
            kind="plot",
            ts=1.0,
            body={"runtime_schema_version": "aurora-runtime-v3"},
        )
    )
    for index in range(6):
        store.upsert(
            Document(
                id=f"plot_new_{index}",
                kind="plot",
                ts=100.0 + index,
                body={"runtime_schema_version": "aurora-runtime-v4"},
            )
        )
    store.close()

    with pytest.raises(ConfigurationError):
        AuroraRuntime(
            settings=AuroraSettings(
                data_dir=str(tmp_path),
                embedding_provider="hash",
                axis_embedding_provider="hash",
                meaning_provider="heuristic",
                narrative_provider="heuristic",
            )
        )
