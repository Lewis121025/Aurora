"""Aurora v2 SDK kernel。"""

from __future__ import annotations

import time
from pathlib import Path

from aurora.expression.cognition import run_cognition
from aurora.expression.context import ExpressionContext
from aurora.llm.config import load_llm_config
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.memory.ledger import Archive, HashEmbeddingEncoder
from aurora.memory.store import SQLiteMemoryStore
from aurora.pipelines.distillation import apply_memory_ops, compile_memory_ops
from aurora.runtime.contracts import CompileFailure, CompileReport, RecallResult, RelationSnapshot, TurnOutput
from aurora.runtime.projections import build_memory_projection, choose_archive_activation


class AuroraKernel:
    """嵌入式 SDK 内核。"""

    __slots__ = ("store", "archive", "llm")

    def __init__(
        self,
        *,
        store: SQLiteMemoryStore,
        archive: Archive,
        llm: LLMProvider,
    ) -> None:
        self.store = store
        self.archive = archive
        self.llm = llm

    @classmethod
    def create(
        cls,
        data_dir: str | None = None,
        llm: LLMProvider | None = None,
    ) -> "AuroraKernel":
        if llm is None:
            llm_config = load_llm_config()
            if llm_config is None:
                raise RuntimeError(
                    "Aurora requires an LLM provider. "
                    "Set AURORA_LLM_BASE_URL, AURORA_LLM_API_KEY, and AURORA_LLM_MODEL."
                )
            llm = OpenAICompatProvider(llm_config)

        root = Path(data_dir or ".aurora")
        store = SQLiteMemoryStore(str(root / "aurora_v2.db"))
        encoder = HashEmbeddingEncoder()
        archive = Archive(store, encoder)
        return cls(store=store, archive=archive, llm=llm)

    def turn(self, session_id: str, text: str, now_ts: float | None = None) -> TurnOutput:
        timestamp = time.time() if now_ts is None else now_ts
        field = self.store.ensure_relation_field(session_id)
        loops = self.store.list_open_loops(session_id)
        recent_events = self.store.recent_turns(session_id)
        recent_turns = tuple(f"{event.role}: {event.text}" for event in recent_events)

        should_recall = choose_archive_activation(text, loops)
        recall_result = (
            self.archive.recall(session_id, text)
            if should_recall
            else RecallResult(relation_id=session_id, query=text, hits=())
        )
        relation_segment, loop_segment, recall_hits, projected_turns = build_memory_projection(
            field=field,
            loops=loops,
            recall_hits=recall_result.hits,
            recent_turns=recent_turns,
            now_ts=timestamp,
        )

        user_event = self.store.append_event(
            relation_id=session_id,
            session_id=session_id,
            kind="user_turn",
            role="user",
            text=text,
            created_at=timestamp,
            pending_compile=True,
        )
        response_text = run_cognition(
            ExpressionContext(
                input_text=text,
                relation_segment=relation_segment,
                open_loop_segment=loop_segment,
                recent_turns=projected_turns,
                recalled_hits=recall_hits,
            ),
            self.llm,
        )
        assistant_event = self.store.append_event(
            relation_id=session_id,
            session_id=session_id,
            kind="assistant_turn",
            role="assistant",
            text=response_text,
            created_at=timestamp,
            pending_compile=True,
        )
        return TurnOutput(
            turn_id=user_event.event_id,
            relation_id=session_id,
            response_text=response_text,
            recall_used=should_recall,
            recalled_ids=tuple(hit.item_id for hit in recall_hits),
            pending_event_ids=(user_event.event_id, assistant_event.event_id),
        )

    def compile_pending(
        self,
        session_id: str | None = None,
        now_ts: float | None = None,
    ) -> CompileReport:
        timestamp = time.time() if now_ts is None else now_ts
        relation_ids = (session_id,) if session_id is not None else self.store.pending_relation_ids()
        compiled: list[str] = []
        failures: list[CompileFailure] = []
        applied_ops = 0

        for relation_id in relation_ids:
            pending_turns = self.store.pending_turns(relation_id)
            if not pending_turns:
                continue

            field = self.store.ensure_relation_field(relation_id)
            loops = self.store.list_open_loops(relation_id)
            facts = self.store.list_facts(relation_id)
            evidence_refs = tuple(event.event_id for event in pending_turns)

            try:
                ops = compile_memory_ops(
                    pending_turns=pending_turns,
                    field=field,
                    loops=loops,
                    facts=facts,
                    llm=self.llm,
                )
                with self.store.transaction():
                    applied_ops += apply_memory_ops(
                        store=self.store,
                        relation_id=relation_id,
                        field=field,
                        ops=ops,
                        evidence_refs=evidence_refs,
                        now_ts=timestamp,
                        encoder=self.archive.encoder,
                    )
                    self.store.save_relation_field(field)
                    self.store.mark_events_compiled(evidence_refs, timestamp)
            except Exception as exc:
                reason = str(exc) or exc.__class__.__name__
                self.store.append_event(
                    relation_id=relation_id,
                    session_id=relation_id,
                    kind="compile_failure",
                    role="system",
                    text=reason,
                    created_at=timestamp,
                    payload={"reason": reason},
                    pending_compile=False,
                )
                failures.append(CompileFailure(relation_id=relation_id, reason=reason))
                continue

            compiled.append(relation_id)

        return CompileReport(
            compiled_relations=tuple(compiled),
            applied_ops=applied_ops,
            failures=tuple(failures),
        )

    def snapshot(self, session_id: str) -> RelationSnapshot:
        field = self.store.ensure_relation_field(session_id)
        return RelationSnapshot(
            relation_id=session_id,
            field=field,
            open_loops=self.store.list_open_loops(session_id),
            facts=self.store.list_facts(session_id),
            pending_compile_count=self.store.pending_compile_count(session_id),
        )

    def recall(self, session_id: str, query: str, limit: int = 5) -> RecallResult:
        return self.archive.recall(session_id, query, limit=limit)

    def close(self) -> None:
        self.store.close()
