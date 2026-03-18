"""Aurora v3 relation-only kernel."""

from __future__ import annotations

import re
import time
from pathlib import Path

from aurora.expression.cognition import run_cognition
from aurora.expression.context import ExpressionContext
from aurora.llm.config import load_llm_config
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.memory.atoms import derive_facts, derive_open_loops, derive_relation_field
from aurora.memory.ledger import Archive, HashEmbeddingEncoder
from aurora.memory.store import SQLiteMemoryStore
from aurora.pipelines.distillation import CompileError, apply_memory_ops, compile_post_response_ops, compile_pre_response_ops
from aurora.runtime.contracts import OpenLoop, RecallResult, RelationField, RelationSnapshot, TurnOutput
from aurora.runtime.projections import build_memory_projection


class AuroraKernel:
    """Embedded relation-first Aurora runtime."""

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
        store = SQLiteMemoryStore(str(root / "aurora_v3.db"))
        archive = Archive(store, HashEmbeddingEncoder())
        return cls(store=store, archive=archive, llm=llm)

    def turn(self, relation_id: str, text: str, now_ts: float | None = None) -> TurnOutput:
        timestamp = time.time() if now_ts is None else now_ts
        user_event = self.store.append_event(
            relation_id=relation_id,
            kind="user_turn",
            role="user",
            text=text,
            created_at=timestamp,
        )

        applied_atom_ids: list[str] = []
        pre_ops = compile_pre_response_ops(
            text=text,
            atoms=self.store.list_atoms(relation_id),
        )
        if pre_ops:
            with self.store.transaction():
                applied_atom_ids.extend(
                    apply_memory_ops(
                        store=self.store,
                        relation_id=relation_id,
                        ops=pre_ops,
                        evidence_event_ids=(user_event.event_id,),
                        now_ts=timestamp,
                    )
                )

        recall_result = (
            self.archive.recall(relation_id, text)
            if not _suppress_recall(text)
            else RecallResult(relation_id=relation_id, query=text, hits=())
        )
        field, loops = self._runtime_views(relation_id)
        relation_segment, loop_segment, recall_hits = build_memory_projection(
            field=field,
            loops=loops,
            recall_hits=recall_result.hits,
            now_ts=timestamp,
        )
        response_text = run_cognition(
            ExpressionContext(
                input_text=text,
                relation_segment=relation_segment,
                open_loop_segment=loop_segment,
                recalled_hits=recall_hits,
            ),
            self.llm,
        )
        assistant_event = self.store.append_event(
            relation_id=relation_id,
            kind="assistant_turn",
            role="assistant",
            text=response_text,
            created_at=timestamp,
        )

        try:
            post_ops = compile_post_response_ops(
                relation_id=relation_id,
                user_turn=text,
                assistant_turn=response_text,
                atoms=self.store.list_atoms(relation_id),
                llm=self.llm,
            )
            if post_ops:
                with self.store.transaction():
                    applied_atom_ids.extend(
                        apply_memory_ops(
                            store=self.store,
                            relation_id=relation_id,
                            ops=post_ops,
                            evidence_event_ids=(user_event.event_id, assistant_event.event_id),
                            now_ts=timestamp,
                        )
                    )
        except Exception as exc:
            reason = str(exc) or exc.__class__.__name__
            self.store.append_event(
                relation_id=relation_id,
                kind="compile_failure",
                role="system",
                text=reason,
                created_at=timestamp,
                payload={"reason": reason},
            )
            if isinstance(exc, CompileError):
                pass

        return TurnOutput(
            turn_id=user_event.event_id,
            relation_id=relation_id,
            response_text=response_text,
            recall_used=bool(recall_result.hits),
            recalled_ids=tuple(hit.item_id for hit in recall_result.hits),
            applied_atom_ids=tuple(applied_atom_ids),
        )

    def snapshot(self, relation_id: str) -> RelationSnapshot:
        atoms = self.store.list_atoms(relation_id)
        recent_events = self.store.recent_turns(relation_id, limit=8)
        field = derive_relation_field(relation_id, atoms)
        return RelationSnapshot(
            relation_id=relation_id,
            field=field,
            open_loops=derive_open_loops(atoms),
            facts=derive_facts(atoms),
            atoms=atoms,
            recent_events=recent_events,
        )

    def recall(self, relation_id: str, query: str, limit: int = 5) -> RecallResult:
        return self.archive.recall(relation_id, query, limit=limit)

    def close(self) -> None:
        self.store.close()

    def _runtime_views(self, relation_id: str) -> tuple[RelationField, tuple[OpenLoop, ...]]:
        atoms = self.store.list_atoms(relation_id)
        field = derive_relation_field(relation_id, atoms)
        loops = derive_open_loops(atoms)
        return field, loops


def _suppress_recall(text: str) -> bool:
    if any(marker in text for marker in ("更正", "改成", "记错", "别再记", "不要记", "别记", "忘了这个")):
        return True
    return re.search(r"不是[^？?。！!\n]{0,24}(?:，|,|、|\s|而)是", text) is not None
