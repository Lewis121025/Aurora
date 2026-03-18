"""Aurora unified-atom human memory kernel."""

from __future__ import annotations

import time
from pathlib import Path

from aurora.expression.cognition import DEFAULT_SYSTEM_PROMPT, Responder
from aurora.expression.context import ExpressionContext
from aurora.llm.config import load_llm_config
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.memory.ledger import Archive, HashEmbeddingEncoder
from aurora.memory.state import project_subject_state
from aurora.memory.store import SQLiteMemoryStore
from aurora.pipelines.distillation import MemoryCompiler, make_evidence_atom
from aurora.runtime.contracts import (
    MemoryAtom,
    RecallMode,
    RecallResult,
    RecallTemporalScope,
    SubjectMemoryState,
    TurnOutput,
)
from aurora.runtime.projections import build_memory_projection

_TURN_STEP_S = 1e-6


class AuroraKernel:
    """Embedded subject-centric Aurora runtime."""

    __slots__ = ("store", "archive", "compiler", "responder")

    def __init__(
        self,
        *,
        store: SQLiteMemoryStore,
        archive: Archive,
        compiler: MemoryCompiler,
        responder: Responder,
    ) -> None:
        self.store = store
        self.archive = archive
        self.compiler = compiler
        self.responder = responder

    @classmethod
    def create(
        cls,
        data_dir: str | None = None,
        llm: LLMProvider | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> "AuroraKernel":
        if llm is None:
            llm_config = load_llm_config()
            if llm_config is None:
                raise RuntimeError(
                    "Aurora requires an LLM provider. "
                    "Set AURORA_LLM_BASE_URL and AURORA_LLM_API_KEY. "
                    "AURORA_LLM_MODEL defaults to gpt-4o-mini."
                )
            llm = OpenAICompatProvider(llm_config)

        root = Path(data_dir or ".aurora")
        store = SQLiteMemoryStore(str(root / "aurora_vnext.db"))
        archive = Archive(store, HashEmbeddingEncoder())
        compiler = MemoryCompiler(llm)
        responder = Responder(llm, system_prompt=system_prompt)
        return cls(store=store, archive=archive, compiler=compiler, responder=responder)

    def turn(self, subject_id: str, text: str, now_ts: float | None = None) -> TurnOutput:
        user_timestamp = time.time() if now_ts is None else now_ts
        assistant_timestamp = user_timestamp + _TURN_STEP_S
        user_atom = make_evidence_atom(
            subject_id=subject_id,
            event_kind="user_turn",
            role="user",
            text=text,
            now_ts=user_timestamp,
        )
        self.store.add_atom(user_atom)
        applied_atom_ids: list[str] = []

        try:
            created_atoms, updated_atoms = self.compiler.compile_user_turn(
                subject_id=subject_id,
                user_atom=user_atom,
                existing_atoms=self.store.list_atoms(subject_id),
                now_ts=user_timestamp,
            )
            self._apply_atom_changes(
                created_atoms=created_atoms,
                updated_atoms=updated_atoms,
                applied_atom_ids=applied_atom_ids,
            )
        except Exception as exc:
            self._record_compile_failure(subject_id=subject_id, reason=exc, now_ts=user_timestamp)

        recall_result = self.archive.recall(
            subject_id,
            text,
            temporal_scope="current",
            limit=5,
            mode="blended",
        )
        state = self.state(subject_id)
        state_segment, episode_segment, recall_hits = build_memory_projection(state, recall_result.hits)
        response_text = self.responder.respond(
            ExpressionContext(
                input_text=text,
                state_segment=state_segment,
                episode_segment=episode_segment,
                recalled_hits=recall_hits,
            )
        )
        assistant_atom = make_evidence_atom(
            subject_id=subject_id,
            event_kind="assistant_turn",
            role="assistant",
            text=response_text,
            now_ts=assistant_timestamp,
        )
        self.store.add_atom(assistant_atom)

        try:
            episode_atom, created_atoms, updated_atoms = self.compiler.compile_completed_turn(
                subject_id=subject_id,
                user_atom=user_atom,
                assistant_atom=assistant_atom,
                existing_atoms=self.store.list_atoms(subject_id),
                now_ts=assistant_timestamp,
            )
            self._apply_atom_changes(
                created_atoms=(episode_atom, *created_atoms),
                updated_atoms=updated_atoms,
                applied_atom_ids=applied_atom_ids,
            )
        except Exception as exc:
            self._record_compile_failure(subject_id=subject_id, reason=exc, now_ts=assistant_timestamp)

        return TurnOutput(
            turn_id=user_atom.atom_id,
            subject_id=subject_id,
            response_text=response_text,
            recall_used=bool(recall_result.hits),
            applied_atom_ids=tuple(applied_atom_ids),
        )

    def state(self, subject_id: str) -> SubjectMemoryState:
        return project_subject_state(subject_id=subject_id, atoms=self.store.list_atoms(subject_id))

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        temporal_scope: RecallTemporalScope,
        limit: int = 5,
        mode: RecallMode = "blended",
    ) -> RecallResult:
        return self.archive.recall(
            subject_id,
            query,
            temporal_scope=temporal_scope,
            limit=limit,
            mode=mode,
        )

    def close(self) -> None:
        self.store.close()

    def _apply_atom_changes(
        self,
        *,
        created_atoms: tuple[MemoryAtom, ...],
        updated_atoms: tuple[MemoryAtom, ...],
        applied_atom_ids: list[str],
    ) -> None:
        if not created_atoms and not updated_atoms:
            return
        with self.store.transaction():
            for atom in updated_atoms:
                self.store.update_atom(atom)
            for atom in created_atoms:
                self.store.add_atom(atom)
                applied_atom_ids.append(atom.atom_id)

    def _record_compile_failure(self, *, subject_id: str, reason: Exception, now_ts: float) -> None:
        self.store.add_atom(self._compile_failure_atom(subject_id=subject_id, reason=reason, now_ts=now_ts))

    def _compile_failure_atom(self, *, subject_id: str, reason: Exception, now_ts: float) -> MemoryAtom:
        message = str(reason) or reason.__class__.__name__
        return make_evidence_atom(
            subject_id=subject_id,
            event_kind="compile_failure",
            role="system",
            text=message,
            now_ts=now_ts,
            payload={"reason": message},
        )
