"""Aurora memory-field runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from aurora.expression.cognition import DEFAULT_SYSTEM_PROMPT, Responder
from aurora.expression.context import ExpressionContext
from aurora.llm.config import LLMSettings, coerce_llm_settings, load_llm_settings
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.memory.ledger import HashEmbeddingEncoder, MemoryField
from aurora.memory.state import project_subject_state
from aurora.memory.store import SQLiteMemoryStore
from aurora.pipelines.distillation import MemoryCompiler, make_evidence_atom
from aurora.runtime.contracts import MemoryAtom, MemoryEdge, RecallResult, SubjectMemoryState, TurnOutput
from aurora.runtime.projections import build_memory_brief

_TURN_STEP_S = 1e-6
_OPENAI_COMPAT_PROVIDERS = frozenset({"openai", "openai_compatible", "bailian"})


def _build_llm_provider(settings: LLMSettings) -> LLMProvider:
    if settings.provider in _OPENAI_COMPAT_PROVIDERS:
        return OpenAICompatProvider(settings.config)
    raise RuntimeError(
        "Unsupported AURORA_LLM_PROVIDER. "
        "Supported providers: openai, openai_compatible, bailian."
    )


class AuroraKernel:
    """Embedded subject-centric Aurora runtime."""

    __slots__ = ("store", "field", "compiler", "responder")

    def __init__(
        self,
        *,
        store: SQLiteMemoryStore,
        field: MemoryField,
        compiler: MemoryCompiler,
        responder: Responder,
    ) -> None:
        self.store = store
        self.field = field
        self.compiler = compiler
        self.responder = responder

    @classmethod
    def create(
        cls,
        data_dir: str | None = None,
        llm: LLMProvider | None = None,
        llm_settings: LLMSettings | Mapping[str, object] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> "AuroraKernel":
        if llm is not None and llm_settings is not None:
            raise ValueError("Pass either llm or llm_settings, not both")
        if llm is None:
            settings = coerce_llm_settings(llm_settings) if llm_settings is not None else load_llm_settings()
            if settings is None:
                raise RuntimeError(
                    "Aurora requires llm_settings. "
                    "Pass llm_settings={'provider': '...', 'config': {...}} "
                    "or set AURORA_LLM_PROVIDER and AURORA_LLM_CONFIG_*."
                )
            llm = _build_llm_provider(settings)

        root = Path(data_dir or ".aurora")
        store = SQLiteMemoryStore(str(root / "aurora_vnext.db"))
        field = MemoryField(store, HashEmbeddingEncoder())
        compiler = MemoryCompiler(llm)
        responder = Responder(llm, system_prompt=system_prompt)
        return cls(store=store, field=field, compiler=compiler, responder=responder)

    def turn(self, subject_id: str, text: str, now_ts: float | None = None) -> TurnOutput:
        user_timestamp = time.time() if now_ts is None else now_ts
        assistant_timestamp = user_timestamp + _TURN_STEP_S
        created_atom_ids: list[str] = []
        created_edge_ids: list[str] = []

        user_atom = make_evidence_atom(
            subject_id=subject_id,
            event_kind="user_turn",
            role="user",
            text=text,
            now_ts=user_timestamp,
        )
        self.store.add_atom(user_atom)
        created_atom_ids.append(user_atom.atom_id)

        try:
            user_atoms, user_edges = self.compiler.compile_user_turn(
                subject_id=subject_id,
                user_atom=user_atom,
                existing_atoms=self.store.list_atoms(subject_id),
                now_ts=user_timestamp,
            )
            self._apply_field_writes(subject_id=subject_id, atoms=user_atoms, edges=user_edges)
            created_atom_ids.extend(atom.atom_id for atom in user_atoms)
            created_edge_ids.extend(edge.edge_id for edge in user_edges)
        except Exception as exc:
            failure = self._record_compile_failure(subject_id=subject_id, reason=exc, now_ts=user_timestamp)
            created_atom_ids.append(failure.atom_id)

        self.field.evolve(subject_id)

        recall_result = self.recall(subject_id, text)
        state = self.state(subject_id)
        memory_brief = build_memory_brief(state, recall_result)
        response_text = self.responder.respond(
            ExpressionContext(
                input_text=text,
                memory_brief=memory_brief,
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
        created_atom_ids.append(assistant_atom.atom_id)

        try:
            turn_atoms, turn_edges = self.compiler.compile_completed_turn(
                subject_id=subject_id,
                user_atom=user_atom,
                assistant_atom=assistant_atom,
                existing_atoms=self.store.list_atoms(subject_id),
                now_ts=assistant_timestamp,
            )
            self._apply_field_writes(subject_id=subject_id, atoms=turn_atoms, edges=turn_edges)
            created_atom_ids.extend(atom.atom_id for atom in turn_atoms)
            created_edge_ids.extend(edge.edge_id for edge in turn_edges)
        except Exception as exc:
            failure = self._record_compile_failure(subject_id=subject_id, reason=exc, now_ts=assistant_timestamp)
            created_atom_ids.append(failure.atom_id)

        self.field.evolve(subject_id)
        return TurnOutput(
            turn_id=user_atom.atom_id,
            subject_id=subject_id,
            response_text=response_text,
            recall_used=bool(recall_result.atoms),
            created_atom_ids=tuple(created_atom_ids),
            created_edge_ids=tuple(created_edge_ids),
        )

    def state(self, subject_id: str) -> SubjectMemoryState:
        atoms = self.store.list_atoms(subject_id)
        edges = self.store.list_edges(subject_id)
        activations = self.store.list_activation_cache(subject_id)
        if atoms and not activations:
            activations = self.field.evolve(subject_id)
        return project_subject_state(subject_id=subject_id, atoms=atoms, edges=edges, activations=activations)

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        limit: int = 8,
    ) -> RecallResult:
        return self.field.recall(subject_id, query, limit=limit)

    def close(self) -> None:
        self.store.close()

    def _apply_field_writes(
        self,
        *,
        subject_id: str,
        atoms: tuple[MemoryAtom, ...],
        edges: tuple[MemoryEdge, ...],
    ) -> None:
        if not atoms and not edges:
            return
        with self.store.transaction():
            for atom in atoms:
                self.store.add_atom(atom)
            for edge in edges:
                self.store.add_edge(edge)

    def _record_compile_failure(self, *, subject_id: str, reason: Exception, now_ts: float) -> MemoryAtom:
        message = str(reason) or reason.__class__.__name__
        failure = make_evidence_atom(
            subject_id=subject_id,
            event_kind="compile_failure",
            role="system",
            text=message,
            now_ts=now_ts,
            payload={"reason": message},
        )
        self.store.add_atom(failure)
        self.field.evolve(subject_id)
        return failure
