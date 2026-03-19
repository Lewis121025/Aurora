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
from aurora.memory.ledger import HashEmbeddingEncoder, MemoryField, tokenize
from aurora.memory.store import SQLiteMemoryStore
from aurora.pipelines.distillation import MemoryCompiler, make_evidence_atom
from aurora.runtime.contracts import (
    IngestOutput,
    MemoryAtom,
    MemoryEdge,
    RecallResult,
    SubjectMemoryState,
    EventKind,
    TranscriptItem,
    TurnOutput,
)
from aurora.runtime.projections import build_memory_brief

_TURN_STEP_S = 1e-6
_SEGMENT_TOKEN_CAP = 1024
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

    def turn(
        self,
        subject_id: str,
        session_id: str,
        text: str,
        now_ts: float | None = None,
    ) -> TurnOutput:
        user_timestamp = time.time() if now_ts is None else now_ts
        session = self.store.ensure_session(subject_id, session_id, started_at=user_timestamp)
        if session.finalized_at is not None:
            raise RuntimeError(f"session {session_id} is already finalized")
        segment = self.store.ensure_open_segment(subject_id, session_id, started_at=user_timestamp)
        user_atom = self._append_session_evidence(
            subject_id=subject_id,
            session_id=session_id,
            segment_id=segment.segment_id,
            event_kind="user_turn",
            role="user",
            text=text,
            now_ts=user_timestamp,
        )

        route = self.field.route(subject_id, text, now_ts=user_timestamp, touch=True)
        recall_result = self.field.recall(subject_id, text, route=route)
        state = self.state(subject_id)
        memory_brief = build_memory_brief(state, recall_result)
        session_transcript = self._render_segment_transcript(subject_id, session_id, segment.segment_id)
        response_text = self.responder.respond(
            ExpressionContext(
                input_text=text,
                memory_brief=memory_brief,
                session_transcript=session_transcript,
            )
        )

        assistant_timestamp = user_timestamp + _TURN_STEP_S
        self._append_session_evidence(
            subject_id=subject_id,
            session_id=session_id,
            segment_id=segment.segment_id,
            event_kind="assistant_turn",
            role="assistant",
            text=response_text,
            now_ts=assistant_timestamp,
        )
        segment_committed = self._maybe_commit_segment(
            subject_id=subject_id,
            session_id=session_id,
            segment_id=segment.segment_id,
            now_ts=assistant_timestamp,
        )
        return TurnOutput(
            turn_id=user_atom.atom_id,
            subject_id=subject_id,
            session_id=session_id,
            response_text=response_text,
            recall_used=bool(recall_result.atoms),
            segment_committed=segment_committed,
        )

    def finalize_session(
        self,
        subject_id: str,
        session_id: str,
        ended_at: float | None = None,
    ) -> IngestOutput:
        timestamp = time.time() if ended_at is None else ended_at
        session = self.store.get_session(subject_id, session_id)
        if session is None:
            raise RuntimeError(f"unknown session {session_id}")
        if session.finalized_at is not None:
            raise RuntimeError(f"session {session_id} is already finalized")
        segment = self.store.get_open_segment(subject_id, session_id)
        if segment is None:
            self.store.close_session(subject_id, session_id, finalized_at=timestamp)
            return IngestOutput(subject_id=subject_id, session_id=session_id)
        self.store.close_segment(
            subject_id,
            session_id,
            segment.segment_id,
            closed_at=timestamp,
            closed_reason="finalize",
        )
        output = self._ingest_segment(
            subject_id=subject_id,
            session_id=session_id,
            segment_id=segment.segment_id,
            now_ts=timestamp,
        )
        self.store.close_session(subject_id, session_id, finalized_at=timestamp)
        return output

    def ingest_transcript(
        self,
        subject_id: str,
        session_id: str,
        transcript: tuple[TranscriptItem, ...],
        ended_at: float | None = None,
    ) -> IngestOutput:
        if not transcript:
            raise RuntimeError("transcript must not be empty")
        if self.store.get_session(subject_id, session_id) is not None:
            raise RuntimeError(f"session {session_id} already exists")
        first_timestamp = transcript[0].created_at
        final_timestamp = transcript[-1].created_at if ended_at is None else ended_at
        self.store.ensure_session(subject_id, session_id, started_at=first_timestamp)
        segment = self.store.create_segment(subject_id, session_id, started_at=first_timestamp)
        for item in transcript:
            event_kind: EventKind = "user_turn" if item.role == "user" else "assistant_turn"
            self._append_session_evidence(
                subject_id=subject_id,
                session_id=session_id,
                segment_id=segment.segment_id,
                event_kind=event_kind,
                role=item.role,
                text=item.text,
                now_ts=item.created_at,
            )
        self.store.close_segment(
            subject_id,
            session_id,
            segment.segment_id,
            closed_at=final_timestamp,
            closed_reason="ingest",
        )
        output = self._ingest_segment(
            subject_id=subject_id,
            session_id=session_id,
            segment_id=segment.segment_id,
            now_ts=final_timestamp,
        )
        self.store.close_session(subject_id, session_id, finalized_at=final_timestamp)
        return output

    def state(self, subject_id: str) -> SubjectMemoryState:
        return self.field.state(subject_id)

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
        return failure

    def _append_session_evidence(
        self,
        *,
        subject_id: str,
        session_id: str,
        segment_id: int,
        event_kind: EventKind,
        role: str,
        text: str,
        now_ts: float,
    ) -> MemoryAtom:
        atom = make_evidence_atom(
            subject_id=subject_id,
            event_kind=event_kind,
            role=role,
            text=text,
            now_ts=now_ts,
            payload={"session_id": session_id, "segment_id": segment_id},
        )
        token_count_increment = len(tokenize(text))
        with self.store.transaction():
            self.store.add_atom(atom)
            self.store.append_session_event(
                subject_id,
                session_id,
                segment_id,
                atom.atom_id,
                token_count_increment=token_count_increment,
            )
        return atom

    def _maybe_commit_segment(
        self,
        *,
        subject_id: str,
        session_id: str,
        segment_id: int,
        now_ts: float,
    ) -> bool:
        segment = self.store.get_open_segment(subject_id, session_id)
        if segment is None or segment.segment_id != segment_id or segment.token_count < _SEGMENT_TOKEN_CAP:
            return False
        self.store.close_segment(
            subject_id,
            session_id,
            segment_id,
            closed_at=now_ts,
            closed_reason="token_cap",
        )
        self._ingest_segment(subject_id=subject_id, session_id=session_id, segment_id=segment_id, now_ts=now_ts)
        return True

    def _ingest_segment(
        self,
        *,
        subject_id: str,
        session_id: str,
        segment_id: int,
        now_ts: float,
    ) -> IngestOutput:
        transcript = self.store.list_segment_transcript(subject_id, session_id, segment_id)
        if not transcript:
            return IngestOutput(subject_id=subject_id, session_id=session_id)
        transcript_text = _transcript_text(transcript)
        route = self.field.route(subject_id, transcript_text, now_ts=now_ts, touch=True)
        try:
            local_atoms, stable_field_summary = self.field.build_compile_context(subject_id, transcript_text, route)
            atoms, edges = self.compiler.compile_session(
                subject_id=subject_id,
                transcript=transcript,
                local_atoms=local_atoms,
                stable_field_summary=stable_field_summary,
                now_ts=now_ts,
            )
            self._apply_field_writes(subject_id=subject_id, atoms=atoms, edges=edges)
            self.field.integrate(subject_id, route, atoms, edges, now_ts=now_ts)
            return IngestOutput(
                subject_id=subject_id,
                session_id=session_id,
                created_atom_ids=tuple(atom.atom_id for atom in atoms),
                created_edge_ids=tuple(edge.edge_id for edge in edges),
            )
        except Exception as exc:
            failure = self._record_compile_failure(subject_id=subject_id, reason=exc, now_ts=now_ts)
            return IngestOutput(
                subject_id=subject_id,
                session_id=session_id,
                created_atom_ids=(failure.atom_id,),
            )

    def _render_segment_transcript(self, subject_id: str, session_id: str, segment_id: int) -> str:
        return _render_session_transcript(self.store.list_segment_transcript(subject_id, session_id, segment_id))


def _transcript_text(transcript: tuple[TranscriptItem, ...]) -> str:
    return "\n".join(f"{item.role}: {item.text}" for item in transcript)


def _render_session_transcript(transcript: tuple[TranscriptItem, ...]) -> str:
    lines = ["[ACTIVE_SESSION_TRANSCRIPT]"]
    if not transcript:
        lines.append("- none")
        return "\n".join(lines)
    for item in transcript:
        lines.append(f"- {item.role}: {item.text}")
    return "\n".join(lines)
