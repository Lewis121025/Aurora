"""Public Aurora system facade."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from aurora.expression.cognition import DEFAULT_SYSTEM_PROMPT, Responder
from aurora.expression.context import ExpressionContext
from aurora.field_engine import EventIngestResult, MemoryKernel, RecallResult
from aurora.llm.config import LLMSettings, coerce_llm_settings, load_llm_settings
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.projections import build_memory_brief
from aurora.store import SQLiteSnapshotStore

_OPENAI_COMPAT_PROVIDERS = frozenset({"openai", "openai_compatible", "bailian"})


@dataclass(frozen=True)
class ResponseOutput:
    session_id: str
    response_text: str
    user_event_id: str
    assistant_event_id: str
    memory_brief: str


@dataclass
class AuroraSystemConfig:
    db_path: str = ".aurora/aurora.sqlite"
    seed: int = 7
    autosave: bool = True
    save_on_retrieve: bool = True
    max_snapshots: int = 256
    background_replay_interval: float = 5.0
    background_replay_budget: int = 6
    session_context_messages: int = 12


def build_llm_provider(settings: LLMSettings) -> LLMProvider:
    if settings.provider in _OPENAI_COMPAT_PROVIDERS:
        return OpenAICompatProvider(settings.config)
    raise RuntimeError(
        "Unsupported AURORA_LLM_PROVIDER. "
        "Supported providers: openai, openai_compatible, bailian."
    )


class AuroraSystem:
    """Thread-safe Aurora service built on the unified memory field."""

    def __init__(
        self,
        config: AuroraSystemConfig | None = None,
        *,
        llm: LLMProvider | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.config = config or AuroraSystemConfig()
        self._lock = threading.RLock()
        self._store = SQLiteSnapshotStore(self.config.db_path, max_snapshots=self.config.max_snapshots)
        self.kernel = self._store.load_latest_kernel() or MemoryKernel(seed=self.config.seed)
        self._responder = Responder(llm, system_prompt=system_prompt) if llm is not None else None
        self._stop_replay = threading.Event()
        self._replay_thread: Optional[threading.Thread] = None

    @classmethod
    def create(
        cls,
        data_dir: str | None = None,
        llm: LLMProvider | None = None,
        llm_settings: LLMSettings | Mapping[str, object] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> "AuroraSystem":
        if llm is not None and llm_settings is not None:
            raise ValueError("Pass either llm or llm_settings, not both")
        provider = llm
        if provider is None:
            settings = coerce_llm_settings(llm_settings) if llm_settings is not None else load_llm_settings()
            if settings is not None:
                provider = build_llm_provider(settings)
        db_path = str(Path(data_dir or ".aurora") / "aurora.sqlite")
        return cls(
            AuroraSystemConfig(db_path=db_path),
            llm=provider,
            system_prompt=system_prompt,
        )

    def close(self) -> None:
        self.stop_background_replay()
        with self._lock:
            if self.config.autosave:
                self._save("shutdown", {"message": "graceful shutdown"})
            self._store.close()

    def start_background_replay(self, interval: Optional[float] = None, budget: Optional[int] = None) -> None:
        replay_interval = float(interval or self.config.background_replay_interval)
        replay_budget = int(budget or self.config.background_replay_budget)
        if self._replay_thread is not None and self._replay_thread.is_alive():
            return
        self._stop_replay.clear()

        def loop() -> None:
            while not self._stop_replay.wait(replay_interval):
                try:
                    self.replay(budget=replay_budget, reason="background_replay")
                except Exception:
                    pass

        self._replay_thread = threading.Thread(target=loop, name="aurora-replay", daemon=True)
        self._replay_thread.start()

    def stop_background_replay(self) -> None:
        self._stop_replay.set()
        if self._replay_thread is not None and self._replay_thread.is_alive():
            self._replay_thread.join(timeout=1.0)
        self._replay_thread = None

    def ingest(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "dialogue",
        *,
        now_ts: float | None = None,
    ) -> EventIngestResult:
        with self._lock:
            result = self.kernel.ingest(text, metadata=metadata, source=source, now_ts=now_ts)
            self._save(
                "ingest",
                {
                    "event_id": result.event_id,
                    "anchor_id": result.anchor_id,
                    "atom_ids": list(result.atom_ids),
                    "source": source,
                    "text": text,
                    "metadata": metadata or {},
                },
            )
            return result

    def ingest_batch(
        self,
        events: List[Dict[str, Any]],
        source: str = "dialogue",
    ) -> Dict[str, Any]:
        with self._lock:
            results: List[Dict[str, Any]] = []
            for event in events:
                text = str(event.get("text", ""))
                metadata = dict(event.get("metadata") or {})
                event_source = str(event.get("source", source))
                now_ts = event.get("now_ts")
                result = self.kernel.ingest(text, metadata=metadata, source=event_source, now_ts=now_ts if isinstance(now_ts, (int, float)) else None)
                results.append(
                    {
                        "event_id": result.event_id,
                        "anchor_id": result.anchor_id,
                        "atom_ids": list(result.atom_ids),
                        "text": text,
                        "metadata": metadata,
                        "source": event_source,
                    }
                )
            payload = {"count": len(results), "results": results}
            self._save("ingest_batch", payload)
            return payload

    def retrieve(self, cue: str, top_k: int = 8, propagation_steps: int = 3) -> RecallResult:
        with self._lock:
            result = self._retrieve_locked(cue, top_k=top_k, propagation_steps=propagation_steps, save=True)
            return result

    def current_state(self, top_k: int = 10) -> RecallResult:
        with self._lock:
            return self._current_state_locked(top_k=top_k, save=True)

    def replay(self, budget: int = 8, reason: str = "replay") -> Dict[str, Any]:
        with self._lock:
            traces = self.kernel.replay(budget=budget)
            payload = {"budget": budget, "trace_count": len(traces), "traces": traces}
            self._save(reason, payload)
            return payload

    def respond(
        self,
        session_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "dialogue",
        *,
        top_k: int = 8,
        propagation_steps: int = 3,
        now_ts: float | None = None,
    ) -> ResponseOutput:
        with self._lock:
            if self._responder is None:
                raise RuntimeError("AuroraSystem.respond requires llm or llm_settings")
            timestamp = time.time() if now_ts is None else now_ts
            user_metadata = {"speaker": "user", "epistemic_mode": "asserted", **(metadata or {})}
            user_result = self.kernel.ingest(text, metadata=user_metadata, source=source, now_ts=timestamp)
            self._store.append_session_turn(session_id, "user", text, event_id=user_result.event_id, created_at=timestamp)

            try:
                state = self._current_state_locked(top_k=top_k, save=False)
                recall = self._retrieve_locked(text, top_k=top_k, propagation_steps=propagation_steps, save=False)
                memory_brief = build_memory_brief(state, recall)
                response_text = self._responder.respond(
                    ExpressionContext(
                        input_text=text,
                        memory_brief=memory_brief,
                        session_transcript=self._render_session_transcript(session_id),
                    )
                )
                assistant_timestamp = timestamp + 1e-6
                assistant_metadata = {"speaker": "assistant", "epistemic_mode": "derived"}
                assistant_result = self.kernel.ingest(
                    response_text,
                    metadata=assistant_metadata,
                    source=source,
                    now_ts=assistant_timestamp,
                )
                self._store.append_session_turn(
                    session_id,
                    "assistant",
                    response_text,
                    event_id=assistant_result.event_id,
                    created_at=assistant_timestamp,
                )
                self._save(
                    "respond",
                    {
                        "session_id": session_id,
                        "user_event_id": user_result.event_id,
                        "assistant_event_id": assistant_result.event_id,
                        "text": text,
                        "response_text": response_text,
                    },
                )
                return ResponseOutput(
                    session_id=session_id,
                    response_text=response_text,
                    user_event_id=user_result.event_id,
                    assistant_event_id=assistant_result.event_id,
                    memory_brief=memory_brief,
                )
            except Exception as exc:
                self._save(
                    "respond_failure",
                    {
                        "session_id": session_id,
                        "user_event_id": user_result.event_id,
                        "text": text,
                        "error": str(exc) or exc.__class__.__name__,
                    },
                )
                raise

    def operation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._store.list_operations(limit=limit)

    def get_atom(self, atom_id: str) -> Dict[str, Any]:
        with self._lock:
            atom = self.kernel.atoms.get(atom_id)
            if atom is None:
                raise KeyError(atom_id)
            out_edges = [asdict(edge) for edge in self.kernel.out_edges.get(atom_id, {}).values()]
            in_edges = [asdict(edge) for edge in self.kernel.in_edges.get(atom_id, {}).values()]
            return {
                "atom_id": atom_id,
                "text": self.kernel._display_text(atom_id),
                "core": {
                    **asdict(atom.core),
                    "address": {head: list(tokens) for head, tokens in atom.core.address.items()},
                },
                "state": asdict(atom.state),
                "out_edges": out_edges,
                "in_edges": in_edges,
            }

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            atoms = list(self.kernel.atoms.values())
            atom_counter = Counter(atom.core.kind for atom in atoms)
            edge_counter = Counter(edge.kind for edge in self.kernel._iter_edges())
            hot_atoms = sorted(
                atoms,
                key=lambda atom: (
                    atom.state.activation * 0.30
                    + atom.state.stability * 0.25
                    + atom.state.confidence * 0.20
                    + atom.state.utility * 0.15
                    + atom.state.fidelity * 0.10
                ),
                reverse=True,
            )[:12]
            def average(fn: Callable[[Any], float]) -> float:
                return round(sum(fn(atom) for atom in atoms) / max(1, len(atoms)), 4)

            return {
                "db_path": str(Path(self.config.db_path).resolve()),
                "step": self.kernel.step,
                "atom_count": len(self.kernel.atoms),
                "edge_count": sum(1 for _ in self.kernel._iter_edges()),
                "atoms_by_kind": dict(atom_counter),
                "edges_by_kind": dict(edge_counter),
                "event_count": len(self.kernel.event_members),
                "abstract_count": atom_counter.get("abstract", 0),
                "conflict_edge_count": edge_counter.get("contradicts", 0) + edge_counter.get("suppresses", 0),
                "avg_activation": average(lambda atom: atom.state.activation),
                "avg_stability": average(lambda atom: atom.state.stability),
                "avg_confidence": average(lambda atom: atom.state.confidence),
                "avg_utility": average(lambda atom: atom.state.utility),
                "avg_fidelity": average(lambda atom: atom.state.fidelity),
                "snapshots": self._store.snapshot_count(),
                "latest_snapshot": self._store.latest_snapshot_meta(),
                "session_count": self._store.session_count(),
                "hot_atoms": [
                    {
                        "atom_id": atom.core.atom_id,
                        "kind": atom.core.kind,
                        "text": self.kernel._display_text(atom.core.atom_id),
                        "activation": round(atom.state.activation, 4),
                        "stability": round(atom.state.stability, 4),
                        "confidence": round(atom.state.confidence, 4),
                    }
                    for atom in hot_atoms
                ],
                "background_replay_running": bool(self._replay_thread is not None and self._replay_thread.is_alive()),
            }

    def export_snapshot_json(self, path: str | Path) -> Path:
        with self._lock:
            return self._store.export_latest_json(path)

    def _retrieve_locked(self, cue: str, *, top_k: int, propagation_steps: int, save: bool) -> RecallResult:
        result = self.kernel.retrieve(cue, top_k=top_k, propagation_steps=propagation_steps)
        if save and self.config.autosave and self.config.save_on_retrieve:
            self._save(
                "retrieve",
                {
                    "cue": cue,
                    "top_k": top_k,
                    "propagation_steps": propagation_steps,
                    "top_ids": [item.atom_id for item in result.items],
                },
            )
        return result

    def _current_state_locked(self, *, top_k: int, save: bool) -> RecallResult:
        result = self.kernel.current_state(top_k=top_k)
        if save and self.config.autosave and self.config.save_on_retrieve:
            self._save(
                "current_state",
                {"top_k": top_k, "top_ids": [item.atom_id for item in result.items]},
            )
        return result

    def _save(self, reason: str, payload: Dict[str, Any]) -> None:
        if not self.config.autosave:
            return
        self._store.save_snapshot(self.kernel, reason=reason, operation_summary=payload)

    def _render_session_transcript(self, session_id: str) -> str:
        turns = self._store.list_session_turns(session_id, limit=self.config.session_context_messages)
        if not turns:
            return "[SESSION_TRANSCRIPT]\n- none"
        lines = ["[SESSION_TRANSCRIPT]"]
        for turn in turns:
            lines.append(f"- {turn.role}: {turn.text}")
        return "\n".join(lines)


def recall_result_to_dict(result: RecallResult) -> Dict[str, Any]:
    return {
        "cue": result.cue,
        "items": [asdict(item) for item in result.items],
        "edges": [asdict(edge) for edge in result.edges],
        "trace": result.trace,
    }


def response_output_to_dict(result: ResponseOutput) -> Dict[str, Any]:
    return asdict(result)


def event_ingest_to_dict(result: EventIngestResult) -> Dict[str, Any]:
    return asdict(result)


__all__ = [
    "AuroraSystem",
    "AuroraSystemConfig",
    "ResponseOutput",
    "build_llm_provider",
    "event_ingest_to_dict",
    "recall_result_to_dict",
    "response_output_to_dict",
]
