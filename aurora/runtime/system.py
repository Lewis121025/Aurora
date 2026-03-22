"""Aurora runtime orchestration and provider construction."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any

from aurora.core.config import FieldConfig
from aurora.core.types import DecoderRequest, ResponseResult, Workspace
from aurora.expression.cognition import DEFAULT_SYSTEM_PROMPT, Responder
from aurora.expression.context import ExpressionContext
from aurora.expression.projections import render_workspace_for_llm
from aurora.llm.config import LLMSettings, coerce_llm_settings, load_llm_settings
from aurora.llm.openai_compat import OpenAICompatProvider
from aurora.llm.provider import LLMProvider
from aurora.models import build_local_decoder
from aurora.runtime.field import AuroraField
from aurora.store import SQLiteSnapshotStore

_OPENAI_COMPAT_PROVIDERS = frozenset({"openai", "openai_compatible", "bailian"})


@dataclass(frozen=True, slots=True)
class AuroraSystemConfig:
    db_path: str = ".aurora/aurora.sqlite"
    data_dir: str = ".aurora"
    blob_dir: str = ".aurora/blobs"
    autosave: bool = True
    max_snapshots: int = 256
    encoder_dim: int = 128
    packet_chars: int = 512
    ann_top_k: int = 64
    hot_trace_limit: int = 32
    settle_steps: int = 4
    workspace_k: int = 12
    maintenance_ms_budget: int = 20
    trace_budget: int = 256
    edge_budget: int = 2048
    anchor_budget: int = 1024
    default_encoder_model: str = "intfloat/multilingual-e5-small"
    default_decoder_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_decoder_backend: str = "transformers"
    background_maintenance_interval: float = 5.0
    background_maintenance_budget: int = 6

    def field_config(self) -> FieldConfig:
        return FieldConfig(
            data_dir=self.data_dir,
            db_path=self.db_path,
            blob_dir=self.blob_dir,
            latent_dim=self.encoder_dim,
            packet_chars=self.packet_chars,
            candidate_size=self.ann_top_k,
            frontier_size=self.hot_trace_limit,
            settle_steps=self.settle_steps,
            workspace_size=self.workspace_k,
            maintenance_ms_budget=self.maintenance_ms_budget,
            trace_budget=self.trace_budget,
            edge_budget=self.edge_budget,
            anchor_budget=self.anchor_budget,
            default_encoder_model=self.default_encoder_model,
            default_decoder_model=self.default_decoder_model,
            local_decoder_backend=self.local_decoder_backend,
            max_snapshots=self.max_snapshots,
        )


@dataclass(frozen=True, slots=True)
class FieldStats:
    db_path: str
    step: int
    packet_count: int
    anchor_count: int
    trace_count: int
    edge_count: int
    posterior_group_count: int
    hot_trace_ids: tuple[str, ...]
    budget_state: dict[str, Any]
    snapshot_count: int
    latest_snapshot: dict[str, Any]
    background_maintenance_running: bool


def build_llm_provider(settings: LLMSettings) -> LLMProvider:
    if settings.provider in _OPENAI_COMPAT_PROVIDERS:
        return OpenAICompatProvider(settings.config)
    raise RuntimeError(
        "Unsupported AURORA_LLM_PROVIDER. Supported providers: openai, openai_compatible, bailian."
    )


def to_dict(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return {field_name: to_dict(getattr(value, field_name)) for field_name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_dict(item) for item in value]
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    return value


class AuroraSystem:
    """Thread-safe runtime orchestration over the Aurora trace field."""

    def __init__(
        self,
        config: AuroraSystemConfig | None = None,
        *,
        llm: LLMProvider | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.config = _normalize_config(config or AuroraSystemConfig())
        self._lock = threading.RLock()
        self._store = SQLiteSnapshotStore(self.config.db_path, max_snapshots=self.config.max_snapshots)
        self.field = self._store.load_latest_field() or AuroraField(self.config.field_config())
        self._responder = Responder(llm, system_prompt=system_prompt) if llm is not None else None
        decoder_kwargs: dict[str, Any] = {}
        if self.config.local_decoder_backend == "transformers":
            decoder_kwargs["model_id"] = self.config.default_decoder_model
        self._local_decoder = build_local_decoder(kind=self.config.local_decoder_backend, **decoder_kwargs)
        self._stop_maintenance = threading.Event()
        self._maintenance_thread: threading.Thread | None = None
        self._closed = False

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
        root = Path(data_dir or ".aurora")
        config = AuroraSystemConfig(
            data_dir=str(root),
            db_path=str(root / "aurora.sqlite"),
            blob_dir=str(root / "blobs"),
        )
        return cls(config, llm=provider, system_prompt=system_prompt)

    def close(self) -> None:
        if self._closed:
            return
        self.stop_background_maintenance()
        with self._lock:
            if self.config.autosave:
                self._save("shutdown", {"message": "graceful shutdown"})
            self._store.close()
            self._closed = True

    def start_background_maintenance(self, interval: float | None = None, ms_budget: int | None = None) -> None:
        maintenance_interval = float(interval or self.config.background_maintenance_interval)
        maintenance_budget = int(ms_budget or self.config.background_maintenance_budget)
        if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
            return
        self._stop_maintenance.clear()

        def loop() -> None:
            while not self._stop_maintenance.wait(maintenance_interval):
                self.maintenance_cycle(ms_budget=maintenance_budget)

        self._maintenance_thread = threading.Thread(target=loop, name="aurora-maintenance", daemon=True)
        self._maintenance_thread.start()

    def stop_background_maintenance(self) -> None:
        self._stop_maintenance.set()
        if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=1.0)
        self._maintenance_thread = None

    def inject(self, raw_event: Mapping[str, Any] | str) -> Any:
        with self._lock:
            result = self.field.inject(_normalize_raw_event(raw_event))
            self._save("inject", to_dict(result))
            return result

    def maintenance_cycle(self, ms_budget: int | None = None) -> Any:
        with self._lock:
            result = self.field.maintenance_cycle(ms_budget=ms_budget)
            self._save("maintenance_cycle", to_dict(result))
            return result

    def read_workspace(self, cue: str | Mapping[str, Any], k: int | None = None) -> Workspace:
        with self._lock:
            return self.field.read_workspace(dict(cue) if isinstance(cue, Mapping) else cue, k=k)

    def respond(self, cue: str | Mapping[str, Any]) -> ResponseResult:
        event = _normalize_raw_event(cue)
        cue_text = str(event.get("payload") or "")
        with self._lock:
            field_snapshot = self.field.to_snapshot_payload()
            try:
                user_result = self.field.inject(event)
                self.field.maintenance_cycle(ms_budget=self.config.maintenance_ms_budget)
                workspace = self.field.read_workspace(event, k=self.config.workspace_k)
                rendered_workspace = render_workspace_for_llm(workspace, cue=cue_text)
                response_text = self._generate_response(cue_text, workspace, rendered_workspace)
                assistant_event = {
                    "session_id": str(event.get("session_id") or ""),
                    "turn_id": f"response-{uuid.uuid4().hex[:8]}",
                    "source": "assistant",
                    "payload_type": "text",
                    "payload": response_text,
                    "meta": {"response_to": cue_text},
                }
                assistant_result = self.field.inject(assistant_event)
                result = ResponseResult(
                    response_text=response_text,
                    workspace=workspace,
                    trace_ids=[*user_result.trace_ids, *assistant_result.trace_ids],
                    anchor_ids=[*user_result.anchor_ids, *assistant_result.anchor_ids],
                    response_vector=workspace.summary_vector,
                )
                self._save("respond", to_dict(result))
                return result
            except Exception:
                self.field.restore_from_snapshot_payload(field_snapshot)
                raise

    def snapshot(self) -> Any:
        with self._lock:
            result = self.field.snapshot()
            self._save("snapshot", {"path": result.snapshot_path})
            return result

    def field_stats(self) -> FieldStats:
        raw = self.field.field_stats()
        return FieldStats(
            db_path=str(self._store.db_path.resolve()),
            step=int(raw.get("step", 0)),
            packet_count=int(raw.get("packet_count", 0)),
            anchor_count=int(raw.get("anchor_count", 0)),
            trace_count=int(raw.get("trace_count", 0)),
            edge_count=int(raw.get("edge_count", 0)),
            posterior_group_count=int(raw.get("posterior_group_count", 0)),
            hot_trace_ids=tuple(raw.get("hot_trace_ids", [])),
            budget_state=dict(raw.get("budget_state", {})),
            snapshot_count=self._store.snapshot_count(),
            latest_snapshot=self._store.latest_snapshot_meta(),
            background_maintenance_running=bool(
                self._maintenance_thread is not None and self._maintenance_thread.is_alive()
            ),
        )

    def _generate_response(self, cue_text: str, workspace: Workspace, rendered_workspace: str) -> str:
        if self._responder is not None:
            return self._responder.respond(
                ExpressionContext(
                    input_text=cue_text,
                    workspace=workspace,
                    rendered_workspace=rendered_workspace,
                )
            )
        request = DecoderRequest(
            cue=cue_text,
            workspace=workspace,
            active_trace_ids=workspace.active_trace_ids,
            anchor_refs=workspace.anchor_refs,
            prompt=rendered_workspace,
        )
        return self._local_decoder.decode(request).text

    def _save(self, reason: str, payload: dict[str, Any]) -> None:
        if not self.config.autosave:
            return
        self._store.save_snapshot(self.field, reason=reason, operation_summary=payload)


def _normalize_raw_event(raw_event: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(raw_event, str):
        return {
            "session_id": "",
            "turn_id": f"turn-{uuid.uuid4().hex[:10]}",
            "source": "user",
            "payload_type": "text",
            "payload": raw_event,
            "meta": {},
        }
    payload = str(raw_event.get("payload") or raw_event.get("text") or "")
    if not payload.strip():
        raise ValueError("payload must not be empty")
    return {
        "ts": raw_event.get("ts"),
        "session_id": str(raw_event.get("session_id") or ""),
        "turn_id": str(raw_event.get("turn_id") or f"turn-{uuid.uuid4().hex[:10]}"),
        "source": _normalize_source(raw_event.get("source") or raw_event.get("role") or "user"),
        "payload_type": str(raw_event.get("payload_type") or "text"),
        "payload": payload.strip(),
        "meta": dict(raw_event.get("meta") or raw_event.get("metadata") or {}),
    }


def _normalize_source(value: object) -> str:
    normalized = str(value or "user").strip().lower()
    if normalized in {"user", "assistant", "tool", "env"}:
        return normalized
    raise ValueError("source must be one of user, assistant, tool, or env")


def _normalize_config(config: AuroraSystemConfig) -> AuroraSystemConfig:
    db_path = Path(config.db_path)
    root = db_path.parent
    data_dir = config.data_dir
    blob_dir = config.blob_dir
    if data_dir == ".aurora":
        data_dir = str(root)
    if blob_dir == ".aurora/blobs":
        blob_dir = str(root / "blobs")
    return AuroraSystemConfig(
        db_path=str(db_path),
        data_dir=data_dir,
        blob_dir=blob_dir,
        autosave=config.autosave,
        max_snapshots=config.max_snapshots,
        encoder_dim=config.encoder_dim,
        packet_chars=config.packet_chars,
        ann_top_k=config.ann_top_k,
        hot_trace_limit=config.hot_trace_limit,
        settle_steps=config.settle_steps,
        workspace_k=config.workspace_k,
        maintenance_ms_budget=config.maintenance_ms_budget,
        trace_budget=config.trace_budget,
        edge_budget=config.edge_budget,
        anchor_budget=config.anchor_budget,
        default_encoder_model=config.default_encoder_model,
        default_decoder_model=config.default_decoder_model,
        local_decoder_backend=config.local_decoder_backend,
        background_maintenance_interval=config.background_maintenance_interval,
        background_maintenance_budget=config.background_maintenance_budget,
    )


__all__ = [
    "AuroraSystem",
    "AuroraSystemConfig",
    "FieldStats",
    "build_llm_provider",
    "to_dict",
]
