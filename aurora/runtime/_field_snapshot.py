"""Snapshot and serialization helpers for AuroraField."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

import numpy as np

from aurora.core.config import BudgetConfig, FieldConfig, PredictorConfig, ReplayConfig
from aurora.core.types import (
    ActionName,
    Anchor,
    EdgeKind,
    ExperienceFrame,
    PayloadType,
    PosteriorGroup,
    SnapshotMeta,
    SourceType,
    TraceEdge,
    TraceRecord,
    Workspace,
)
from aurora.ingest.packetizer import PacketRecord

if TYPE_CHECKING:
    from aurora.runtime.field import AuroraField


class FieldSnapshotMixin:
    def snapshot(self: Any) -> SnapshotMeta:
        target = Path(self.config.data_dir) / "snapshots" / f"field-{self.step:08d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_snapshot_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        return SnapshotMeta(
            snapshot_path=str(target),
            trace_count=len(self.traces),
            anchor_count=len(self.anchors),
            edge_count=len(self.edges),
        )

    def field_stats(self: Any) -> dict[str, Any]:
        trace_mass, edge_mass, group_mass = self._budget_masses()
        return {
            "step": self.step,
            "current_ts": self.current_ts,
            "packet_count": len(self.anchor_store.packets),
            "anchor_count": len(self.anchors),
            "trace_count": len(self.traces),
            "edge_count": len(self.edges),
            "posterior_group_count": len(self.groups),
            "hot_trace_ids": list(self.hot_trace_ids),
            "objective": {
                "ema": float(self.objective_ema),
                "last": dict(self.last_objective),
            },
            "budget_state": {
                "max_traces": self.budget_config.max_traces,
                "max_edges": self.budget_config.max_edges,
                "max_groups": self.budget_config.max_groups,
                "trace_mass": trace_mass,
                "edge_mass": edge_mass,
                "group_mass": group_mass,
                "pressure": self._budget_pressure(),
            },
        }

    def to_snapshot_payload(self: Any) -> dict[str, Any]:
        return {
            "schema_version": 6,
            "step": self.step,
            "current_ts": self.current_ts,
            "config": _jsonable(asdict(self.config)),
            "replay_config": _jsonable(asdict(self.replay_config)),
            "budget_config": _jsonable(asdict(self.budget_config)),
            "predictor_config": _jsonable(asdict(self.predictor_config)),
            "packet_records": [_jsonable(asdict(packet)) for packet in self.anchor_store.packets.values()],
            "anchors": [_jsonable(asdict(anchor)) for anchor in self.anchors.values()],
            "traces": [_jsonable(asdict(trace)) for trace in self.traces.values()],
            "edges": [_jsonable(asdict(edge)) for edge in self.edges.values()],
            "posterior_groups": [_jsonable(asdict(group)) for group in self.groups.values()],
            "workspace_state": _jsonable(asdict(self.workspace)),
            "frontier_state": {
                "global": _jsonable(self.frontier_weights),
                "sessions": _jsonable(self.session_frontiers),
                "hot_trace_ids": list(self.hot_trace_ids),
                "last_trace_by_session": dict(self._last_trace_by_session),
            },
            "experience_frames": [_jsonable(asdict(frame)) for frame in self.frames],
            "predictor_state": _jsonable(asdict(self.predictor.export_state())),
            "objective_state": {
                "ema": float(self.objective_ema),
                "last": dict(self.last_objective),
            },
        }

    @classmethod
    def from_snapshot_payload(cls: type[Any], payload: Mapping[str, Any]) -> "AuroraField":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != 6:
            raise ValueError(f"unsupported snapshot schema_version: {schema_version}")
        config = FieldConfig(**dict(payload.get("config", {})))
        replay_config = ReplayConfig(**dict(payload.get("replay_config", {})))
        budget_config = BudgetConfig(**dict(payload.get("budget_config", {})))
        predictor_config = PredictorConfig(**dict(payload.get("predictor_config", {})))
        field = cast(
            "AuroraField",
            cls(
                config,
                replay_config=replay_config,
                budget_config=budget_config,
                predictor_config=predictor_config,
            ),
        )
        field.step = int(payload.get("step", 0))
        field.current_ts = float(payload.get("current_ts", 0.0))

        for packet_row in payload.get("packet_records", []):
            packet = PacketRecord(
                packet_id=str(packet_row["packet_id"]),
                ts=float(packet_row["ts"]),
                session_id=str(packet_row.get("session_id", "")),
                turn_id=str(packet_row.get("turn_id", "")),
                source=cast(SourceType, packet_row["source"]),
                payload_type=cast(PayloadType, packet_row["payload_type"]),
                payload_ref=str(packet_row["payload_ref"]),
                token_count=int(packet_row.get("token_count", 0)),
                meta=dict(packet_row.get("meta", {})),
            )
            field.anchor_store.add_packet(packet)
        for anchor_row in payload.get("anchors", []):
            field.anchor_store.add_anchor(Anchor(**anchor_row))
        for trace_row in payload.get("traces", []):
            trace = TraceRecord(
                trace_id=str(trace_row["trace_id"]),
                z_mu=np.asarray(trace_row["z_mu"], dtype=np.float64),
                z_sigma_diag=np.asarray(trace_row["z_sigma_diag"], dtype=np.float64),
                ctx_mu=np.asarray(trace_row["ctx_mu"], dtype=np.float64),
                ctx_sigma_diag=np.asarray(trace_row["ctx_sigma_diag"], dtype=np.float64),
                evidence=float(trace_row.get("evidence", 1.0)),
                stability=float(trace_row.get("stability", 0.0)),
                uncertainty=float(trace_row.get("uncertainty", 1.0)),
                fidelity=float(trace_row.get("fidelity", 1.0)),
                compression_stage=int(trace_row.get("compression_stage", 0)),
                pred_loss_ema=float(trace_row.get("pred_loss_ema", 0.0)),
                future_alignment_ema=float(trace_row.get("future_alignment_ema", 0.0)),
                future_drift_ema=float(trace_row.get("future_drift_ema", 0.0)),
                access_ema=float(trace_row.get("access_ema", 0.0)),
                last_access_ts=float(trace_row.get("last_access_ts", 0.0)),
                t_start=float(trace_row.get("t_start", 0.0)),
                t_end=float(trace_row.get("t_end", 0.0)),
                anchor_reservoir=deque(trace_row.get("anchor_reservoir", [])),
                role_logits=dict(trace_row.get("role_logits", {})),
                role_support=dict(trace_row.get("role_support", {})),
                role_gain_ema=dict(trace_row.get("role_gain_ema", {})),
                member_ids=tuple(trace_row.get("member_ids", ())),
                path_signature=tuple(trace_row.get("path_signature", ())),
                posterior_group_ids=tuple(trace_row.get("posterior_group_ids", ())),
                parent_trace_id=trace_row.get("parent_trace_id"),
                metadata=dict(trace_row.get("metadata", {})),
            )
            field.trace_store.add(trace)
            field.ann_index.add_or_update(trace)
        for edge_row in payload.get("edges", []):
            field.edge_store.upsert(
                TraceEdge(
                    src=str(edge_row["src"]),
                    dst=str(edge_row["dst"]),
                    kind=cast(EdgeKind, edge_row["kind"]),
                    weight=float(edge_row.get("weight", 0.0)),
                    support_ema=float(edge_row.get("support_ema", 0.0)),
                    last_update_ts=float(edge_row.get("last_update_ts", 0.0)),
                    bf_sep_ema=float(edge_row.get("bf_sep_ema", 0.0)),
                    success_alpha=float(edge_row.get("success_alpha", 1.0)),
                    success_beta=float(edge_row.get("success_beta", 1.0)),
                )
            )
        for group_row in payload.get("posterior_groups", []):
            field.groups[str(group_row["group_id"])] = PosteriorGroup(
                group_id=str(group_row["group_id"]),
                member_ids=list(group_row.get("member_ids", [])),
                alpha=np.asarray(group_row.get("alpha", []), dtype=np.float64),
                ctx_mu=np.asarray(group_row.get("ctx_mu", []), dtype=np.float64),
                ctx_sigma_diag=np.asarray(group_row.get("ctx_sigma_diag", []), dtype=np.float64),
                pred_success_ema=np.asarray(group_row.get("pred_success_ema", []), dtype=np.float64),
                temperature=float(group_row.get("temperature", 1.0)),
                unresolved_mass=float(group_row.get("unresolved_mass", 0.0)),
                replay_alignment_ema=float(group_row.get("replay_alignment_ema", 0.0)),
                replay_tension_ema=float(group_row.get("replay_tension_ema", 0.0)),
                ambiguous_buffer=deque(group_row.get("ambiguous_buffer", [])),
                ambiguous_ctx_buffer=deque(
                    np.asarray(item, dtype=np.float64) for item in group_row.get("ambiguous_ctx_buffer", [])
                ),
            )
        workspace_row = dict(payload.get("workspace_state", {}))
        field.workspace = Workspace(
            active_trace_ids=tuple(workspace_row.get("active_trace_ids", ())),
            weights=tuple(float(value) for value in workspace_row.get("weights", ())),
            activation={str(key): float(value) for key, value in workspace_row.get("activation", {}).items()},
            posterior_groups=tuple(workspace_row.get("posterior_groups", ())),
            active_procedure_ids=tuple(workspace_row.get("active_procedure_ids", ())),
            anchor_refs=tuple(workspace_row.get("anchor_refs", ())),
            summary_vector=np.asarray(workspace_row.get("summary_vector", np.zeros(field.config.latent_dim)), dtype=np.float64),
            metadata=dict(workspace_row.get("metadata", {})),
        )
        frontier_state = dict(payload.get("frontier_state", {}))
        field.frontier_weights = {str(key): float(value) for key, value in dict(frontier_state.get("global", {})).items()}
        field.session_frontiers = {
            str(session_id): {str(key): float(value) for key, value in dict(weights).items()}
            for session_id, weights in dict(frontier_state.get("sessions", {})).items()
        }
        field.hot_trace_ids = [str(trace_id) for trace_id in frontier_state.get("hot_trace_ids", [])]
        field._last_trace_by_session = {
            str(session_id): str(trace_id)
            for session_id, trace_id in dict(frontier_state.get("last_trace_by_session", {})).items()
        }
        for frame_row in payload.get("experience_frames", []):
            frame = ExperienceFrame(
                frame_id=str(frame_row["frame_id"]),
                anchor_id=str(frame_row["anchor_id"]),
                trace_id=str(frame_row["trace_id"]),
                session_id=str(frame_row["session_id"]),
                turn_id=str(frame_row["turn_id"]),
                action=cast(ActionName, frame_row["action"]),
                x=np.asarray(frame_row["x"], dtype=np.float64),
                context=np.asarray(frame_row["context"], dtype=np.float64),
                workspace_vec=np.asarray(frame_row["workspace_vec"], dtype=np.float64),
                frontier_vec=np.asarray(frame_row["frontier_vec"], dtype=np.float64),
                action_vec=np.asarray(frame_row["action_vec"], dtype=np.float64),
                ts=float(frame_row["ts"]),
                active_ids=tuple(frame_row.get("active_ids", ())),
                activation={str(key): float(value) for key, value in dict(frame_row.get("activation", {})).items()},
                group_entropy=float(frame_row.get("group_entropy", 0.0)),
                next_x=(
                    None
                    if frame_row.get("next_x") is None
                    else np.asarray(frame_row.get("next_x"), dtype=np.float64)
                ),
                next_trace_id=frame_row.get("next_trace_id"),
                next_ts=frame_row.get("next_ts"),
            )
            field.frames.append(frame)
            field._last_frame_by_session[frame.session_id] = frame
        predictor_state = payload.get("predictor_state")
        if predictor_state:
            field.predictor.restore_state(
                self_predictor_state := type(field.predictor.export_state())(
                    h=np.asarray(predictor_state["h"], dtype=np.float64),
                    theta={str(k): np.asarray(v, dtype=np.float64) for k, v in predictor_state["theta"].items()},
                    theta_target={
                        str(k): np.asarray(v, dtype=np.float64) for k, v in predictor_state["theta_target"].items()
                    },
                )
            )
            del self_predictor_state
        objective_state = dict(payload.get("objective_state", {}))
        field.objective_ema = float(objective_state.get("ema", 0.0))
        field.last_objective = {str(key): float(value) for key, value in dict(objective_state.get("last", {})).items()}
        return field

    def restore_from_snapshot_payload(self: Any, payload: Mapping[str, Any]) -> None:
        restored = self.from_snapshot_payload(payload)
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, deque):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field_name: _jsonable(getattr(value, field_name)) for field_name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
