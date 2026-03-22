"""Aurora v2 trace-field runtime."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from aurora.budget import BudgetController
from aurora.core.math import (
    EPS,
    cosine_similarity,
    diag_gaussian_logpdf,
    entropy,
    l2_normalize,
    safe_variance,
    weighted_mean,
)
from aurora.core.types import (
    ActionName,
    Anchor,
    BudgetConfig,
    EdgeKind,
    ExperienceFrame,
    FieldConfig,
    InjectResult,
    MaintenanceStats,
    PosteriorGroup,
    PredictorConfig,
    ProposalDecision,
    SourceType,
    PayloadType,
    ReplayConfig,
    SnapshotMeta,
    TraceEdge,
    TraceRecord,
    Workspace,
)
from aurora.ingest import AnchorStore, HashingEncoder, Packetizer
from aurora.ingest.packetizer import PacketRecord
from aurora.models import SlowPredictor
from aurora.readout.workspace import settle_workspace
from aurora.replay import (
    compute_uncertainty,
    push_anchor_reference,
    reconsolidate_trace,
    trace_forget_risk,
    update_stability,
)
from aurora.runtime.objective import (
    ActionEvaluation,
    ObjectiveObservation,
    ObjectiveTerms,
    empirical_patch_terms,
    edge_storage_cost,
    group_storage_cost,
    group_tension_cost,
    objective_from_components,
    posterior_slice,
    safe_kl,
    trace_log_prior,
    trace_fidelity_cost,
    trace_role_cost,
    trace_storage_cost,
    trace_structural_role_mass,
)
from aurora.runtime.proposals import ACTION_ORDER, action_vector
from aurora.store import BlobStore, EdgeStore, ExactANNIndex, TraceStore


class AuroraField:
    """Unified trace-field runtime with local-energy proposal dynamics."""

    def __init__(
        self,
        config: FieldConfig | None = None,
        *,
        replay_config: ReplayConfig | None = None,
        budget_config: BudgetConfig | None = None,
        predictor_config: PredictorConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or FieldConfig()
        self.replay_config = replay_config or ReplayConfig()
        self.budget_config = budget_config or BudgetConfig(
            max_traces=self.config.trace_budget,
            max_edges=self.config.edge_budget,
        )
        if predictor_config is None:
            predictor_config = PredictorConfig(
                latent_dim=self.config.latent_dim,
                action_dim=len(ACTION_ORDER),
            )
        self.predictor_config = predictor_config

        self.blob_store = BlobStore(self.config.blob_dir)
        self.packetizer = Packetizer(self.config, self.blob_store)
        self.encoder = HashingEncoder(self.config, self.blob_store)
        self.anchor_store = AnchorStore()
        self.trace_store = TraceStore()
        self.edge_store = EdgeStore()
        self.ann_index = ExactANNIndex()
        self.predictor = SlowPredictor(
            self.predictor_config,
            lr=self.replay_config.predictor_lr,
            weight_decay=self.replay_config.predictor_weight_decay,
        )
        self.budget_controller = BudgetController(self.budget_config)

        self.anchors = self.anchor_store.anchors
        self.traces = self.trace_store.traces
        self.edges = self.edge_store.edges
        self.posterior_groups: dict[str, PosteriorGroup] = {}
        self.groups = self.posterior_groups
        self.workspace = self._empty_workspace()
        self.frontier_weights: dict[str, float] = {}
        self.session_frontiers: dict[str, dict[str, float]] = {}
        self.hot_trace_ids: list[str] = []
        self.frames: list[ExperienceFrame] = []
        self._last_frame_by_session: dict[str, ExperienceFrame] = {}
        self._last_trace_by_session: dict[str, str] = {}
        self.step = 0
        self.current_ts = 0.0
        self.objective_ema = 0.0
        self.last_objective: dict[str, float] = {}
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def inject(self, raw_event: Mapping[str, Any] | str) -> InjectResult:
        packets = self.packetizer.split(raw_event)
        packet_ids: list[str] = []
        anchor_ids: list[str] = []
        trace_ids: list[str] = []
        proposal_kinds: list[str] = []
        touched_trace_ids: list[str] = []

        for packet in packets:
            self.step += 1
            self.anchor_store.add_packet(packet)
            packet_ids.append(packet.packet_id)
            anchor = self.encoder.to_anchor(packet)
            self.anchor_store.add_anchor(anchor)
            anchor_ids.append(anchor.anchor_id)
            self.current_ts = float(anchor.ts)

            zero_action = np.zeros(len(ACTION_ORDER), dtype=np.float64)
            pred = self.predictor.peek(
                self.workspace.summary_vector,
                self.frontier_summary(anchor.session_id),
                zero_action,
                delta_t=1.0,
            )
            decision = self._score_primary_actions(anchor, pred)
            trace_id = self._apply_primary_action(decision, anchor)
            proposal_kinds.append(decision.action)

            trace = self.traces.get(trace_id)
            if trace is not None:
                trace.access_ema = self.config.access_decay * trace.access_ema + 1.0
                trace.last_access_ts = float(anchor.ts)
                if trace.fidelity < 1.0 and trace.access_ema >= self.config.fidelity_restore_activation:
                    self._rehydrate_trace(trace)
                self.ann_index.add_or_update(trace)
            group_entropy = 0.0
            if decision.apply_inhibit and decision.inhibit_pair is not None:
                group_entropy = self._apply_inhibit_pair(
                    decision.inhibit_pair,
                    np.asarray(anchor.z, dtype=np.float64),
                    np.asarray(decision.context, dtype=np.float64),
                    anchor_id=anchor.anchor_id,
                    ts=anchor.ts,
                    pred_loss=decision.objective_terms.get("slow_alignment", 0.0),
                )
            if trace is not None:
                for group_id in trace.posterior_group_ids:
                    group = self.groups.get(group_id)
                    if group is not None:
                        group_entropy = max(
                            group_entropy,
                            self._update_group(
                                group,
                                anchor.z,
                                decision.context,
                                anchor_id=anchor.anchor_id,
                                ts=anchor.ts,
                                pred_loss=decision.objective_terms.get("slow_alignment", 0.0),
                            ),
                        )

            self.workspace = self._settle_workspace(anchor.z, decision.context, pred.mu, anchor.session_id)
            self._update_frontier(self.workspace, anchor.session_id)
            self._update_assoc_edges(self.workspace, ts=anchor.ts)
            self._record_frame(anchor, trace_id, decision, self.workspace, group_entropy)
            self.last_objective = dict(decision.objective_terms)
            self.objective_ema = 0.95 * self.objective_ema + 0.05 * float(decision.objective_terms.get("total", 0.0))

            trace_ids.append(trace_id)
            touched_trace_ids.append(trace_id)

        self._refresh_hot_traces()
        return InjectResult(
            packet_ids=packet_ids,
            anchor_ids=anchor_ids,
            trace_ids=trace_ids,
            proposal_kinds=proposal_kinds,
            touched_trace_ids=list(dict.fromkeys(touched_trace_ids)),
        )

    def maintenance_cycle(self, ms_budget: int | None = None) -> MaintenanceStats:
        started = time.time()
        batch = self._sample_replay_batch(ms_budget)
        if not batch:
            compressed_trace_ids, rehydrated_trace_ids = self._fidelity_step()
            field_terms = self._replay_structural_objective(
                (),
                predictor_loss=0.0,
                replayed_group_ids=(),
                structural_trace_ids=(),
                compressed_trace_ids=compressed_trace_ids,
                demoted_trace_ids=(),
                pruned_trace_ids=(),
            )
            self.last_objective = field_terms.as_dict()
            self.objective_ema = 0.95 * self.objective_ema + 0.05 * field_terms.total
            return MaintenanceStats(
                elapsed_ms=int((time.time() - started) * 1000),
                compressed_trace_ids=compressed_trace_ids,
                rehydrated_trace_ids=rehydrated_trace_ids,
                objective_total=float(field_terms.total),
            )
        predictor_metrics = self.predictor.fit_batch(
            batch,
            train_steps=self.replay_config.train_steps,
            target_ema=self.replay_config.theta_target_ema,
            drift_penalty=self.replay_config.predictor_weight_decay,
        )
        replayed_trace_ids = self._reconsolidate_batch(batch)
        replayed_group_ids: list[str] = []
        structural_trace_ids: list[str] = []
        for _ in range(max(int(self.config.maintenance_structural_passes), 1)):
            replayed_group_ids.extend(self._replay_groups_from_batch(batch))
            new_trace_ids = self._replay_trace_mutations(batch)
            structural_trace_ids.extend(new_trace_ids)
            if new_trace_ids:
                replayed_group_ids.extend(self._replay_groups_from_batch(batch))
        replayed_group_ids = list(dict.fromkeys(replayed_group_ids))
        structural_trace_ids = list(dict.fromkeys(structural_trace_ids))
        self._update_edges_from_batch(batch)
        self._update_continuation_stats(batch)
        prototype_trace_ids = self._maybe_promote_prototype(batch)
        procedure_trace_ids = self._maybe_promote_procedure(batch)
        compressed_trace_ids, rehydrated_trace_ids = self._fidelity_step()
        demoted_trace_ids = self._role_lifecycle_step()
        pruned_trace_ids = self._budget_step()
        self._refresh_hot_traces()
        elapsed_ms = int((time.time() - started) * 1000)
        field_terms = self._replay_structural_objective(
            batch,
            predictor_loss=float(predictor_metrics.get("nll", 0.0)),
            replayed_group_ids=replayed_group_ids,
            structural_trace_ids=structural_trace_ids,
            compressed_trace_ids=compressed_trace_ids,
            demoted_trace_ids=demoted_trace_ids,
            pruned_trace_ids=pruned_trace_ids,
        )
        self.last_objective = field_terms.as_dict()
        self.objective_ema = 0.95 * self.objective_ema + 0.05 * field_terms.total
        return MaintenanceStats(
            replayed_trace_ids=replayed_trace_ids,
            structural_trace_ids=structural_trace_ids,
            prototype_trace_ids=prototype_trace_ids,
            procedure_trace_ids=procedure_trace_ids,
            pruned_trace_ids=pruned_trace_ids,
            compressed_trace_ids=compressed_trace_ids,
            rehydrated_trace_ids=rehydrated_trace_ids,
            demoted_trace_ids=demoted_trace_ids,
            replayed_group_ids=replayed_group_ids,
            elapsed_ms=elapsed_ms,
            replay_batch=len(batch),
            predictor_loss=float(predictor_metrics.get("loss", 0.0)),
            predictor_nll=float(predictor_metrics.get("nll", 0.0)),
            objective_total=float(field_terms.total),
        )

    def read_workspace(self, cue: str | Mapping[str, Any], k: int | None = None) -> Workspace:
        if isinstance(cue, str):
            cue_text = cue
            session_id = ""
        else:
            cue_text = str(cue.get("payload") or cue.get("text") or "")
            session_id = str(cue.get("session_id") or "")
        cue_vec = self.encoder.encode_query(cue_text).astype(np.float64)
        zero_action = np.zeros(len(ACTION_ORDER), dtype=np.float64)
        pred = self.predictor.peek(
            self.workspace.summary_vector,
            self.frontier_summary(session_id),
            zero_action,
            delta_t=1.0,
        )
        context = self._make_context(cue_vec, self.frontier_summary(session_id), pred.mu)
        workspace = self._settle_workspace(cue_vec, context, pred.mu, session_id, workspace_size=k)
        self.workspace = workspace
        for trace_id in workspace.active_trace_ids:
            trace = self.traces.get(trace_id)
            if trace is None:
                continue
            trace.access_ema = self.config.access_decay * trace.access_ema + 1.0
            trace.last_access_ts = max(trace.last_access_ts, self.current_ts)
        self._update_frontier(workspace, session_id)
        self._refresh_hot_traces()
        return workspace

    def snapshot(self) -> SnapshotMeta:
        target = Path(self.config.data_dir) / "snapshots" / f"field-{self.step:08d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_snapshot_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        return SnapshotMeta(
            snapshot_path=str(target),
            trace_count=len(self.traces),
            anchor_count=len(self.anchors),
            edge_count=len(self.edges),
        )

    def field_stats(self) -> dict[str, Any]:
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

    def to_snapshot_payload(self) -> dict[str, Any]:
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
    def from_snapshot_payload(cls, payload: Mapping[str, Any]) -> "AuroraField":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != 6:
            raise ValueError(f"unsupported snapshot schema_version: {schema_version}")
        config = FieldConfig(**dict(payload.get("config", {})))
        replay_config = ReplayConfig(**dict(payload.get("replay_config", {})))
        budget_config = BudgetConfig(**dict(payload.get("budget_config", {})))
        predictor_config = PredictorConfig(**dict(payload.get("predictor_config", {})))
        field = cls(
            config,
            replay_config=replay_config,
            budget_config=budget_config,
            predictor_config=predictor_config,
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

    def restore_from_snapshot_payload(self, payload: Mapping[str, Any]) -> None:
        restored = self.from_snapshot_payload(payload)
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)

    # ------------------------------------------------------------------
    # field helpers
    # ------------------------------------------------------------------
    def _empty_workspace(self) -> Workspace:
        return Workspace(
            active_trace_ids=tuple(),
            weights=tuple(),
            activation={},
            posterior_groups=tuple(),
            active_procedure_ids=tuple(),
            anchor_refs=tuple(),
            summary_vector=np.zeros(self.config.latent_dim, dtype=np.float64),
            metadata={"iterations": 0},
        )

    def frontier_summary(self, session_id: str = "") -> np.ndarray:
        frontier_weights = self.session_frontiers.get(session_id, self.frontier_weights) if session_id else self.frontier_weights
        if not frontier_weights:
            return np.zeros(self.config.latent_dim, dtype=np.float64)
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        for trace_id, weight in list(frontier_weights.items()):
            trace = self.traces.get(trace_id)
            if trace is None:
                continue
            age = max(self.current_ts - trace.last_access_ts, 0.0)
            unresolved_bonus = 1.0 + trace.uncertainty
            decayed_weight = weight * math.exp(-age / 10.0) * unresolved_bonus
            vectors.append(trace.z_mu)
            weights.append(decayed_weight)
        if not vectors:
            return np.zeros(self.config.latent_dim, dtype=np.float64)
        return weighted_mean(vectors, weights)

    def _record_frame(
        self,
        anchor: Anchor,
        trace_id: str,
        decision: ProposalDecision,
        workspace: Workspace,
        group_entropy: float,
    ) -> None:
        previous = self._last_frame_by_session.get(anchor.session_id)
        if previous is not None:
            previous.next_x = np.asarray(anchor.z, dtype=np.float64).copy()
            previous.next_trace_id = trace_id
            previous.next_ts = float(anchor.ts)
            if previous.trace_id in self.traces and trace_id in self.traces:
                self._upsert_edge(previous.trace_id, trace_id, "temporal", delta_weight=0.10, ts=anchor.ts)
        frontier_vec = self.frontier_summary(anchor.session_id)
        frame = ExperienceFrame(
            frame_id=f"frame_{uuid.uuid4().hex[:12]}",
            anchor_id=anchor.anchor_id,
            trace_id=trace_id,
            session_id=anchor.session_id,
            turn_id=anchor.turn_id,
            action=decision.action,
            x=np.asarray(anchor.z, dtype=np.float64).copy(),
            context=np.asarray(decision.context, dtype=np.float64).copy(),
            workspace_vec=np.asarray(workspace.summary_vector, dtype=np.float64).copy(),
            frontier_vec=np.asarray(frontier_vec, dtype=np.float64).copy(),
            action_vec=action_vector(decision.action),
            ts=float(anchor.ts),
            active_ids=tuple(workspace.active_trace_ids),
            activation=dict(workspace.activation),
            group_entropy=float(group_entropy),
        )
        self.frames.append(frame)
        self._last_frame_by_session[anchor.session_id] = frame
        self._last_trace_by_session[anchor.session_id] = trace_id
        self.predictor.step(frame.workspace_vec, frame.frontier_vec, frame.action_vec, delta_t=1.0)

    def _refresh_hot_traces(self) -> None:
        ranked = sorted(
            self.traces.values(),
            key=lambda trace: (
                trace.evidence
                + trace.access_ema
                + trace.stability
                - trace.uncertainty
                + trace_structural_role_mass(trace)
            ),
            reverse=True,
        )
        self.hot_trace_ids = [trace.trace_id for trace in ranked[: self.config.frontier_size]]

    def _trace_matches_session(self, trace: TraceRecord, session_id: str) -> bool:
        if not session_id:
            return True
        return trace.metadata.get("session_id") == session_id

    def _make_context(self, q: np.ndarray, frontier: np.ndarray, pred_mu: np.ndarray) -> np.ndarray:
        cue_term = self.config.cue_context_weight * l2_normalize(np.asarray(q, dtype=np.float64))
        frontier_term = self.config.frontier_context_weight * l2_normalize(np.asarray(frontier, dtype=np.float64))
        pred_term = self.config.predictor_context_weight * l2_normalize(np.asarray(pred_mu, dtype=np.float64))
        context = l2_normalize(cue_term + frontier_term + pred_term)
        if context.shape[0] != self.config.context_dim:
            if context.shape[0] > self.config.context_dim:
                context = context[: self.config.context_dim]
            else:
                context = np.pad(context, (0, self.config.context_dim - context.shape[0]))
        return context.astype(np.float64)

    def _neighbors(self, trace_id: str) -> set[str]:
        out: set[str] = set()
        for edge in self.edges.values():
            if edge.src == trace_id:
                out.add(edge.dst)
            if edge.dst == trace_id:
                out.add(edge.src)
        return out

    def _make_candidates(self, q: np.ndarray, c: np.ndarray, session_id: str = "") -> tuple[str, ...]:
        if not self.traces:
            return tuple()
        candidates: set[str] = set(self.workspace.active_trace_ids)
        candidates.update(self.frontier_weights.keys())
        if session_id:
            candidates.update(self.session_frontiers.get(session_id, {}).keys())
        for trace_id in list(candidates):
            candidates.update(self._neighbors(trace_id))

        for trace_id in self.ann_index.search(q.tolist(), top_k=self.config.candidate_size):
            trace = self.traces.get(trace_id)
            if trace is None or not self._trace_matches_session(trace, session_id):
                continue
            candidates.add(trace_id)

        ranked: list[tuple[float, str]] = []
        for trace_id, trace in self.traces.items():
            if not self._trace_matches_session(trace, session_id):
                continue
            latent_score = cosine_similarity(q, trace.z_mu)
            ctx_score = cosine_similarity(c, trace.ctx_mu)
            utility = 0.1 * trace.evidence + 0.1 * trace.access_ema - 0.05 * trace.uncertainty
            session_bonus = 0.25 if session_id and trace.metadata.get("session_id") == session_id else 0.0
            score = 0.65 * latent_score + 0.25 * ctx_score + utility + session_bonus
            ranked.append((score, trace_id))
            if trace_structural_role_mass(trace) >= self.config.active_role_threshold:
                candidates.add(trace_id)
        ranked.sort(reverse=True)
        for _, trace_id in ranked[: self.config.candidate_size]:
            candidates.add(trace_id)

        expanded = set(candidates)
        for trace_id in list(candidates):
            trace = self.traces.get(trace_id)
            if trace is None:
                continue
            for group_id in trace.posterior_group_ids:
                group = self.groups.get(group_id)
                if group is not None:
                    expanded.update(member_id for member_id in group.member_ids if member_id in self.traces)

        rescored: list[tuple[float, str]] = []
        for trace_id in expanded:
            trace = self.traces.get(trace_id)
            if trace is None or not self._trace_matches_session(trace, session_id):
                continue
            session_bonus = 0.25 if session_id and trace.metadata.get("session_id") == session_id else 0.0
            score = (
                0.60 * cosine_similarity(q, trace.z_mu)
                + 0.20 * cosine_similarity(c, trace.ctx_mu)
                + 0.15 * trace.evidence
                - 0.10 * trace.uncertainty
                + 0.05 * trace_structural_role_mass(trace)
                + session_bonus
            )
            rescored.append((score, trace_id))
        rescored.sort(reverse=True)
        return tuple(trace_id for _, trace_id in rescored[: self.config.candidate_size])

    def _degree(self, trace_id: str) -> int:
        return sum(1 for edge in self.edges.values() if edge.src == trace_id or edge.dst == trace_id)

    def _new_trace(
        self,
        anchor: Anchor,
        c: np.ndarray,
        *,
        trace_id: str | None = None,
        parent_trace_id: str | None = None,
        sigma: np.ndarray | None = None,
        ctx_sigma: np.ndarray | None = None,
    ) -> TraceRecord:
        trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        sigma_diag = (
            np.full_like(anchor.z, self.config.init_sigma)
            if sigma is None
            else safe_variance(sigma, min_value=self.config.sigma_floor, max_value=self.config.sigma_ceiling)
        )
        ctx_sigma_diag = (
            np.full_like(c, self.config.init_ctx_sigma)
            if ctx_sigma is None
            else safe_variance(ctx_sigma, min_value=self.config.sigma_floor, max_value=self.config.sigma_ceiling)
        )
        trace = TraceRecord(
            trace_id=trace_id,
            z_mu=np.asarray(anchor.z, dtype=np.float64).copy(),
            z_sigma_diag=sigma_diag,
            ctx_mu=np.asarray(c, dtype=np.float64).copy(),
            ctx_sigma_diag=ctx_sigma_diag,
            evidence=1.0,
            stability=0.0,
            uncertainty=1.0,
            fidelity=1.0,
            pred_loss_ema=0.0,
            access_ema=1.0,
            last_access_ts=float(anchor.ts),
            t_start=float(anchor.ts),
            t_end=float(anchor.ts),
            anchor_reservoir=deque(),
            parent_trace_id=parent_trace_id,
            metadata={
                "session_id": anchor.session_id,
                "turn_id": anchor.turn_id,
                "source": anchor.source,
                **dict(anchor.meta),
            },
        )
        push_anchor_reference(trace, anchor.anchor_id, reservoir_size=self.config.reservoir_size)
        trace.uncertainty = compute_uncertainty(trace, self.config)
        return trace

    def _collect_local_frames(
        self,
        trace_ids: Sequence[str],
        *,
        session_id: str = "",
        limit: int | None = None,
    ) -> list[ExperienceFrame]:
        wanted = {trace_id for trace_id in trace_ids if trace_id in self.traces}
        if not wanted:
            return []
        budget = int(limit or self.config.objective_local_window)
        frames: list[ExperienceFrame] = []
        for frame in reversed(self.frames):
            if session_id and frame.session_id and frame.session_id != session_id:
                continue
            if frame.trace_id in wanted or (frame.next_trace_id is not None and frame.next_trace_id in wanted):
                frames.append(frame)
            elif any(trace_id in wanted for trace_id in frame.active_ids):
                frames.append(frame)
            if len(frames) >= budget:
                break
        frames.reverse()
        return frames

    def _objective_observations(
        self,
        x: np.ndarray,
        c: np.ndarray,
        pred: Any,
        *,
        trace_ids: Sequence[str],
        session_id: str = "",
        extra_frames: Sequence[ExperienceFrame] | None = None,
    ) -> list[ObjectiveObservation]:
        observations: list[ObjectiveObservation] = [
            ObjectiveObservation(
                x=np.asarray(x, dtype=np.float64),
                c=np.asarray(c, dtype=np.float64),
                target=np.asarray(pred.mu, dtype=np.float64),
                target_sigma_diag=np.asarray(pred.sigma_diag, dtype=np.float64),
            )
        ]
        seen_frame_ids: set[str] = set()
        frames = list(extra_frames) if extra_frames is not None else self._collect_local_frames(trace_ids, session_id=session_id)
        for frame in frames:
            if frame.frame_id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame.frame_id)
            target = None if frame.next_x is None else np.asarray(frame.next_x, dtype=np.float64)
            observations.append(
                ObjectiveObservation(
                    x=np.asarray(frame.x, dtype=np.float64),
                    c=np.asarray(frame.context, dtype=np.float64),
                    target=target,
                )
            )
        return observations

    def _replay_workspace_for_frame(self, frame: ExperienceFrame) -> Workspace:
        candidates = self._make_candidates(frame.x, frame.context, frame.session_id)
        size = min(
            max(len(frame.active_ids), 1),
            self.config.workspace_size,
        )
        return settle_workspace(
            candidate_ids=candidates,
            traces=self.traces,
            edges=self.edges,
            groups=self.groups,
            cue=np.asarray(frame.x, dtype=np.float64),
            context=np.asarray(frame.context, dtype=np.float64),
            frontier=np.asarray(frame.frontier_vec, dtype=np.float64),
            pred_mu=np.asarray(frame.workspace_vec, dtype=np.float64),
            config=self.config,
            posterior_slice_fn=self._posterior_slice,
            session_id=frame.session_id,
            workspace_size=size,
        )

    def _transition_support(self, src: str, dst: str) -> float:
        support = 0.0
        temporal = self.edges.get((src, dst, "temporal"))
        if temporal is not None:
            support = max(support, float(np.clip(temporal.weight, 0.0, 1.0)))
        option = self.edges.get((src, dst, "option"))
        if option is not None:
            total = max(option.success_alpha + option.success_beta, EPS)
            support = max(support, float(option.success_alpha / total))
        return float(np.clip(support, 0.0, 1.0))

    def _update_continuation_stats(self, batch: Sequence[ExperienceFrame]) -> None:
        if not batch:
            return
        decay = self.config.continuation_decay
        align_acc: defaultdict[str, list[float]] = defaultdict(list)
        drift_acc: defaultdict[str, list[float]] = defaultdict(list)
        group_align_acc: defaultdict[str, list[float]] = defaultdict(list)
        for frame in batch:
            nll = self.predictor.score_transition(frame) if frame.next_x is not None else 0.0
            align_acc[frame.trace_id].append(float(max(nll, 0.0)))
            drift_value = float(max(frame.group_entropy, 0.0))
            group_ids: set[str] = set()
            trace = self.traces.get(frame.trace_id)
            if trace is not None:
                group_ids.update(trace.posterior_group_ids)
            if frame.next_trace_id is not None and frame.trace_id in self.traces and frame.next_trace_id in self.traces:
                drift_value += 1.0 - self._transition_support(frame.trace_id, frame.next_trace_id)
                align_acc[frame.next_trace_id].append(0.5 * float(max(nll, 0.0)))
                next_trace = self.traces.get(frame.next_trace_id)
                if next_trace is not None:
                    group_ids.update(next_trace.posterior_group_ids)
            drift_acc[frame.trace_id].append(drift_value)
            for group_id in group_ids:
                group_align_acc[group_id].append(float(max(nll, 0.0)))
        for trace_id, values in align_acc.items():
            trace = self.traces.get(trace_id)
            if trace is None:
                continue
            trace.future_alignment_ema = decay * trace.future_alignment_ema + (1.0 - decay) * float(np.mean(values))
        for trace_id, values in drift_acc.items():
            trace = self.traces.get(trace_id)
            if trace is None:
                continue
            trace.future_drift_ema = decay * trace.future_drift_ema + (1.0 - decay) * float(np.mean(values))
        group_decay = self.config.continuation_group_decay
        for group_id, values in group_align_acc.items():
            group = self.groups.get(group_id)
            if group is None:
                continue
            group.replay_alignment_ema = group_decay * group.replay_alignment_ema + (1.0 - group_decay) * float(
                np.mean(values)
            )
        for group in self.groups.values():
            group.replay_tension_ema = group_decay * group.replay_tension_ema + (1.0 - group_decay) * float(
                group.unresolved_mass + abs(group.temperature - 1.0)
            )

    def _replay_structural_objective(
        self,
        batch: Sequence[ExperienceFrame],
        *,
        predictor_loss: float,
        replayed_group_ids: Sequence[str],
        structural_trace_ids: Sequence[str],
        compressed_trace_ids: Sequence[str],
        demoted_trace_ids: Sequence[str],
        pruned_trace_ids: Sequence[str],
    ) -> ObjectiveTerms:
        touched_trace_ids = tuple(
            dict.fromkeys(
                trace_id
                for frame in batch
                for trace_id in (frame.trace_id, frame.next_trace_id, *frame.active_ids)
                if trace_id is not None and trace_id in self.traces
            )
        )
        touched_traces = [self.traces[trace_id] for trace_id in touched_trace_ids]
        trace_mass, edge_mass, group_mass = self._budget_masses()

        sample_count = min(len(batch), max(1, int(self.config.objective_replay_window)))
        sampled_frames = list(batch[:sample_count])
        activation_kls: list[float] = []
        workspace_group_kls: list[float] = []
        transition_gaps: list[float] = []
        for frame in sampled_frames:
            workspace = self._replay_workspace_for_frame(frame)
            comparison_ids = tuple(dict.fromkeys((*frame.active_ids, *workspace.active_trace_ids)))
            if comparison_ids:
                past = np.asarray([frame.activation.get(trace_id, 0.0) for trace_id in comparison_ids], dtype=np.float64)
                current = np.asarray(
                    [workspace.activation.get(trace_id, 0.0) for trace_id in comparison_ids],
                    dtype=np.float64,
                )
                if float(np.sum(past)) > EPS and float(np.sum(current)) > EPS:
                    activation_kls.append(0.5 * (safe_kl(past, current) + safe_kl(current, past)))
            workspace_group_kls.append(float(workspace.metadata.get("group_kl", 0.0)))
            if frame.next_trace_id is not None and frame.trace_id in self.traces and frame.next_trace_id in self.traces:
                transition_gaps.append(1.0 - self._transition_support(frame.trace_id, frame.next_trace_id))

        replay_groups = [self.groups[group_id] for group_id in replayed_group_ids if group_id in self.groups]
        group_heat = float(np.mean([group_tension_cost(group) for group in replay_groups])) if replay_groups else 0.0
        structural_churn = float(
            len(set(structural_trace_ids) | set(compressed_trace_ids) | set(demoted_trace_ids) | set(pruned_trace_ids))
            / max(len(touched_trace_ids), 1)
        )
        activation_kl = float(np.mean(activation_kls)) if activation_kls else 0.0
        transition_gap = float(np.mean(transition_gaps)) if transition_gaps else 0.0
        workspace_group_kl = float(np.mean(workspace_group_kls)) if workspace_group_kls else 0.0
        observations = [
            ObjectiveObservation(
                x=np.asarray(frame.x, dtype=np.float64),
                c=np.asarray(frame.context, dtype=np.float64),
                target=(None if frame.next_x is None else np.asarray(frame.next_x, dtype=np.float64)),
            )
            for frame in sampled_frames
        ]
        maintenance_ids = tuple(
            dict.fromkeys(
                trace_id
                for trace_id in (*touched_trace_ids, *structural_trace_ids)
                if trace_id in self.traces
            )
        )
        plasticity = float(np.mean([max(trace.pred_loss_ema, 0.0) for trace in touched_traces])) if touched_traces else 0.0
        terms, _ = empirical_patch_terms(
            config=self.config,
            candidate_ids=maintenance_ids,
            traces=self.traces,
            edges=self.edges,
            groups=self.groups,
            observations=observations,
            plasticity=plasticity,
            extra_drift=activation_kl + transition_gap + structural_churn + float(self.workspace.metadata.get("group_kl", 0.0)),
            extra_group_kl=workspace_group_kl + group_heat,
        )
        terms.extras.update(
            {
                "trace_count": float(len(self.traces)),
                "edge_count": float(len(self.edges)),
                "group_count": float(len(self.groups)),
                "trace_mass": float(trace_mass),
                "edge_mass": float(edge_mass),
                "group_mass": float(group_mass),
                "structural_trace_count": float(len(set(structural_trace_ids))),
                "replay_activation_kl": float(activation_kl),
                "replay_transition_gap": float(transition_gap),
                "maintenance_future_alignment": float(terms.extras.get("future_alignment", 0.0)),
                "maintenance_future_drift": float(terms.extras.get("future_drift", 0.0)),
                "workspace_group_kl": float(workspace_group_kl),
                "group_heat": float(group_heat),
                "structural_churn": float(structural_churn),
                "sampled_replay_frames": float(len(sampled_frames)),
            }
        )
        return terms

    def _objective_terms(
        self,
        candidate_ids: Sequence[str],
        x: np.ndarray,
        c: np.ndarray,
        pred: Any,
        *,
        shadow: Mapping[str, TraceRecord] | None = None,
        extra_trace_cost: float = 0.0,
        extra_edge_cost: float = 0.0,
        plasticity: float = 0.0,
        inhibit_bonus: float = 0.0,
        observations: Sequence[ObjectiveObservation] | None = None,
    ) -> tuple[ObjectiveTerms, dict[str, float]]:
        shadow = shadow or {}
        observations = list(
            observations
            or [
                ObjectiveObservation(
                    x=np.asarray(x, dtype=np.float64),
                    c=np.asarray(c, dtype=np.float64),
                    target=np.asarray(pred.mu, dtype=np.float64),
                    target_sigma_diag=np.asarray(pred.sigma_diag, dtype=np.float64),
                )
            ]
        )
        drift = float(plasticity)
        workspace_drift = 0.0
        if self.workspace.summary_vector.size > 0:
            workspace_drift = float(np.mean(np.square(self.workspace.summary_vector - np.asarray(pred.mu, dtype=np.float64))))
            drift += workspace_drift
        terms, responsibilities = empirical_patch_terms(
            config=self.config,
            candidate_ids=candidate_ids,
            traces=self.traces,
            edges=self.edges,
            groups=self.groups,
            observations=observations,
            shadow=shadow,
            extra_trace_cost=extra_trace_cost,
            extra_edge_cost=extra_edge_cost,
            plasticity=plasticity,
            inhibit_bonus=inhibit_bonus,
            extra_drift=drift,
        )
        if workspace_drift > 0.0:
            terms.extras["workspace_drift"] = float(workspace_drift)
        return terms, responsibilities

    def _ambiguity_pair(self, responsibilities: Mapping[str, float]) -> tuple[str, str] | None:
        if len(responsibilities) < 2:
            return None
        ranked = sorted(responsibilities.items(), key=lambda item: item[1], reverse=True)
        (left_id, left_prob), (right_id, right_prob) = ranked[:2]
        if left_prob - right_prob > self.config.inhibit_margin:
            return None
        return (left_id, right_id)

    def _inhibit_delta(self, pair: tuple[str, str] | None, x: np.ndarray, c: np.ndarray) -> float:
        if pair is None:
            return 0.0
        left_id, right_id = pair
        if left_id not in self.traces or right_id not in self.traces:
            return 0.0
        bf = self._bf_separate(left_id, right_id, x, c)
        return float(
            -self.config.lambda_inhibit_gain * max(0.0, bf)
            + self.config.lambda_storage * self.config.base_edge_cost
            + self.config.lambda_complexity * 0.1
        )

    def _score_action_candidate(
        self,
        action: ActionName,
        trace_id: str | None,
        anchor: Anchor,
        c: np.ndarray,
        candidate_ids: Sequence[str],
        pred: Any,
        observations: Sequence[ObjectiveObservation],
    ) -> ActionEvaluation:
        x = np.asarray(anchor.z, dtype=np.float64)
        shadow: dict[str, TraceRecord] = {}
        extra_trace_cost = 0.0
        extra_edge_cost = 0.0
        plasticity = 0.0
        target_id = trace_id
        if action == "BIRTH":
            child = self._new_trace(anchor, c, trace_id="__birth__")
            shadow[child.trace_id] = child
            extra_trace_cost += self.config.base_trace_cost
        elif trace_id is None:
            raise RuntimeError("non-birth action requires a trace id")
        elif action == "ASSIMILATE":
            trace = self.traces[trace_id].clone()
            old_mu = trace.z_mu.copy()
            eta = self.config.eta_mu / (1.0 + self.config.lambda_stability * trace.stability)
            trace.z_mu = trace.z_mu + eta * (x - trace.z_mu)
            trace.z_sigma_diag = safe_variance(
                (1.0 - eta) * trace.z_sigma_diag + eta * np.square(x - old_mu),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            trace.ctx_mu = trace.ctx_mu + self.config.eta_ctx * (c - trace.ctx_mu)
            trace.ctx_sigma_diag = safe_variance(
                (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag + self.config.eta_ctx * np.square(c - trace.ctx_mu),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            trace.evidence = self.config.evidence_decay * trace.evidence + 1.0
            trace.uncertainty = compute_uncertainty(trace, self.config)
            shadow[trace_id] = trace
            plasticity = trace.stability * float(np.mean(np.square(trace.z_mu - old_mu)))
        elif action == "ATTACH":
            trace = self.traces[trace_id].clone()
            trace.ctx_mu = trace.ctx_mu + self.config.eta_ctx * (c - trace.ctx_mu)
            trace.ctx_sigma_diag = safe_variance(
                (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag + self.config.eta_ctx * np.square(c - trace.ctx_mu),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            trace.evidence = self.config.evidence_decay * trace.evidence + 1.0
            trace.uncertainty = compute_uncertainty(trace, self.config)
            shadow[trace_id] = trace
        elif action == "SPLIT":
            parent = self.traces[trace_id]
            sigma = np.minimum(np.full_like(parent.z_sigma_diag, self.config.init_sigma), 0.5 * parent.z_sigma_diag)
            child = self._new_trace(
                anchor,
                parent.ctx_mu.copy(),
                parent_trace_id=parent.trace_id,
                sigma=sigma,
                ctx_sigma=parent.ctx_sigma_diag.copy(),
                trace_id="__split__",
            )
            child.fidelity = max(parent.fidelity, 0.85)
            shadow[child.trace_id] = child
            extra_trace_cost += self.config.base_trace_cost
            extra_edge_cost += self.config.base_edge_cost
        else:
            raise ValueError(f"unsupported action: {action}")

        terms, responsibilities = self._objective_terms(
            candidate_ids,
            x,
            c,
            pred,
            shadow=shadow,
            extra_trace_cost=extra_trace_cost,
            extra_edge_cost=extra_edge_cost,
            plasticity=plasticity,
            observations=observations,
        )
        base_eval = ActionEvaluation(action=action, trace_id=target_id, total=float(terms.total), terms=terms)
        pair = self._ambiguity_pair(responsibilities)
        inhibit_delta = self._inhibit_delta(pair, x, c)
        if pair is None or inhibit_delta >= 0.0:
            return base_eval
        terms_inhibit, _ = self._objective_terms(
            candidate_ids,
            x,
            c,
            pred,
            shadow=shadow,
            extra_trace_cost=extra_trace_cost,
            extra_edge_cost=extra_edge_cost,
            plasticity=plasticity,
            inhibit_bonus=inhibit_delta,
            observations=observations,
        )
        if terms_inhibit.total < base_eval.total:
            return ActionEvaluation(
                action=action,
                trace_id=target_id,
                total=float(terms_inhibit.total),
                terms=terms_inhibit,
                apply_inhibit=True,
                inhibit_pair=pair,
            )
        return base_eval

    def _score_primary_actions(self, anchor: Anchor, pred: Any) -> ProposalDecision:
        return self._score_primary_actions_with_state(
            anchor,
            pred,
            frontier=self.frontier_summary(anchor.session_id),
        )

    def _score_primary_actions_with_state(
        self,
        anchor: Anchor,
        pred: Any,
        *,
        frontier: np.ndarray,
        candidate_ids: Sequence[str] | None = None,
    ) -> ProposalDecision:
        x = np.asarray(anchor.z, dtype=np.float64)
        c = self._make_context(anchor.z, np.asarray(frontier, dtype=np.float64), pred.mu)
        if candidate_ids is None:
            candidate_ids = self._make_candidates(x, c, anchor.session_id)
        else:
            candidate_ids = tuple(
                dict.fromkeys(
                    trace_id
                    for trace_id in candidate_ids
                    if trace_id in self.traces and self._trace_matches_session(self.traces[trace_id], anchor.session_id)
                )
            )
        if not candidate_ids:
            observations = self._objective_observations(x, c, pred, trace_ids=tuple(), session_id=anchor.session_id)
            birth_eval = self._score_action_candidate("BIRTH", None, anchor, c, tuple(), pred, observations)
            return ProposalDecision(
                action="BIRTH",
                trace_id=None,
                delta_energy=float(birth_eval.total),
                candidate_ids=tuple(),
                context=c,
                base_energy=0.0,
                top_responsibilities={},
                apply_inhibit=birth_eval.apply_inhibit,
                inhibit_pair=birth_eval.inhibit_pair,
                objective_terms=birth_eval.terms.as_dict(),
            )
        observations = self._objective_observations(x, c, pred, trace_ids=candidate_ids, session_id=anchor.session_id)
        base_terms, responsibilities = self._objective_terms(candidate_ids, x, c, pred, observations=observations)
        evaluations: list[ActionEvaluation] = [
            self._score_action_candidate("BIRTH", None, anchor, c, candidate_ids, pred, observations)
        ]
        for trace_id in candidate_ids:
            evaluations.append(self._score_action_candidate("ASSIMILATE", trace_id, anchor, c, candidate_ids, pred, observations))
            evaluations.append(self._score_action_candidate("ATTACH", trace_id, anchor, c, candidate_ids, pred, observations))
            evaluations.append(self._score_action_candidate("SPLIT", trace_id, anchor, c, candidate_ids, pred, observations))
        best = min(evaluations, key=lambda item: item.total)
        top_resp = dict(sorted(responsibilities.items(), key=lambda item: item[1], reverse=True)[:3])
        return ProposalDecision(
            action=cast(ActionName, best.action),
            trace_id=best.trace_id,
            delta_energy=float(best.total - base_terms.total),
            candidate_ids=tuple(candidate_ids),
            context=c,
            base_energy=float(base_terms.total),
            top_responsibilities=top_resp,
            apply_inhibit=best.apply_inhibit,
            inhibit_pair=best.inhibit_pair,
            objective_terms=best.terms.as_dict(),
        )

    def _anchor_membership_count(self, anchor_id: str) -> int:
        return sum(1 for trace in self.traces.values() if anchor_id in trace.anchor_reservoir)

    def _has_replay_spawn_from_frame(self, frame_id: str) -> bool:
        return any(trace.metadata.get("spawned_from_frame") == frame_id for trace in self.traces.values())

    def _materialize_replay_trace_mutation(
        self,
        decision: ProposalDecision,
        anchor: Anchor,
        *,
        frame: ExperienceFrame,
    ) -> str:
        context = np.asarray(decision.context, dtype=np.float64)
        if decision.action == "BIRTH":
            trace = self._new_trace(anchor, context)
            trace.metadata.update(
                {
                    "spawn_mode": "replay_birth",
                    "spawned_from_frame": frame.frame_id,
                    "source_anchor_ids": [anchor.anchor_id],
                }
            )
        elif decision.action == "SPLIT":
            if decision.trace_id is None or decision.trace_id not in self.traces:
                raise RuntimeError("split mutation requires a live parent trace")
            parent = self.traces[decision.trace_id]
            sigma = np.minimum(np.full_like(parent.z_sigma_diag, self.config.init_sigma), 0.5 * parent.z_sigma_diag)
            trace = self._new_trace(
                anchor,
                parent.ctx_mu.copy(),
                parent_trace_id=parent.trace_id,
                sigma=sigma,
                ctx_sigma=parent.ctx_sigma_diag.copy(),
            )
            trace.fidelity = max(parent.fidelity, 0.85)
            trace.metadata.update(
                {
                    "spawn_mode": "replay_split",
                    "spawned_from_frame": frame.frame_id,
                    "source_anchor_ids": [anchor.anchor_id],
                }
            )
        else:
            raise ValueError(f"unsupported replay structural action: {decision.action}")
        self.trace_store.add(trace)
        self.ann_index.add_or_update(trace)
        return trace.trace_id

    def _apply_primary_action(self, decision: ProposalDecision, anchor: Anchor) -> str:
        x = np.asarray(anchor.z, dtype=np.float64)
        c = np.asarray(decision.context, dtype=np.float64)
        if decision.action == "BIRTH":
            trace = self._new_trace(anchor, c)
            self.trace_store.add(trace)
            self.ann_index.add_or_update(trace)
            return trace.trace_id

        if decision.trace_id is None:
            raise RuntimeError("proposal decision missing trace_id")
        trace = self.traces[decision.trace_id]
        trace.last_access_ts = float(anchor.ts)
        trace.access_ema = self.config.access_decay * trace.access_ema + 1.0
        trace.t_end = max(trace.t_end, float(anchor.ts))
        push_anchor_reference(trace, anchor.anchor_id, reservoir_size=self.config.reservoir_size)
        if decision.action == "ASSIMILATE":
            old_mu = trace.z_mu.copy()
            eta = self.config.eta_mu / (1.0 + self.config.lambda_stability * trace.stability)
            trace.z_mu = trace.z_mu + eta * (x - trace.z_mu)
            trace.z_sigma_diag = safe_variance(
                (1.0 - eta) * trace.z_sigma_diag + eta * np.square(x - old_mu),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            trace.ctx_mu = trace.ctx_mu + self.config.eta_ctx * (c - trace.ctx_mu)
            trace.ctx_sigma_diag = safe_variance(
                (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag + self.config.eta_ctx * np.square(c - trace.ctx_mu),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            trace.evidence = self.config.evidence_decay * trace.evidence + 1.0
        elif decision.action == "ATTACH":
            trace.ctx_mu = trace.ctx_mu + self.config.eta_ctx * (c - trace.ctx_mu)
            trace.ctx_sigma_diag = safe_variance(
                (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag + self.config.eta_ctx * np.square(c - trace.ctx_mu),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            trace.evidence = self.config.evidence_decay * trace.evidence + 1.0
        elif decision.action == "SPLIT":
            sigma = np.minimum(np.full_like(trace.z_sigma_diag, self.config.init_sigma), 0.5 * trace.z_sigma_diag)
            child = self._new_trace(
                anchor,
                trace.ctx_mu.copy(),
                parent_trace_id=trace.trace_id,
                sigma=sigma,
                ctx_sigma=trace.ctx_sigma_diag.copy(),
            )
            child.fidelity = max(trace.fidelity, 0.85)
            self.trace_store.add(child)
            self.ann_index.add_or_update(child)
            return child.trace_id
        else:
            raise ValueError(f"unsupported action: {decision.action}")
        trace.metadata["session_id"] = anchor.session_id
        trace.metadata["turn_id"] = anchor.turn_id
        trace.metadata["source"] = anchor.source
        trace.uncertainty = compute_uncertainty(trace, self.config)
        update_stability(trace, self.config)
        self.ann_index.add_or_update(trace)
        return trace.trace_id

    def _bf_separate(self, left_id: str, right_id: str, x: np.ndarray, c: np.ndarray) -> float:
        left = self.traces[left_id]
        right = self.traces[right_id]
        left_score = trace_log_prior(left, c) + diag_gaussian_logpdf(x, left.z_mu, left.z_sigma_diag)
        right_score = trace_log_prior(right, c) + diag_gaussian_logpdf(x, right.z_mu, right.z_sigma_diag)
        total_evidence = max(left.evidence + right.evidence, EPS)
        mu_ij = (left.evidence * left.z_mu + right.evidence * right.z_mu) / total_evidence
        sigma_ij = (
            left.evidence * (left.z_sigma_diag + np.square(left.z_mu - mu_ij))
            + right.evidence * (right.z_sigma_diag + np.square(right.z_mu - mu_ij))
        ) / total_evidence
        ctx_mu = (left.evidence * left.ctx_mu + right.evidence * right.ctx_mu) / total_evidence
        ctx_sigma = (
            left.evidence * (left.ctx_sigma_diag + np.square(left.ctx_mu - ctx_mu))
            + right.evidence * (right.ctx_sigma_diag + np.square(right.ctx_mu - ctx_mu))
        ) / total_evidence
        merged_score = (
            math.log(total_evidence + EPS)
            + diag_gaussian_logpdf(c, ctx_mu, ctx_sigma)
            + diag_gaussian_logpdf(x, mu_ij, sigma_ij)
        )
        return float(np.logaddexp(left_score, right_score) - merged_score)

    def _context_overlap(self, left_id: str, right_id: str) -> float:
        left = self.traces[left_id]
        right = self.traces[right_id]
        return 0.5 * (1.0 + cosine_similarity(left.ctx_mu, right.ctx_mu))

    def _upsert_edge(self, src: str, dst: str, kind: EdgeKind, *, delta_weight: float, ts: float) -> TraceEdge:
        key = (src, dst, kind)
        edge = self.edges.get(key)
        if edge is None:
            edge = TraceEdge(
                src=src,
                dst=dst,
                kind=kind,
                weight=max(delta_weight, 0.0),
                support_ema=max(delta_weight, 0.0),
                last_update_ts=float(ts),
            )
            self.edge_store.upsert(edge)
        else:
            edge.weight = max(self.config.inhibit_decay * edge.weight + delta_weight, 0.0)
            edge.support_ema = self.config.inhibit_decay * edge.support_ema + max(delta_weight, 0.0)
            edge.last_update_ts = float(ts)
        return edge

    def _find_group_with_members(self, trace_ids: Sequence[str]) -> PosteriorGroup | None:
        target = set(trace_ids)
        for group in self.groups.values():
            if target.issubset(set(group.member_ids)):
                return group
        return None

    def _create_or_extend_group(self, left_id: str, right_id: str) -> PosteriorGroup:
        existing = self._find_group_with_members([left_id, right_id])
        if existing is not None:
            return existing
        candidate_group_ids = set(self.traces[left_id].posterior_group_ids) | set(self.traces[right_id].posterior_group_ids)
        for group_id in candidate_group_ids:
            group = self.groups.get(group_id)
            if group is None:
                continue
            for member_id in (left_id, right_id):
                if member_id not in group.member_ids:
                    group.member_ids.append(member_id)
                    member_trace = self.traces[member_id]
                    group.alpha = np.concatenate([group.alpha[:-1], np.asarray([1.0]), group.alpha[-1:]])
                    group.ctx_mu = np.vstack([group.ctx_mu[:-1], member_trace.ctx_mu[None, :], group.ctx_mu[-1:]])
                    group.ctx_sigma_diag = np.vstack(
                        [group.ctx_sigma_diag[:-1], member_trace.ctx_sigma_diag[None, :], group.ctx_sigma_diag[-1:]]
                    )
                    group.pred_success_ema = np.concatenate(
                        [group.pred_success_ema[:-1], np.asarray([0.5]), group.pred_success_ema[-1:]]
                    )
                    self.traces[member_id].posterior_group_ids = tuple(
                        sorted(set(self.traces[member_id].posterior_group_ids) | {group.group_id})
                    )
            return group
        group_id = f"group_{uuid.uuid4().hex[:10]}"
        left = self.traces[left_id]
        right = self.traces[right_id]
        null_ctx_mu = np.zeros_like(left.ctx_mu)
        null_ctx_sigma = np.full_like(left.ctx_sigma_diag, self.config.null_ctx_sigma)
        group = PosteriorGroup(
            group_id=group_id,
            member_ids=[left_id, right_id],
            alpha=np.asarray([1.0, 1.0, 0.5], dtype=np.float64),
            ctx_mu=np.stack([left.ctx_mu, right.ctx_mu, null_ctx_mu], axis=0),
            ctx_sigma_diag=np.stack([left.ctx_sigma_diag, right.ctx_sigma_diag, null_ctx_sigma], axis=0),
            pred_success_ema=np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
            temperature=1.0,
            unresolved_mass=0.0,
            ambiguous_buffer=deque(maxlen=self.config.unresolved_buffer_maxlen),
            ambiguous_ctx_buffer=deque(maxlen=self.config.unresolved_buffer_maxlen),
        )
        self.groups[group_id] = group
        self.traces[left_id].posterior_group_ids = tuple(
            sorted(set(self.traces[left_id].posterior_group_ids) | {group_id})
        )
        self.traces[right_id].posterior_group_ids = tuple(
            sorted(set(self.traces[right_id].posterior_group_ids) | {group_id})
        )
        return group

    def _posterior_slice(self, group: PosteriorGroup, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return posterior_slice(group, self.traces, x, c, self.config)

    def _spawn_trace_from_unresolved(
        self,
        group: PosteriorGroup,
        anchors: Sequence[Anchor],
        contexts: np.ndarray,
    ) -> str:
        parent_id = None
        if group.member_ids:
            member_scores = group.alpha[:-1] if group.alpha.size > len(group.member_ids) else group.alpha
            if member_scores.size > 0:
                parent_id = group.member_ids[int(np.argmax(member_scores))]
        xs = np.stack([anchor.z for anchor in anchors], axis=0)
        cs = np.asarray(contexts, dtype=np.float64)
        trace = TraceRecord(
            trace_id=f"trace_{uuid.uuid4().hex[:12]}",
            z_mu=np.mean(xs, axis=0),
            z_sigma_diag=safe_variance(
                np.var(xs, axis=0) + self.config.sigma_floor,
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            ),
            ctx_mu=np.mean(cs, axis=0),
            ctx_sigma_diag=safe_variance(
                np.var(cs, axis=0) + self.config.sigma_floor,
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            ),
            evidence=float(max(len(anchors), 1)),
            stability=0.0,
            uncertainty=1.0,
            fidelity=min(1.0, 0.85 + 0.05 * min(len(anchors), 3)),
            compression_stage=0,
            pred_loss_ema=0.0,
            access_ema=1.0,
            last_access_ts=float(max(anchor.ts for anchor in anchors)),
            t_start=float(min(anchor.ts for anchor in anchors)),
            t_end=float(max(anchor.ts for anchor in anchors)),
            anchor_reservoir=deque(),
            parent_trace_id=parent_id,
            metadata={
                "spawned_from_group": group.group_id,
                "spawn_mode": "unresolved_cluster",
                "source_anchor_ids": [anchor.anchor_id for anchor in anchors],
                "session_id": anchors[-1].session_id if anchors else "",
            },
        )
        for anchor in anchors[-self.config.reservoir_size :]:
            push_anchor_reference(trace, anchor.anchor_id, reservoir_size=self.config.reservoir_size)
        trace.uncertainty = compute_uncertainty(trace, self.config)
        self.trace_store.add(trace)
        self.ann_index.add_or_update(trace)
        return trace.trace_id

    def _maybe_spawn_from_unresolved(self, group: PosteriorGroup, *, ts: float) -> str | None:
        if (
            group.unresolved_mass < self.config.group_null_trigger
            or len(group.ambiguous_buffer) < self.config.unresolved_spawn_min_anchors
        ):
            return None
        anchors = [self.anchors[anchor_id] for anchor_id in group.ambiguous_buffer if anchor_id in self.anchors]
        if (
            len(anchors) < self.config.unresolved_spawn_min_anchors
            or len(group.ambiguous_ctx_buffer) < self.config.unresolved_spawn_min_anchors
        ):
            return None
        xs = np.stack([anchor.z for anchor in anchors], axis=0)
        cs = np.stack(list(group.ambiguous_ctx_buffer), axis=0)
        compact = float(np.mean(np.var(xs, axis=0)))
        cohesion = 0.5 * (1.0 + cosine_similarity(np.mean(cs, axis=0), group.ctx_mu[-1]))
        if compact > self.config.unresolved_compact_threshold or cohesion < self.config.unresolved_context_cohesion_threshold:
            return None
        new_trace_id = self._spawn_trace_from_unresolved(group, anchors, cs)
        group.member_ids.append(new_trace_id)
        group.alpha = np.concatenate([group.alpha[:-1], np.asarray([1.0]), group.alpha[-1:]])
        group.ctx_mu = np.vstack([group.ctx_mu[:-1], np.mean(cs, axis=0, keepdims=True), group.ctx_mu[-1:]])
        group.ctx_sigma_diag = np.vstack(
            [
                group.ctx_sigma_diag[:-1],
                safe_variance(
                    np.var(cs, axis=0, keepdims=True) + self.config.sigma_floor,
                    min_value=self.config.sigma_floor,
                    max_value=self.config.sigma_ceiling,
                ),
                group.ctx_sigma_diag[-1:],
            ]
        )
        group.pred_success_ema = np.concatenate([group.pred_success_ema[:-1], np.asarray([0.5]), group.pred_success_ema[-1:]])
        self.traces[new_trace_id].posterior_group_ids = tuple(
            sorted(set(self.traces[new_trace_id].posterior_group_ids) | {group.group_id})
        )
        group.unresolved_mass = 0.0
        group.ambiguous_buffer.clear()
        group.ambiguous_ctx_buffer.clear()
        return new_trace_id

    def _update_group(
        self,
        group: PosteriorGroup,
        x: np.ndarray,
        c: np.ndarray,
        *,
        anchor_id: str,
        ts: float,
        pred_loss: float | None = None,
    ) -> float:
        probs = self._posterior_slice(group, x, c)
        for index, member_id in enumerate(group.member_ids):
            if member_id not in self.traces:
                continue
            weight = probs[index]
            group.alpha[index] = self.config.group_alpha_decay * group.alpha[index] + weight
            group.ctx_mu[index] = group.ctx_mu[index] + self.config.group_ctx_lr * weight * (c - group.ctx_mu[index])
            group.ctx_sigma_diag[index] = safe_variance(
                (1.0 - self.config.group_ctx_lr * weight) * group.ctx_sigma_diag[index]
                + self.config.group_ctx_lr * weight * np.square(c - group.ctx_mu[index]),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            fit = math.exp(
                diag_gaussian_logpdf(x, self.traces[member_id].z_mu, self.traces[member_id].z_sigma_diag)
                / max(x.size, 1)
            )
            if pred_loss is not None:
                fit = max(fit, math.exp(-float(pred_loss)))
            group.pred_success_ema[index] = 0.95 * group.pred_success_ema[index] + 0.05 * weight * fit
        null_index = len(group.member_ids)
        group.alpha[null_index] = self.config.group_alpha_decay * group.alpha[null_index] + probs[null_index]
        group.unresolved_mass = 0.95 * group.unresolved_mass + probs[null_index]
        if probs[null_index] > 0.25 and anchor_id in self.anchors:
            group.ambiguous_buffer.append(anchor_id)
            group.ambiguous_ctx_buffer.append(np.asarray(c, dtype=np.float64).copy())
        target_entropy = self.config.group_target_entropy * math.log(len(group.member_ids) + 1)
        group.temperature = float(
            np.clip(
                group.temperature + self.config.group_temp_lr * (entropy(probs) - target_entropy),
                self.config.group_temp_min,
                self.config.group_temp_max,
            )
        )
        group_entropy = entropy(probs)
        decay = self.config.continuation_group_decay
        group.replay_alignment_ema = decay * group.replay_alignment_ema + (1.0 - decay) * float(max(pred_loss or 0.0, 0.0))
        group.replay_tension_ema = decay * group.replay_tension_ema + (1.0 - decay) * float(group_entropy + group.unresolved_mass)
        spawned = self._maybe_spawn_from_unresolved(group, ts=ts)
        if spawned is None and probs[null_index] > 0.25:
            for member_id, weight in zip(group.member_ids, probs[:-1], strict=False):
                trace = self.traces.get(member_id)
                if trace is None:
                    continue
                trace.z_sigma_diag = safe_variance(
                    trace.z_sigma_diag * (1.0 + self.config.group_uncertainty_inflate * weight),
                    min_value=self.config.sigma_floor,
                    max_value=self.config.sigma_ceiling,
                )
                trace.uncertainty = compute_uncertainty(trace, self.config)
        return group_entropy

    def _apply_inhibit_pair(
        self,
        pair: tuple[str, str],
        x: np.ndarray,
        c: np.ndarray,
        *,
        anchor_id: str,
        ts: float,
        pred_loss: float | None = None,
    ) -> float:
        left_id, right_id = pair
        if left_id not in self.traces or right_id not in self.traces:
            return 0.0
        bf = self._bf_separate(left_id, right_id, x, c)
        delta = self._inhibit_delta(pair, x, c)
        if delta >= 0.0:
            return 0.0
        left_edge = self._upsert_edge(left_id, right_id, "inhib", delta_weight=self.config.inhibit_lr * max(0.0, bf), ts=ts)
        right_edge = self._upsert_edge(right_id, left_id, "inhib", delta_weight=self.config.inhibit_lr * max(0.0, bf), ts=ts)
        left_edge.bf_sep_ema = self.config.inhibit_decay * left_edge.bf_sep_ema + (1.0 - self.config.inhibit_decay) * bf
        right_edge.bf_sep_ema = self.config.inhibit_decay * right_edge.bf_sep_ema + (1.0 - self.config.inhibit_decay) * bf
        ctx_overlap = self._context_overlap(left_id, right_id)
        if left_edge.bf_sep_ema > self.config.group_bf_threshold and ctx_overlap > self.config.group_ctx_overlap_threshold:
            if self._accept_pair_group_mutation(
                left_id,
                right_id,
                ambiguity=1.0,
                bf=left_edge.bf_sep_ema,
                ctx_overlap=ctx_overlap,
            ):
                group = self._create_or_extend_group(left_id, right_id)
                return self._update_group(group, x, c, anchor_id=anchor_id, ts=ts, pred_loss=pred_loss)
        return 0.0

    def _settle_workspace(
        self,
        cue: np.ndarray,
        context: np.ndarray,
        pred_mu: np.ndarray,
        session_id: str,
        workspace_size: int | None = None,
    ) -> Workspace:
        candidates = self._make_candidates(cue, context, session_id)
        return settle_workspace(
            candidate_ids=candidates,
            traces=self.traces,
            edges=self.edges,
            groups=self.groups,
            cue=np.asarray(cue, dtype=np.float64),
            context=np.asarray(context, dtype=np.float64),
            frontier=self.frontier_summary(session_id),
            pred_mu=np.asarray(pred_mu, dtype=np.float64),
            config=self.config,
            posterior_slice_fn=self._posterior_slice,
            session_id=session_id,
            workspace_size=workspace_size,
        )

    def _update_frontier(self, workspace: Workspace, session_id: str) -> None:
        weights = dict(
            sorted(workspace.activation.items(), key=lambda item: item[1], reverse=True)[: self.config.frontier_size]
        )
        self.frontier_weights = weights
        if session_id:
            self.session_frontiers[session_id] = weights

    def _update_assoc_edges(self, workspace: Workspace, *, ts: float) -> None:
        active_ids = list(workspace.active_trace_ids)
        for i, src in enumerate(active_ids):
            for dst in active_ids[i + 1 :]:
                weight = math.sqrt(max(workspace.activation[src] * workspace.activation[dst], 0.0))
                self._upsert_edge(src, dst, "assoc", delta_weight=0.10 * weight, ts=ts)
                self._upsert_edge(dst, src, "assoc", delta_weight=0.10 * weight, ts=ts)

    def _anchor_contexts(self, anchor_ids: Sequence[str]) -> list[np.ndarray]:
        contexts: list[np.ndarray] = []
        wanted = set(anchor_ids)
        if not wanted:
            return contexts
        for frame in reversed(self.frames):
            if frame.anchor_id in wanted:
                contexts.append(np.asarray(frame.context, dtype=np.float64))
        return contexts

    def _trace_utility(self, trace: TraceRecord) -> float:
        role_mass = trace_structural_role_mass(trace)
        forget = trace_forget_risk(trace, now_ts=self.current_ts)
        return float(trace.evidence + trace.stability + trace.access_ema + 0.5 * role_mass - trace.uncertainty - forget)

    def _budget_masses(self) -> tuple[float, float, float]:
        trace_mass = float(sum(trace_storage_cost(trace, self.config) for trace in self.traces.values()))
        edge_mass = float(sum(edge_storage_cost(edge, self.config) for edge in self.edges.values()))
        group_mass = float(sum(group_storage_cost(group, self.config) for group in self.groups.values()))
        return trace_mass, edge_mass, group_mass

    def _budget_pressure(self) -> float:
        trace_mass, edge_mass, group_mass = self._budget_masses()
        return self.budget_controller.pressure(
            trace_mass=trace_mass,
            edge_mass=edge_mass,
            group_mass=group_mass,
        )

    def _rehydrate_trace(self, trace: TraceRecord) -> bool:
        if trace.fidelity >= 0.999:
            return False
        anchors = [self.anchors[anchor_id] for anchor_id in trace.anchor_reservoir if anchor_id in self.anchors]
        if not anchors:
            trace.fidelity = min(1.0, trace.fidelity + 0.5 * self.config.fidelity_expand_step)
            trace.compression_stage = max(trace.compression_stage - 1, 0)
            return True
        xs = np.stack([anchor.z for anchor in anchors], axis=0)
        trace.z_mu = np.mean(xs, axis=0)
        trace.z_sigma_diag = safe_variance(
            np.var(xs, axis=0) + self.config.sigma_floor,
            min_value=self.config.sigma_floor,
            max_value=self.config.sigma_ceiling,
        )
        contexts = self._anchor_contexts(list(trace.anchor_reservoir))
        if contexts:
            cs = np.stack(contexts, axis=0)
            trace.ctx_mu = np.mean(cs, axis=0)
            trace.ctx_sigma_diag = safe_variance(
                np.var(cs, axis=0) + self.config.sigma_floor,
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
        trace.fidelity = min(1.0, trace.fidelity + self.config.fidelity_expand_step)
        trace.compression_stage = max(trace.compression_stage - 1, 0)
        trace.uncertainty = compute_uncertainty(trace, self.config)
        return True

    def _degrade_trace_fidelity(self, trace: TraceRecord) -> bool:
        if trace.fidelity <= self.config.fidelity_min + EPS:
            return False
        trace.fidelity = max(self.config.fidelity_min, trace.fidelity - self.config.fidelity_compress_step)
        trace.compression_stage += 1
        keep = max(1, int(round(len(trace.anchor_reservoir) * self.config.fidelity_reservoir_keep_ratio)))
        while len(trace.anchor_reservoir) > keep:
            trace.anchor_reservoir.popleft()
        trace.z_sigma_diag = safe_variance(
            trace.z_sigma_diag * self.config.fidelity_sigma_inflate,
            min_value=self.config.sigma_floor,
            max_value=self.config.sigma_ceiling,
        )
        trace.ctx_sigma_diag = safe_variance(
            trace.ctx_sigma_diag * self.config.fidelity_sigma_inflate,
            min_value=self.config.sigma_floor,
            max_value=self.config.sigma_ceiling,
        )
        trace.uncertainty = compute_uncertainty(trace, self.config)
        trace.metadata["last_compress_ts"] = self.current_ts
        return True

    def _fidelity_step(self) -> tuple[list[str], list[str]]:
        compressed: list[str] = []
        rehydrated: list[str] = []
        active_ids = set(self.workspace.active_trace_ids) | set(self.hot_trace_ids)
        for trace_id in active_ids:
            trace = self.traces.get(trace_id)
            if trace is None:
                continue
            activation = self.workspace.activation.get(trace_id, 0.0)
            if activation >= self.config.fidelity_restore_activation or trace.access_ema >= self.config.fidelity_restore_activation:
                if self._rehydrate_trace(trace):
                    rehydrated.append(trace_id)
        pressure = self._budget_pressure()
        if pressure <= 1.0:
            return compressed, rehydrated
        candidates = [
            trace
            for trace in self.traces.values()
            if trace.trace_id not in active_ids
            and trace_structural_role_mass(trace) < self.config.active_role_threshold
        ]
        candidates.sort(
            key=lambda trace: self._trace_utility(trace) / max(trace_storage_cost(trace, self.config), EPS)
        )
        for trace in candidates:
            if self._budget_pressure() <= 1.0:
                break
            if self._degrade_trace_fidelity(trace):
                compressed.append(trace.trace_id)
        return compressed, rehydrated

    def _replay_trace_mutations(self, batch: Sequence[ExperienceFrame]) -> list[str]:
        if not batch:
            return []
        scored: list[tuple[float, ExperienceFrame]] = []
        for frame in batch:
            pred_score = self.predictor.score_transition(frame) if frame.next_x is not None else 0.0
            priority = float(pred_score + frame.group_entropy + self._anchor_membership_count(frame.anchor_id))
            scored.append((priority, frame))
        scored.sort(key=lambda item: item[0], reverse=True)
        max_new = max(1, int(round(len(batch) * self.config.replay_mutation_max_ratio)))
        created: list[str] = []
        for _, frame in scored:
            if len(created) >= max_new:
                break
            anchor = self.anchors.get(frame.anchor_id)
            if anchor is None:
                continue
            if self._has_replay_spawn_from_frame(frame.frame_id):
                continue
            if self._anchor_membership_count(anchor.anchor_id) > 1:
                continue
            delta_t = 1.0 if frame.next_ts is None else max(float(frame.next_ts - frame.ts), 1e-3)
            pred = self.predictor.peek(frame.workspace_vec, frame.frontier_vec, frame.action_vec, delta_t=delta_t)
            replay_candidate_ids = tuple(
                dict.fromkeys(
                    trace_id
                    for trace_id in (*frame.active_ids, frame.trace_id, *self._make_candidates(frame.x, frame.context, frame.session_id))
                    if trace_id in self.traces
                )
            )
            decision = self._score_primary_actions_with_state(
                anchor,
                pred,
                frontier=np.asarray(frame.frontier_vec, dtype=np.float64),
                candidate_ids=replay_candidate_ids,
            )
            if decision.action not in {"BIRTH", "SPLIT"}:
                if (
                    decision.apply_inhibit
                    and decision.inhibit_pair is not None
                    and decision.delta_energy <= -0.5 * self.config.replay_mutation_min_delta
                ):
                    self._apply_inhibit_pair(
                        decision.inhibit_pair,
                        np.asarray(anchor.z, dtype=np.float64),
                        np.asarray(decision.context, dtype=np.float64),
                        anchor_id=anchor.anchor_id,
                        ts=max(frame.next_ts or frame.ts, anchor.ts),
                        pred_loss=decision.objective_terms.get("slow_alignment", 0.0),
                    )
                continue
            if decision.delta_energy > -self.config.replay_mutation_min_delta:
                continue
            new_trace_id = self._materialize_replay_trace_mutation(decision, anchor, frame=frame)
            created.append(new_trace_id)
            if decision.apply_inhibit and decision.inhibit_pair is not None:
                self._apply_inhibit_pair(
                    decision.inhibit_pair,
                    np.asarray(anchor.z, dtype=np.float64),
                    np.asarray(decision.context, dtype=np.float64),
                    anchor_id=anchor.anchor_id,
                    ts=max(frame.next_ts or frame.ts, anchor.ts),
                    pred_loss=decision.objective_terms.get("slow_alignment", 0.0),
                )
        return list(dict.fromkeys(created))

    def _replay_groups_from_batch(self, batch: Sequence[ExperienceFrame]) -> list[str]:
        touched: list[str] = []
        for frame in batch:
            local_group_ids: set[str] = set()
            trace = self.traces.get(frame.trace_id)
            if trace is not None:
                local_group_ids.update(trace.posterior_group_ids)
            ranked = sorted(frame.activation.items(), key=lambda item: item[1], reverse=True)
            if len(local_group_ids) == 0 and len(ranked) >= 2:
                left_id, left_mass = ranked[0]
                right_id, right_mass = ranked[1]
                if left_id in self.traces and right_id in self.traces and abs(left_mass - right_mass) <= self.config.inhibit_margin:
                    bf = self._bf_separate(left_id, right_id, frame.x, frame.context)
                    ctx_overlap = self._context_overlap(left_id, right_id)
                    ambiguity = 1.0 - abs(left_mass - right_mass)
                    if (
                        bf > self.config.group_bf_threshold
                        and ctx_overlap > self.config.group_ctx_overlap_threshold
                        and self._accept_pair_group_mutation(
                            left_id,
                            right_id,
                            ambiguity=ambiguity,
                            bf=bf,
                            ctx_overlap=ctx_overlap,
                        )
                    ):
                        group = self._create_or_extend_group(left_id, right_id)
                        local_group_ids.add(group.group_id)
            for group_id in local_group_ids:
                local_group = self.groups.get(group_id)
                if local_group is None:
                    continue
                pred_loss = self.predictor.score_transition(frame) if frame.next_x is not None else 0.0
                self._update_group(
                    local_group,
                    frame.x,
                    frame.context,
                    anchor_id=frame.anchor_id,
                    ts=frame.ts,
                    pred_loss=pred_loss,
                )
                touched.append(group_id)
        return list(dict.fromkeys(touched))

    def _role_lifecycle_step(self) -> list[str]:
        demoted: list[str] = []
        pressure = self._budget_pressure()
        for trace in self.traces.values():
            for role in ("prototype", "procedure"):
                trace.role_support[role] = self.config.role_support_decay * trace.role_support.get(role, 0.0)
                trace.role_gain_ema[role] = self.config.role_gain_decay * trace.role_gain_ema.get(role, 0.0)
                if trace.role_logits.get(role, 0.0) <= 0.0:
                    continue
                support = trace.role_support.get(role, 0.0)
                gain = trace.role_gain_ema.get(role, 0.0)
                threshold = (
                    self.config.prototype_demote_threshold
                    if role == "prototype"
                    else self.config.procedure_demote_threshold
                )
                if support >= threshold and gain > 0.0 and pressure < self.config.role_demote_pressure:
                    continue
                if self._accept_role_decay(trace, role=role, pressure=pressure):
                    demoted.append(trace.trace_id)
        return list(dict.fromkeys(demoted))

    # ------------------------------------------------------------------
    # replay + promotion + budget
    # ------------------------------------------------------------------
    def _frame_priority(self, frame: ExperienceFrame) -> float:
        trace = self.traces.get(frame.trace_id)
        if trace is None:
            return 0.0
        pred_term = max(trace.pred_loss_ema, 0.0) + 1e-4
        uncertainty_term = max(trace.uncertainty, 0.0) + 1e-4
        group_term = max(frame.group_entropy, 0.0) + 1e-4
        centrality = 1.0 + sum(
            1 for edge in self.edges.values() if edge.src == frame.trace_id or edge.dst == frame.trace_id
        )
        forget = trace_forget_risk(trace, now_ts=self.current_ts) + 1e-4
        return float(
            pred_term ** self.replay_config.priority_alpha_pred
            * uncertainty_term ** self.replay_config.priority_alpha_uncertainty
            * group_term ** self.replay_config.priority_alpha_group
            * centrality ** self.replay_config.priority_alpha_centrality
            * forget ** self.replay_config.priority_alpha_forget
        )

    def _weighted_sample(self, frames: Sequence[ExperienceFrame], sample_size: int) -> list[ExperienceFrame]:
        frames = list(frames)
        if not frames or sample_size <= 0:
            return []
        priorities = np.asarray([self._frame_priority(frame) for frame in frames], dtype=np.float64)
        if float(np.sum(priorities)) <= 0.0:
            priorities = np.ones_like(priorities)
        probs = priorities / np.sum(priorities)
        indices = self.rng.choice(len(frames), size=min(sample_size, len(frames)), replace=False, p=probs)
        return [frames[int(index)] for index in indices.tolist()]

    def _sample_replay_batch(self, ms_budget: int | None = None) -> list[ExperienceFrame]:
        complete_frames = [
            frame for frame in self.frames if frame.next_x is not None and frame.trace_id in self.traces
        ]
        if not complete_frames:
            return []
        target_ms = self.config.maintenance_ms_budget if ms_budget is None else max(int(ms_budget), 1)
        batch_size = min(self.replay_config.batch_size, max(4, target_ms // 2))
        n_trace = max(int(round(batch_size * self.replay_config.trace_mix)), 1)
        n_conflict = max(int(round(batch_size * self.replay_config.conflict_mix)), 0)
        n_path = max(batch_size - n_trace - n_conflict, 0)
        conflict_pool = [
            frame
            for frame in complete_frames
            if frame.group_entropy > 0.05 or bool(self.traces[frame.trace_id].posterior_group_ids)
        ]
        path_pool = [frame for frame in complete_frames if frame.next_trace_id is not None]
        selected: dict[str, ExperienceFrame] = {}
        for frame in self._weighted_sample(complete_frames, n_trace):
            selected[frame.frame_id] = frame
        for frame in self._weighted_sample(conflict_pool, n_conflict):
            selected[frame.frame_id] = frame
        for frame in self._weighted_sample(path_pool, n_path):
            selected[frame.frame_id] = frame
        if len(selected) < batch_size:
            for frame in self._weighted_sample(complete_frames, batch_size - len(selected)):
                selected[frame.frame_id] = frame
        return list(selected.values())

    def _reconsolidate_batch(self, batch: Sequence[ExperienceFrame]) -> list[str]:
        replayed: list[str] = []
        for frame in batch:
            responsibilities = {
                trace_id: float(weight)
                for trace_id, weight in frame.activation.items()
                if trace_id in self.traces and weight > 0.0
            }
            if frame.trace_id in self.traces and frame.trace_id not in responsibilities:
                responsibilities[frame.trace_id] = 1.0
            total = float(sum(responsibilities.values()))
            if total <= EPS:
                continue
            responsibilities = {
                trace_id: weight / total for trace_id, weight in responsibilities.items()
            }
            pred_loss = self.predictor.score_transition(frame) if frame.next_x is not None else 0.0
            primary_trace_id = max(
                responsibilities.items(),
                key=lambda item: (item[1], item[0] == frame.trace_id),
            )[0]
            for trace_id, responsibility in responsibilities.items():
                trace = self.traces.get(trace_id)
                if trace is None:
                    continue
                reconsolidate_trace(
                    trace,
                    np.asarray(frame.x, dtype=np.float64),
                    np.asarray(frame.context, dtype=np.float64),
                    config=self.config,
                    responsibility=responsibility,
                    pred_loss=pred_loss,
                    ts=frame.ts,
                    anchor_id=frame.anchor_id if trace_id == primary_trace_id else None,
                )
                self.ann_index.add_or_update(trace)
                replayed.append(trace.trace_id)
        return replayed

    def _trace_role_objective(self, trace: TraceRecord) -> float:
        return float(
            objective_from_components(
                config=self.config,
                surprise=trace.uncertainty,
                storage=trace_storage_cost(trace, self.config),
                complexity=math.log1p(self._degree(trace.trace_id)),
                fidelity=trace_fidelity_cost(trace),
                role=trace_role_cost(trace),
            ).total
        )

    def _edge_mutation_objective(
        self,
        src_id: str,
        dst_id: str,
        *,
        proposed_support: float,
        materialize: bool,
    ) -> float:
        option_edge = self.edges.get((src_id, dst_id, "option"))
        current_support = self._transition_support(src_id, dst_id)
        current_storage = edge_storage_cost(option_edge, self.config) if option_edge is not None else 0.0
        current_complexity = 0.15 if option_edge is not None else 0.0
        before = objective_from_components(
            config=self.config,
            surprise=0.0,
            storage=current_storage,
            complexity=current_complexity,
            drift=1.0 - current_support,
        ).total
        if not materialize:
            return before
        candidate_edge = option_edge or TraceEdge(src=src_id, dst=dst_id, kind="option", weight=0.0)
        return float(
            objective_from_components(
                config=self.config,
                surprise=0.0,
                storage=edge_storage_cost(candidate_edge, self.config),
                complexity=0.15,
                drift=1.0 - max(current_support, proposed_support),
            ).total
        )

    def _accept_option_edge_mutation(self, src_id: str, dst_id: str, *, proposed_support: float) -> bool:
        before = self._edge_mutation_objective(src_id, dst_id, proposed_support=proposed_support, materialize=False)
        after = self._edge_mutation_objective(src_id, dst_id, proposed_support=proposed_support, materialize=True)
        return after < before

    def _pair_group_mutation_objective(
        self,
        left_id: str,
        right_id: str,
        *,
        ambiguity: float,
        bf: float,
        ctx_overlap: float,
        materialize: bool,
    ) -> float:
        existing = self._find_group_with_members([left_id, right_id])
        before_storage = group_storage_cost(existing, self.config) if existing is not None else 0.0
        before_complexity = 0.25 if existing is not None else 0.0
        before = objective_from_components(
            config=self.config,
            surprise=0.0,
            storage=before_storage,
            complexity=before_complexity,
            group_kl=max(ambiguity, 0.0),
        ).total
        if not materialize:
            return before
        if existing is not None:
            candidate_group = existing
        else:
            context_dim = self.config.context_dim
            if context_dim is None:
                raise RuntimeError("FieldConfig.context_dim must be resolved before group mutation scoring")
            null_ctx_mu = np.zeros(context_dim, dtype=np.float64)
            null_ctx_sigma = np.full(context_dim, self.config.null_ctx_sigma, dtype=np.float64)
            candidate_group = PosteriorGroup(
                group_id="__candidate__",
                member_ids=[left_id, right_id],
                alpha=np.asarray([1.0, 1.0, 0.5], dtype=np.float64),
                ctx_mu=np.stack([self.traces[left_id].ctx_mu, self.traces[right_id].ctx_mu, null_ctx_mu], axis=0),
                ctx_sigma_diag=np.stack(
                    [self.traces[left_id].ctx_sigma_diag, self.traces[right_id].ctx_sigma_diag, null_ctx_sigma],
                    axis=0,
                ),
                pred_success_ema=np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
            )
        group_reduction = max(ambiguity - 0.5 * max(bf, 0.0) * max(ctx_overlap, 0.0), 0.0)
        return float(
            objective_from_components(
                config=self.config,
                surprise=0.0,
                storage=group_storage_cost(candidate_group, self.config),
                complexity=0.25,
                group_kl=group_reduction,
            ).total
        )

    def _accept_pair_group_mutation(
        self,
        left_id: str,
        right_id: str,
        *,
        ambiguity: float,
        bf: float,
        ctx_overlap: float,
    ) -> bool:
        before = self._pair_group_mutation_objective(
            left_id,
            right_id,
            ambiguity=ambiguity,
            bf=bf,
            ctx_overlap=ctx_overlap,
            materialize=False,
        )
        after = self._pair_group_mutation_objective(
            left_id,
            right_id,
            ambiguity=ambiguity,
            bf=bf,
            ctx_overlap=ctx_overlap,
            materialize=True,
        )
        return after < before

    def _accept_role_mutation(
        self,
        trace: TraceRecord,
        *,
        role: str,
        member_ids: Sequence[str] | None = None,
        path_signature: Sequence[str] | None = None,
    ) -> bool:
        before = self._trace_role_objective(trace)
        candidate = trace.clone()
        candidate.role_logits[role] = min(candidate.role_logits.get(role, 0.0) + self.config.role_logit_step, 4.0)
        candidate.role_logits["episodic"] = max(candidate.role_logits.get("episodic", 1.0) - 0.05, 0.0)
        if member_ids is not None:
            candidate.member_ids = tuple(sorted(member_ids))
        if path_signature is not None:
            candidate.path_signature = tuple(path_signature)
        after = self._trace_role_objective(candidate)
        delta = after - before
        trace.metadata["last_role_delta"] = delta
        if delta >= 0.0:
            return False
        trace.role_logits = candidate.role_logits
        trace.member_ids = candidate.member_ids
        trace.path_signature = candidate.path_signature
        return True

    def _accept_role_decay(self, trace: TraceRecord, *, role: str, pressure: float) -> bool:
        before = self._trace_role_objective(trace)
        candidate = trace.clone()
        candidate.role_logits[role] = max(candidate.role_logits.get(role, 0.0) - self.config.role_logit_step, 0.0)
        candidate.role_logits["episodic"] = min(
            candidate.role_logits.get("episodic", 0.0) + 0.5 * self.config.role_logit_step,
            4.0,
        )
        if candidate.role_logits[role] <= 0.1:
            if role == "prototype":
                candidate.member_ids = tuple()
            else:
                candidate.path_signature = tuple()
        after = self._trace_role_objective(candidate)
        delta = after - before
        trace.metadata["last_role_delta"] = delta
        if delta > 0.0 and pressure < self.config.role_demote_pressure:
            return False
        trace.role_logits = candidate.role_logits
        trace.member_ids = candidate.member_ids
        trace.path_signature = candidate.path_signature
        return True

    def _update_edges_from_batch(self, batch: Sequence[ExperienceFrame]) -> None:
        for frame in batch:
            if frame.next_trace_id is None:
                continue
            src = frame.trace_id
            dst = frame.next_trace_id
            if src not in self.traces or dst not in self.traces:
                continue
            self._upsert_edge(src, dst, "temporal", delta_weight=self.replay_config.edge_lr, ts=frame.next_ts or frame.ts)
            success = 1.0 - min(self.predictor.score_transition(frame), 1.0)
            if success > self.replay_config.procedure_success_threshold and self._accept_option_edge_mutation(
                src,
                dst,
                proposed_support=success,
            ):
                edge = self._upsert_edge(
                    src,
                    dst,
                    "option",
                    delta_weight=0.5 * self.replay_config.edge_lr * success,
                    ts=frame.next_ts or frame.ts,
                )
                edge.success_alpha = 0.98 * edge.success_alpha + success
                edge.success_beta = 0.98 * edge.success_beta + (1.0 - success)
            for other_id in frame.active_ids:
                if other_id == src or other_id not in self.traces:
                    continue
                self._upsert_edge(src, other_id, "assoc", delta_weight=0.05, ts=frame.ts)

    def _maybe_promote_prototype(self, batch: Sequence[ExperienceFrame]) -> list[str]:
        pattern_support: Counter[frozenset[str]] = Counter()
        promoted: list[str] = []
        for frame in batch:
            if len(frame.active_ids) < 2:
                continue
            top_ids = tuple(sorted(frame.active_ids[: min(3, len(frame.active_ids))]))
            pattern_support[frozenset(top_ids)] += 1
        for member_set, support in pattern_support.items():
            member_ids = [member_id for member_id in member_set if member_id in self.traces]
            if len(member_ids) < 2:
                continue
            mus = np.stack([self.traces[member_id].z_mu for member_id in member_ids], axis=0)
            center = np.mean(mus, axis=0)
            dispersion = float(np.mean(np.sum((mus - center) ** 2, axis=1)))
            future_alignment = float(np.mean([self.traces[member_id].future_alignment_ema for member_id in member_ids]))
            future_drift = float(np.mean([self.traces[member_id].future_drift_ema for member_id in member_ids]))
            compression_gain = (
                (len(member_ids) - 1) * self.config.base_trace_cost
                - dispersion
                - 0.25 * future_alignment
                - 0.10 * future_drift
            )
            medoid_id = min(
                member_ids,
                key=lambda member_id: float(np.sum((self.traces[member_id].z_mu - center) ** 2)),
            )
            trace = self.traces[medoid_id]
            trace.role_support["prototype"] = (
                self.config.role_support_decay * trace.role_support.get("prototype", 0.0) + float(support)
            )
            trace.role_gain_ema["prototype"] = (
                self.config.role_gain_decay * trace.role_gain_ema.get("prototype", 0.0)
                + (1.0 - self.config.role_gain_decay) * float(compression_gain)
            )
            if support < self.replay_config.prototype_support_threshold:
                continue
            if dispersion > self.replay_config.prototype_dispersion_threshold:
                continue
            if compression_gain < self.replay_config.prototype_gain_threshold:
                continue
            if self._accept_role_mutation(trace, role="prototype", member_ids=member_ids):
                promoted.append(trace.trace_id)
        return promoted

    def _maybe_promote_procedure(self, batch: Sequence[ExperienceFrame]) -> list[str]:
        pair_support: Counter[tuple[str, str]] = Counter()
        pair_success: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
        next_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        promoted: list[str] = []
        for frame in batch:
            if frame.next_trace_id is None:
                continue
            pair = (frame.trace_id, frame.next_trace_id)
            pair_support[pair] += 1
            success = 1.0 - min(self.predictor.score_transition(frame), 1.0)
            pair_success[pair].append(success)
            next_counts[frame.trace_id][frame.next_trace_id] += 1
        for pair, support in pair_support.items():
            src_id, dst_id = pair
            if src_id not in self.traces or dst_id not in self.traces:
                continue
            successes = pair_success[pair]
            success_rate = float(np.mean(successes))
            counts = np.asarray(list(next_counts[src_id].values()), dtype=np.float64)
            probs = counts / max(np.sum(counts), 1.0)
            transition_entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
            future_alignment = 0.5 * (self.traces[src_id].future_alignment_ema + self.traces[dst_id].future_alignment_ema)
            future_drift = self.traces[src_id].future_drift_ema
            gain = success_rate - 0.1 * transition_entropy - 0.15 * future_alignment - 0.05 * future_drift
            trace = self.traces[src_id]
            trace.role_support["procedure"] = (
                self.config.role_support_decay * trace.role_support.get("procedure", 0.0) + float(support)
            )
            trace.role_gain_ema["procedure"] = (
                self.config.role_gain_decay * trace.role_gain_ema.get("procedure", 0.0)
                + (1.0 - self.config.role_gain_decay) * float(gain)
            )
            if support < self.replay_config.procedure_support_threshold:
                continue
            if success_rate < self.replay_config.procedure_success_threshold:
                continue
            if transition_entropy > self.replay_config.procedure_entropy_threshold:
                continue
            if gain < self.replay_config.procedure_gain_threshold:
                continue
            if not self._accept_role_mutation(trace, role="procedure", path_signature=(src_id, dst_id)):
                continue
            if self._accept_option_edge_mutation(src_id, dst_id, proposed_support=success_rate):
                edge = self._upsert_edge(src_id, dst_id, "option", delta_weight=0.10 * success_rate, ts=self.current_ts)
                edge.success_alpha = 0.98 * edge.success_alpha + success_rate
                edge.success_beta = 0.98 * edge.success_beta + (1.0 - success_rate)
            promoted.append(trace.trace_id)
        return promoted

    def _budget_step(self) -> list[str]:
        pressure = self._budget_pressure()
        if pressure <= 1.0:
            return []
        prune_fraction = min(max(pressure - 1.0, 0.0) + self.budget_config.pressure_prune_fraction, 0.5)
        active = set(self.workspace.active_trace_ids)
        if len(self.edges) > self.budget_config.max_edges:
            removable_edges = [
                edge for edge in self.edges.values() if edge.src not in active and edge.dst not in active
            ]
            removable_edges.sort(
                key=lambda edge: self.budget_controller.edge_score(edge)
                / max(edge_storage_cost(edge, self.config), EPS)
            )
            n_drop = min(
                int(len(removable_edges) * prune_fraction) + 1,
                len(self.edges) - self.budget_config.max_edges,
            )
            for edge in removable_edges[: max(n_drop, 0)]:
                self.edge_store.remove(edge.key)
        if len(self.groups) > self.budget_config.max_groups:
            removable_groups = sorted(
                self.groups.values(),
                key=lambda group: self.budget_controller.group_score(group)
                / max(group_storage_cost(group, self.config), EPS),
            )
            n_drop = min(
                int(len(removable_groups) * prune_fraction) + 1,
                len(self.groups) - self.budget_config.max_groups,
            )
            for group in removable_groups[: max(n_drop, 0)]:
                self.groups.pop(group.group_id, None)
                for trace_id in group.member_ids:
                    trace = self.traces.get(trace_id)
                    if trace is None:
                        continue
                    trace.posterior_group_ids = tuple(
                        group_id for group_id in trace.posterior_group_ids if group_id != group.group_id
                    )
        pruned: list[str] = []
        if len(self.traces) > self.budget_config.max_traces:
            removable_traces = [
                trace
                for trace in self.traces.values()
                if trace.trace_id not in active
                and trace_structural_role_mass(trace) < self.config.active_role_threshold
                and trace.fidelity <= self.config.fidelity_prune_floor
            ]
            removable_traces.sort(
                key=lambda trace: self.budget_controller.trace_score(trace, now_ts=self.current_ts)
                / max(trace_storage_cost(trace, self.config), EPS)
            )
            n_drop = min(
                int(len(removable_traces) * prune_fraction) + 1,
                len(self.traces) - self.budget_config.max_traces,
            )
            for trace in removable_traces[: max(n_drop, 0)]:
                pruned.append(trace.trace_id)
                self.trace_store.remove(trace.trace_id)
                self.ann_index.remove(trace.trace_id)
                self.edge_store.remove_trace(trace.trace_id)
                for group in self.groups.values():
                    if trace.trace_id in group.member_ids:
                        member_index = group.member_ids.index(trace.trace_id)
                        group.member_ids.pop(member_index)
                        group.alpha = np.delete(group.alpha, member_index)
                        group.ctx_mu = np.delete(group.ctx_mu, member_index, axis=0)
                        group.ctx_sigma_diag = np.delete(group.ctx_sigma_diag, member_index, axis=0)
                        group.pred_success_ema = np.delete(group.pred_success_ema, member_index)
                for session_id, weights in list(self.session_frontiers.items()):
                    weights.pop(trace.trace_id, None)
                    if not weights:
                        self.session_frontiers.pop(session_id, None)
                self.frontier_weights.pop(trace.trace_id, None)
                self._last_trace_by_session = {
                    session_id: last_trace_id
                    for session_id, last_trace_id in self._last_trace_by_session.items()
                    if last_trace_id != trace.trace_id
                }
        return pruned


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
