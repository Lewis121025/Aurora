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
    logsumexp,
    safe_variance,
    softmax,
    squared_mahalanobis,
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
            decision = self._score_primary_actions(anchor, pred.mu)
            trace_id = self._apply_primary_action(decision, anchor)
            proposal_kinds.append(decision.action)

            trace = self.traces.get(trace_id)
            if trace is not None:
                trace.access_ema = self.config.access_decay * trace.access_ema + 1.0
                trace.last_access_ts = float(anchor.ts)
                self.ann_index.add_or_update(trace)
            group_entropy = self._maybe_inhibit(decision, anchor)
            if trace is not None:
                for group_id in trace.posterior_group_ids:
                    group = self.groups.get(group_id)
                    if group is not None:
                        group_entropy = max(
                            group_entropy,
                            self._update_group(group, anchor.z, decision.context, anchor_id=anchor.anchor_id, ts=anchor.ts),
                        )

            self.workspace = self._settle_workspace(anchor.z, decision.context, pred.mu, anchor.session_id)
            self._update_frontier(self.workspace, anchor.session_id)
            self._update_assoc_edges(self.workspace, ts=anchor.ts)
            self._record_frame(anchor, trace_id, decision, self.workspace, group_entropy)

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
        batch = self._sample_replay_batch()
        if not batch:
            return MaintenanceStats(elapsed_ms=int((time.time() - started) * 1000))
        predictor_metrics = self.predictor.fit_batch(
            batch,
            train_steps=self.replay_config.train_steps,
            target_ema=self.replay_config.theta_target_ema,
            drift_penalty=self.replay_config.predictor_weight_decay,
        )
        replayed_trace_ids = self._reconsolidate_batch(batch)
        self._update_edges_from_batch(batch)
        prototype_trace_ids = self._maybe_promote_prototype(batch)
        procedure_trace_ids = self._maybe_promote_procedure(batch)
        pruned_trace_ids = self._budget_step()
        self._refresh_hot_traces()
        elapsed_ms = int((time.time() - started) * 1000)
        return MaintenanceStats(
            replayed_trace_ids=replayed_trace_ids,
            prototype_trace_ids=prototype_trace_ids,
            procedure_trace_ids=procedure_trace_ids,
            pruned_trace_ids=pruned_trace_ids,
            elapsed_ms=elapsed_ms,
            replay_batch=len(batch),
            predictor_loss=float(predictor_metrics.get("loss", 0.0)),
            predictor_nll=float(predictor_metrics.get("nll", 0.0)),
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
        return {
            "step": self.step,
            "current_ts": self.current_ts,
            "packet_count": len(self.anchor_store.packets),
            "anchor_count": len(self.anchors),
            "trace_count": len(self.traces),
            "edge_count": len(self.edges),
            "posterior_group_count": len(self.groups),
            "hot_trace_ids": list(self.hot_trace_ids),
            "budget_state": {
                "max_traces": self.budget_config.max_traces,
                "max_edges": self.budget_config.max_edges,
                "max_groups": self.budget_config.max_groups,
                "pressure": self.budget_controller.pressure(
                    trace_count=len(self.traces),
                    edge_count=len(self.edges),
                    group_count=len(self.groups),
                ),
            },
        }

    def to_snapshot_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 5,
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
        }

    @classmethod
    def from_snapshot_payload(cls, payload: Mapping[str, Any]) -> "AuroraField":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != 5:
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
                pred_loss_ema=float(trace_row.get("pred_loss_ema", 0.0)),
                access_ema=float(trace_row.get("access_ema", 0.0)),
                last_access_ts=float(trace_row.get("last_access_ts", 0.0)),
                t_start=float(trace_row.get("t_start", 0.0)),
                t_end=float(trace_row.get("t_end", 0.0)),
                anchor_reservoir=deque(trace_row.get("anchor_reservoir", [])),
                role_logits=dict(trace_row.get("role_logits", {})),
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
                trace.evidence + trace.access_ema + trace.stability - trace.uncertainty + max(trace.role_logits.values())
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
            if max(trace.role_logits.values()) >= self.config.active_role_threshold:
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
                + 0.05 * max(trace.role_logits.values())
                + session_bonus
            )
            rescored.append((score, trace_id))
        rescored.sort(reverse=True)
        return tuple(trace_id for _, trace_id in rescored[: self.config.candidate_size])

    def _trace_log_prior(self, trace: TraceRecord, c: np.ndarray) -> float:
        base = (
            math.log(trace.evidence + EPS)
            + self.config.gamma_stability * trace.stability
            - self.config.gamma_uncertainty * trace.uncertainty
        )
        ctx = diag_gaussian_logpdf(c, trace.ctx_mu, trace.ctx_sigma_diag)
        return float(base + ctx)

    def _null_slot_logprob(self, x: np.ndarray, c: np.ndarray) -> float:
        return float(
            diag_gaussian_logpdf(x, np.zeros_like(x), np.full_like(x, self.config.null_sigma))
            + diag_gaussian_logpdf(c, np.zeros_like(c), np.full_like(c, self.config.null_ctx_sigma))
        )

    def _degree(self, trace_id: str) -> int:
        return sum(1 for edge in self.edges.values() if edge.src == trace_id or edge.dst == trace_id)

    def _local_energy(
        self,
        candidate_ids: Sequence[str],
        x: np.ndarray,
        c: np.ndarray,
        *,
        shadow: Mapping[str, TraceRecord] | None = None,
        extra_trace_cost: float = 0.0,
        extra_edge_cost: float = 0.0,
        drift_penalty: float = 0.0,
    ) -> float:
        shadow = shadow or {}
        scores: list[float] = []
        local_group_ids: set[str] = set()
        for trace_id in candidate_ids:
            trace = shadow.get(trace_id, self.traces[trace_id])
            scores.append(self._trace_log_prior(trace, c) + diag_gaussian_logpdf(x, trace.z_mu, trace.z_sigma_diag))
            local_group_ids.update(trace.posterior_group_ids)
        for trace_id in shadow.keys():
            if trace_id in candidate_ids:
                continue
            trace = shadow[trace_id]
            scores.append(self._trace_log_prior(trace, c) + diag_gaussian_logpdf(x, trace.z_mu, trace.z_sigma_diag))
        scores.append(self._null_slot_logprob(x, c))
        surprise = -logsumexp(scores)
        local_edge_count = sum(1 for edge in self.edges.values() if edge.src in candidate_ids or edge.dst in candidate_ids)
        storage = (
            self.config.base_trace_cost * len(candidate_ids)
            + self.config.base_edge_cost * local_edge_count
            + extra_trace_cost
            + extra_edge_cost
        )
        complexity = sum(math.log1p(self._degree(trace_id)) for trace_id in candidate_ids) + len(local_group_ids)
        return float(
            surprise
            + self.config.lambda_storage * storage
            + self.config.lambda_complexity * complexity
            + self.config.lambda_plasticity * drift_penalty
        )

    def _responsibilities(self, candidate_ids: Sequence[str], x: np.ndarray, c: np.ndarray) -> dict[str, float]:
        if not candidate_ids:
            return {}
        logits = np.asarray(
            [
                self._trace_log_prior(self.traces[trace_id], c)
                + diag_gaussian_logpdf(x, self.traces[trace_id].z_mu, self.traces[trace_id].z_sigma_diag)
                for trace_id in candidate_ids
            ],
            dtype=np.float64,
        )
        probs = softmax(logits)
        return {trace_id: float(prob) for trace_id, prob in zip(candidate_ids, probs.tolist(), strict=False)}

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

    def _split_allowed(self, trace: TraceRecord, x: np.ndarray, c: np.ndarray) -> bool:
        ctx_fit = squared_mahalanobis(c, trace.ctx_mu, trace.ctx_sigma_diag) / max(c.size, 1)
        residual = squared_mahalanobis(x, trace.z_mu, trace.z_sigma_diag) / max(x.size, 1)
        return bool(ctx_fit < 1.5 and residual > self.config.split_residual_threshold and trace.evidence > 1.5)

    def _simulate_assimilate(self, trace_id: str, x: np.ndarray, c: np.ndarray, candidate_ids: Sequence[str]) -> float:
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
        drift = trace.stability * float(np.sum(np.square(trace.z_mu - old_mu)))
        return self._local_energy(candidate_ids, x, c, shadow={trace_id: trace}, drift_penalty=drift)

    def _simulate_attach(self, trace_id: str, x: np.ndarray, c: np.ndarray, candidate_ids: Sequence[str]) -> float:
        trace = self.traces[trace_id].clone()
        trace.ctx_mu = trace.ctx_mu + self.config.eta_ctx * (c - trace.ctx_mu)
        trace.ctx_sigma_diag = safe_variance(
            (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag + self.config.eta_ctx * np.square(c - trace.ctx_mu),
            min_value=self.config.sigma_floor,
            max_value=self.config.sigma_ceiling,
        )
        trace.evidence = self.config.evidence_decay * trace.evidence + 1.0
        trace.uncertainty = compute_uncertainty(trace, self.config)
        return self._local_energy(candidate_ids, x, c, shadow={trace_id: trace}, drift_penalty=0.0)

    def _simulate_split(
        self,
        parent_id: str,
        anchor: Anchor,
        c: np.ndarray,
        candidate_ids: Sequence[str],
    ) -> float:
        parent = self.traces[parent_id]
        sigma = np.minimum(np.full_like(parent.z_sigma_diag, self.config.init_sigma), 0.5 * parent.z_sigma_diag)
        child = self._new_trace(
            anchor,
            parent.ctx_mu.copy(),
            parent_trace_id=parent_id,
            sigma=sigma,
            ctx_sigma=parent.ctx_sigma_diag.copy(),
            trace_id="__split__",
        )
        return self._local_energy(candidate_ids, anchor.z, c, shadow={"__split__": child}, extra_trace_cost=self.config.base_trace_cost)

    def _simulate_birth(self, anchor: Anchor, c: np.ndarray, candidate_ids: Sequence[str]) -> float:
        child = self._new_trace(anchor, c, trace_id="__birth__")
        return self._local_energy(candidate_ids, anchor.z, c, shadow={"__birth__": child}, extra_trace_cost=self.config.base_trace_cost)

    def _score_primary_actions(self, anchor: Anchor, pred_mu: np.ndarray) -> ProposalDecision:
        x = np.asarray(anchor.z, dtype=np.float64)
        frontier = self.frontier_summary(anchor.session_id)
        c = self._make_context(anchor.z, frontier, pred_mu)
        candidate_ids = self._make_candidates(x, c, anchor.session_id)
        if not candidate_ids:
            birth_energy = self._simulate_birth(anchor, c, tuple())
            return ProposalDecision(
                action="BIRTH",
                trace_id=None,
                delta_energy=birth_energy,
                candidate_ids=tuple(),
                context=c,
                base_energy=0.0,
                top_responsibilities={},
            )

        base_energy = self._local_energy(candidate_ids, x, c)
        responsibilities = self._responsibilities(candidate_ids, x, c)
        best_action: ActionName = "BIRTH"
        best_trace_id: str | None = None
        best_energy = self._simulate_birth(anchor, c, candidate_ids)
        for trace_id in candidate_ids:
            assim_energy = self._simulate_assimilate(trace_id, x, c, candidate_ids)
            if assim_energy < best_energy:
                best_action, best_trace_id, best_energy = "ASSIMILATE", trace_id, assim_energy
            attach_energy = self._simulate_attach(trace_id, x, c, candidate_ids)
            if attach_energy < best_energy:
                best_action, best_trace_id, best_energy = "ATTACH", trace_id, attach_energy
            if self._split_allowed(self.traces[trace_id], x, c):
                split_energy = self._simulate_split(trace_id, anchor, c, candidate_ids)
                if split_energy < best_energy:
                    best_action, best_trace_id, best_energy = "SPLIT", trace_id, split_energy
        if best_trace_id is not None and best_action == "ASSIMILATE":
            attach_energy = self._simulate_attach(best_trace_id, x, c, candidate_ids)
            if attach_energy - best_energy < 0.02 and self.traces[best_trace_id].stability >= self.config.attach_stability_threshold:
                best_action, best_energy = "ATTACH", attach_energy
        top_resp = dict(sorted(responsibilities.items(), key=lambda item: item[1], reverse=True)[:3])
        return ProposalDecision(
            action=best_action,
            trace_id=best_trace_id,
            delta_energy=float(best_energy - base_energy),
            candidate_ids=tuple(candidate_ids),
            context=c,
            base_energy=base_energy,
            top_responsibilities=top_resp,
        )

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
        left_score = self._trace_log_prior(left, c) + diag_gaussian_logpdf(x, left.z_mu, left.z_sigma_diag)
        right_score = self._trace_log_prior(right, c) + diag_gaussian_logpdf(x, right.z_mu, right.z_sigma_diag)
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
        scores: list[float] = []
        for index, member_id in enumerate(group.member_ids):
            trace = self.traces[member_id]
            score = (
                math.log(group.alpha[index] + EPS)
                + diag_gaussian_logpdf(c, group.ctx_mu[index], group.ctx_sigma_diag[index])
                + diag_gaussian_logpdf(x, trace.z_mu, trace.z_sigma_diag)
                + 0.15 * math.log(group.pred_success_ema[index] + EPS)
            )
            scores.append(score)
        null_index = len(group.member_ids)
        scores.append(
            math.log(group.alpha[null_index] + EPS)
            + diag_gaussian_logpdf(c, group.ctx_mu[null_index], group.ctx_sigma_diag[null_index])
            + diag_gaussian_logpdf(x, np.zeros_like(x), np.full_like(x, self.config.null_sigma))
        )
        return softmax(scores, temperature=max(group.temperature, 1e-4))

    def _maybe_spawn_from_unresolved(self, group: PosteriorGroup, *, ts: float) -> str | None:
        if group.unresolved_mass < self.config.group_null_trigger or len(group.ambiguous_buffer) < 3:
            return None
        anchors = [self.anchors[anchor_id] for anchor_id in group.ambiguous_buffer if anchor_id in self.anchors]
        if len(anchors) < 3 or len(group.ambiguous_ctx_buffer) < 3:
            return None
        xs = np.stack([anchor.z for anchor in anchors], axis=0)
        cs = np.stack(list(group.ambiguous_ctx_buffer), axis=0)
        compact = float(np.mean(np.var(xs, axis=0)))
        cohesion = 0.5 * (1.0 + cosine_similarity(np.mean(cs, axis=0), group.ctx_mu[-1]))
        if (
            compact > self.config.unresolved_compact_threshold
            or cohesion < self.config.unresolved_context_cohesion_threshold
        ):
            return None
        x_mean = np.mean(xs, axis=0)
        c_mean = np.mean(cs, axis=0)
        anchor = Anchor(
            anchor_id=f"spawn_{uuid.uuid4().hex[:10]}",
            packet_id="spawn",
            session_id="",
            turn_id="spawn",
            source="env",
            z=x_mean,
            ts=ts,
            residual_ref=None,
            meta={"spawned_from_group": group.group_id},
        )
        new_trace = self._new_trace(anchor, c_mean)
        self.trace_store.add(new_trace)
        self.ann_index.add_or_update(new_trace)
        group.member_ids.append(new_trace.trace_id)
        group.alpha = np.concatenate([group.alpha[:-1], np.asarray([1.0]), group.alpha[-1:]])
        group.ctx_mu = np.vstack([group.ctx_mu[:-1], c_mean[None, :], group.ctx_mu[-1:]])
        group.ctx_sigma_diag = np.vstack(
            [group.ctx_sigma_diag[:-1], np.var(cs, axis=0, keepdims=True) + self.config.sigma_floor, group.ctx_sigma_diag[-1:]]
        )
        group.pred_success_ema = np.concatenate([group.pred_success_ema[:-1], np.asarray([0.5]), group.pred_success_ema[-1:]])
        self.traces[new_trace.trace_id].posterior_group_ids = tuple(
            sorted(set(self.traces[new_trace.trace_id].posterior_group_ids) | {group.group_id})
        )
        group.unresolved_mass = 0.0
        group.ambiguous_buffer.clear()
        group.ambiguous_ctx_buffer.clear()
        return new_trace.trace_id

    def _update_group(self, group: PosteriorGroup, x: np.ndarray, c: np.ndarray, *, anchor_id: str, ts: float) -> float:
        probs = self._posterior_slice(group, x, c)
        for index, member_id in enumerate(group.member_ids):
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
            group.pred_success_ema[index] = 0.95 * group.pred_success_ema[index] + 0.05 * weight * fit
        null_index = len(group.member_ids)
        group.alpha[null_index] = self.config.group_alpha_decay * group.alpha[null_index] + probs[null_index]
        group.unresolved_mass = 0.95 * group.unresolved_mass + probs[null_index]
        if probs[null_index] > 0.25:
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
        self._maybe_spawn_from_unresolved(group, ts=ts)
        return entropy(probs)

    def _maybe_inhibit(self, decision: ProposalDecision, anchor: Anchor) -> float:
        responsibilities = decision.top_responsibilities
        if len(responsibilities) < 2:
            return 0.0
        ranked = sorted(responsibilities.items(), key=lambda item: item[1], reverse=True)
        (left_id, left_prob), (right_id, right_prob) = ranked[:2]
        if left_prob - right_prob > self.config.inhibit_margin:
            return 0.0
        x = np.asarray(anchor.z, dtype=np.float64)
        c = np.asarray(decision.context, dtype=np.float64)
        bf = self._bf_separate(left_id, right_id, x, c)
        delta = (
            -self.config.lambda_inhibit_gain * max(0.0, bf)
            + self.config.lambda_storage * self.config.base_edge_cost
            + self.config.lambda_complexity * 0.1
        )
        if delta >= 0.0:
            return 0.0
        left_edge = self._upsert_edge(left_id, right_id, "inhib", delta_weight=self.config.inhibit_lr * max(0.0, bf), ts=anchor.ts)
        right_edge = self._upsert_edge(right_id, left_id, "inhib", delta_weight=self.config.inhibit_lr * max(0.0, bf), ts=anchor.ts)
        left_edge.bf_sep_ema = self.config.inhibit_decay * left_edge.bf_sep_ema + (1.0 - self.config.inhibit_decay) * bf
        right_edge.bf_sep_ema = self.config.inhibit_decay * right_edge.bf_sep_ema + (1.0 - self.config.inhibit_decay) * bf
        ctx_overlap = self._context_overlap(left_id, right_id)
        if left_edge.bf_sep_ema > self.config.group_bf_threshold and ctx_overlap > self.config.group_ctx_overlap_threshold:
            group = self._create_or_extend_group(left_id, right_id)
            return self._update_group(group, x, c, anchor_id=anchor.anchor_id, ts=anchor.ts)
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

    def _sample_replay_batch(self) -> list[ExperienceFrame]:
        complete_frames = [
            frame for frame in self.frames if frame.next_x is not None and frame.trace_id in self.traces
        ]
        if not complete_frames:
            return []
        batch_size = self.replay_config.batch_size
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
            trace = self.traces.get(frame.trace_id)
            if trace is None:
                continue
            pred_loss = self.predictor.score_transition(frame) if frame.next_x is not None else 0.0
            reconsolidate_trace(
                trace,
                np.asarray(frame.x, dtype=np.float64),
                np.asarray(frame.context, dtype=np.float64),
                config=self.config,
                responsibility=1.0,
                pred_loss=pred_loss,
                ts=frame.ts,
                anchor_id=frame.anchor_id,
            )
            self.ann_index.add_or_update(trace)
            replayed.append(trace.trace_id)
        return replayed

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
            if success > self.replay_config.procedure_success_threshold:
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
            if support < self.replay_config.prototype_support_threshold:
                continue
            member_ids = [member_id for member_id in member_set if member_id in self.traces]
            if len(member_ids) < 2:
                continue
            mus = np.stack([self.traces[member_id].z_mu for member_id in member_ids], axis=0)
            center = np.mean(mus, axis=0)
            dispersion = float(np.mean(np.sum((mus - center) ** 2, axis=1)))
            compression_gain = (len(member_ids) - 1) * self.config.base_trace_cost - dispersion
            if (
                dispersion > self.replay_config.prototype_dispersion_threshold
                or compression_gain < self.replay_config.prototype_gain_threshold
            ):
                continue
            medoid_id = min(
                member_ids,
                key=lambda member_id: float(np.sum((self.traces[member_id].z_mu - center) ** 2)),
            )
            trace = self.traces[medoid_id]
            trace.role_logits["prototype"] = min(trace.role_logits.get("prototype", 0.0) + 0.25, 4.0)
            trace.member_ids = tuple(sorted(member_ids))
            trace.role_logits["episodic"] = max(trace.role_logits.get("episodic", 1.0) - 0.05, 0.0)
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
            if support < self.replay_config.procedure_support_threshold:
                continue
            src, dst = pair
            if src not in self.traces or dst not in self.traces:
                continue
            successes = pair_success[pair]
            success_rate = float(np.mean(successes))
            counts = np.asarray(list(next_counts[src].values()), dtype=np.float64)
            probs = counts / max(np.sum(counts), 1.0)
            transition_entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
            gain = success_rate - 0.1 * transition_entropy
            if success_rate < self.replay_config.procedure_success_threshold:
                continue
            if transition_entropy > self.replay_config.procedure_entropy_threshold:
                continue
            if gain < self.replay_config.procedure_gain_threshold:
                continue
            trace = self.traces[src]
            trace.role_logits["procedure"] = min(trace.role_logits.get("procedure", 0.0) + 0.25, 4.0)
            trace.path_signature = (src, dst)
            trace.role_logits["episodic"] = max(trace.role_logits.get("episodic", 1.0) - 0.05, 0.0)
            edge = self._upsert_edge(src, dst, "option", delta_weight=0.10 * success_rate, ts=self.current_ts)
            edge.success_alpha = 0.98 * edge.success_alpha + success_rate
            edge.success_beta = 0.98 * edge.success_beta + (1.0 - success_rate)
            promoted.append(trace.trace_id)
        return promoted

    def _budget_step(self) -> list[str]:
        pressure = self.budget_controller.pressure(
            trace_count=len(self.traces),
            edge_count=len(self.edges),
            group_count=len(self.groups),
        )
        if pressure <= 1.0:
            return []
        prune_fraction = min(max(pressure - 1.0, 0.0) + self.budget_config.pressure_prune_fraction, 0.5)
        active = set(self.workspace.active_trace_ids)
        if len(self.edges) > self.budget_config.max_edges:
            removable_edges = [
                edge for edge in self.edges.values() if edge.src not in active and edge.dst not in active
            ]
            removable_edges.sort(key=self.budget_controller.edge_score)
            n_drop = min(
                int(len(removable_edges) * prune_fraction) + 1,
                len(self.edges) - self.budget_config.max_edges,
            )
            for edge in removable_edges[: max(n_drop, 0)]:
                self.edge_store.remove(edge.key)
        if len(self.groups) > self.budget_config.max_groups:
            removable_groups = sorted(self.groups.values(), key=self.budget_controller.group_score)
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
                if trace.trace_id not in active and max(trace.role_logits.values()) < self.config.active_role_threshold
            ]
            removable_traces.sort(
                key=lambda trace: self.budget_controller.trace_score(trace, now_ts=self.current_ts)
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
