"""Aurora v2 trace-field runtime."""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from aurora.budget.controller import BudgetController
from aurora.core.config import BudgetConfig, FieldConfig, PredictorConfig, ReplayConfig
from aurora.core.types import ExperienceFrame, InjectResult, MaintenanceStats, PosteriorGroup
from aurora.ingest.anchor_store import AnchorStore
from aurora.ingest.encoder import HashingEncoder
from aurora.ingest.packetizer import Packetizer
from aurora.models.predictor import SlowPredictor
from aurora.runtime._field_maintenance import FieldMaintenanceMixin
from aurora.runtime._field_mutation import FieldMutationMixin
from aurora.runtime._field_objective import FieldObjectiveMixin
from aurora.runtime._field_query import FieldQueryMixin
from aurora.runtime._field_snapshot import FieldSnapshotMixin
from aurora.runtime.proposals import ACTION_ORDER
from aurora.store.ann_index import ExactANNIndex
from aurora.store.blob_store import BlobStore
from aurora.store.edge_store import EdgeStore
from aurora.store.trace_store import TraceStore


class AuroraField(
    FieldSnapshotMixin,
    FieldQueryMixin,
    FieldObjectiveMixin,
    FieldMutationMixin,
    FieldMaintenanceMixin,
):
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

    def _apply_runtime_config(self, config: FieldConfig) -> None:
        if self.config.latent_dim != config.latent_dim:
            raise RuntimeError(
                f"stored field latent_dim {self.config.latent_dim} does not match configured latent_dim "
                f"{config.latent_dim}; use a fresh data directory"
            )
        if self.config.context_dim != config.context_dim:
            raise RuntimeError(
                f"stored field context_dim {self.config.context_dim} does not match configured context_dim "
                f"{config.context_dim}; use a fresh data directory"
            )
        self.config = config
        self.blob_store = BlobStore(self.config.blob_dir)
        self.packetizer = Packetizer(self.config, self.blob_store)
        self.encoder = HashingEncoder(self.config, self.blob_store)
        self.budget_config.max_traces = self.config.trace_budget
        self.budget_config.max_edges = self.config.edge_budget
        self.budget_controller = BudgetController(self.budget_config)

    def inject(self, raw_event: Mapping[str, object] | str) -> InjectResult:
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
            self.objective_ema = 0.95 * self.objective_ema + 0.05 * float(
                decision.objective_terms.get("total", 0.0)
            )

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
