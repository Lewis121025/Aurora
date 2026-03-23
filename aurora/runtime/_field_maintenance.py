"""Replay, promotion, fidelity, and budget helpers for AuroraField."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence, cast

import numpy as np

from aurora.core.math import EPS, safe_variance
from aurora.core.types import ExperienceFrame, PosteriorGroup, TraceEdge, TraceRecord
from aurora.replay.reconsolidate import compute_uncertainty, reconsolidate_trace, trace_forget_risk
from aurora.runtime.objective import (
    ObjectiveObservation,
    ObjectiveTerms,
    edge_storage_cost,
    empirical_patch_terms,
    group_storage_cost,
    group_tension_cost,
    objective_from_components,
    safe_kl,
    trace_fidelity_cost,
    trace_role_cost,
    trace_storage_cost,
    trace_structural_role_mass,
)


class FieldMaintenanceMixin:
    def _update_continuation_stats(self: Any, batch: Sequence[ExperienceFrame]) -> None:
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
            group.replay_alignment_ema = group_decay * group.replay_alignment_ema + (1.0 - decay) * float(
                np.mean(values)
            )
        for group in self.groups.values():
            group.replay_tension_ema = group_decay * group.replay_tension_ema + (1.0 - decay) * float(
                group.unresolved_mass + abs(group.temperature - 1.0)
            )

    def _replay_structural_objective(
        self: Any,
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
            extra_drift=activation_kl
            + transition_gap
            + structural_churn
            + float(self.workspace.metadata.get("group_kl", 0.0)),
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

    def _budget_masses(self: Any) -> tuple[float, float, float]:
        trace_mass = float(sum(trace_storage_cost(trace, self.config) for trace in self.traces.values()))
        edge_mass = float(sum(edge_storage_cost(edge, self.config) for edge in self.edges.values()))
        group_mass = float(sum(group_storage_cost(group, self.config) for group in self.groups.values()))
        return trace_mass, edge_mass, group_mass

    def _budget_pressure(self: Any) -> float:
        trace_mass, edge_mass, group_mass = self._budget_masses()
        return cast(
            float,
            self.budget_controller.pressure(
                trace_mass=trace_mass,
                edge_mass=edge_mass,
                group_mass=group_mass,
            ),
        )

    def _rehydrate_trace(self: Any, trace: TraceRecord) -> bool:
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

    def _degrade_trace_fidelity(self: Any, trace: TraceRecord) -> bool:
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

    def _fidelity_step(self: Any) -> tuple[list[str], list[str]]:
        compressed: list[str] = []
        rehydrated: list[str] = []
        active_ids = set(self.workspace.active_trace_ids) | set(self.hot_trace_ids)
        for trace_id in active_ids:
            trace = self.traces.get(trace_id)
            if trace is None:
                continue
            activation = self.workspace.activation.get(trace_id, 0.0)
            if (
                activation >= self.config.fidelity_restore_activation
                or trace.access_ema >= self.config.fidelity_restore_activation
            ) and self._rehydrate_trace(trace):
                rehydrated.append(trace_id)
        if self._budget_pressure() <= 1.0:
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

    def _replay_trace_mutations(self: Any, batch: Sequence[ExperienceFrame]) -> list[str]:
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
                    for trace_id in (
                        *frame.active_ids,
                        frame.trace_id,
                        *self._make_candidates(frame.x, frame.context, frame.session_id),
                    )
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

    def _replay_groups_from_batch(self: Any, batch: Sequence[ExperienceFrame]) -> list[str]:
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
                if (
                    left_id in self.traces
                    and right_id in self.traces
                    and abs(left_mass - right_mass) <= self.config.inhibit_margin
                ):
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

    def _role_lifecycle_step(self: Any) -> list[str]:
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

    def _frame_priority(self: Any, frame: ExperienceFrame) -> float:
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

    def _weighted_sample(self: Any, frames: Sequence[ExperienceFrame], sample_size: int) -> list[ExperienceFrame]:
        frames = list(frames)
        if not frames or sample_size <= 0:
            return []
        priorities = np.asarray([self._frame_priority(frame) for frame in frames], dtype=np.float64)
        if float(np.sum(priorities)) <= 0.0:
            priorities = np.ones_like(priorities)
        probs = priorities / np.sum(priorities)
        indices = self.rng.choice(len(frames), size=min(sample_size, len(frames)), replace=False, p=probs)
        return [frames[int(index)] for index in indices.tolist()]

    def _sample_replay_batch(self: Any, ms_budget: int | None = None) -> list[ExperienceFrame]:
        complete_frames = [frame for frame in self.frames if frame.next_x is not None and frame.trace_id in self.traces]
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

    def _reconsolidate_batch(self: Any, batch: Sequence[ExperienceFrame]) -> list[str]:
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
            responsibilities = {trace_id: weight / total for trace_id, weight in responsibilities.items()}
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

    def _trace_role_objective(self: Any, trace: TraceRecord) -> float:
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
        self: Any,
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

    def _accept_option_edge_mutation(
        self: Any,
        src_id: str,
        dst_id: str,
        *,
        proposed_support: float,
    ) -> bool:
        before = cast(
            float,
            self._edge_mutation_objective(src_id, dst_id, proposed_support=proposed_support, materialize=False),
        )
        after = cast(
            float,
            self._edge_mutation_objective(src_id, dst_id, proposed_support=proposed_support, materialize=True),
        )
        return after < before

    def _pair_group_mutation_objective(
        self: Any,
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
        self: Any,
        left_id: str,
        right_id: str,
        *,
        ambiguity: float,
        bf: float,
        ctx_overlap: float,
    ) -> bool:
        before = cast(
            float,
            self._pair_group_mutation_objective(
                left_id,
                right_id,
                ambiguity=ambiguity,
                bf=bf,
                ctx_overlap=ctx_overlap,
                materialize=False,
            ),
        )
        after = cast(
            float,
            self._pair_group_mutation_objective(
                left_id,
                right_id,
                ambiguity=ambiguity,
                bf=bf,
                ctx_overlap=ctx_overlap,
                materialize=True,
            ),
        )
        return after < before

    def _accept_role_mutation(
        self: Any,
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

    def _accept_role_decay(self: Any, trace: TraceRecord, *, role: str, pressure: float) -> bool:
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

    def _update_edges_from_batch(self: Any, batch: Sequence[ExperienceFrame]) -> None:
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

    def _maybe_promote_prototype(self: Any, batch: Sequence[ExperienceFrame]) -> list[str]:
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
            future_alignment = float(
                np.mean([self.traces[member_id].future_alignment_ema for member_id in member_ids])
            )
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

    def _maybe_promote_procedure(self: Any, batch: Sequence[ExperienceFrame]) -> list[str]:
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
            future_alignment = 0.5 * (
                self.traces[src_id].future_alignment_ema + self.traces[dst_id].future_alignment_ema
            )
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
                edge = self._upsert_edge(
                    src_id,
                    dst_id,
                    "option",
                    delta_weight=0.10 * success_rate,
                    ts=self.current_ts,
                )
                edge.success_alpha = 0.98 * edge.success_alpha + success_rate
                edge.success_beta = 0.98 * edge.success_beta + (1.0 - success_rate)
            promoted.append(trace.trace_id)
        return promoted

    def _budget_step(self: Any) -> list[str]:
        pressure = self._budget_pressure()
        if pressure <= 1.0:
            return []
        prune_fraction = min(max(pressure - 1.0, 0.0) + self.budget_config.pressure_prune_fraction, 0.5)
        active = set(self.workspace.active_trace_ids)
        if len(self.edges) > self.budget_config.max_edges:
            removable_edges = [edge for edge in self.edges.values() if edge.src not in active and edge.dst not in active]
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
