"""Mutation, grouping, and primary action helpers for AuroraField."""

from __future__ import annotations

import math
import uuid
from collections import deque
from typing import Any, Mapping, Sequence, cast

import numpy as np

from aurora.core.math import EPS, cosine_similarity, diag_gaussian_logpdf, entropy, safe_variance
from aurora.core.types import (
    ActionName,
    Anchor,
    EdgeKind,
    ExperienceFrame,
    PosteriorGroup,
    ProposalDecision,
    TraceEdge,
    TraceRecord,
)
from aurora.replay.reconsolidate import compute_uncertainty, push_anchor_reference, update_stability
from aurora.runtime.objective import (
    ActionEvaluation,
    ObjectiveObservation,
    posterior_slice,
    trace_log_prior,
)


class FieldMutationMixin:
    def _new_trace(
        self: Any,
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
            else safe_variance(
                sigma,
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
        )
        ctx_sigma_diag = (
            np.full_like(c, self.config.init_ctx_sigma)
            if ctx_sigma is None
            else safe_variance(
                ctx_sigma,
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
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

    def _ambiguity_pair(self: Any, responsibilities: Mapping[str, float]) -> tuple[str, str] | None:
        if len(responsibilities) < 2:
            return None
        ranked = sorted(responsibilities.items(), key=lambda item: item[1], reverse=True)
        (left_id, left_prob), (right_id, right_prob) = ranked[:2]
        if left_prob - right_prob > self.config.inhibit_margin:
            return None
        return (left_id, right_id)

    def _inhibit_delta(self: Any, pair: tuple[str, str] | None, x: np.ndarray, c: np.ndarray) -> float:
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
        self: Any,
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
                (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag
                + self.config.eta_ctx * np.square(c - trace.ctx_mu),
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
                (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag
                + self.config.eta_ctx * np.square(c - trace.ctx_mu),
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

    def _score_primary_actions(self: Any, anchor: Anchor, pred: Any) -> ProposalDecision:
        return cast(
            ProposalDecision,
            self._score_primary_actions_with_state(
                anchor,
                pred,
                frontier=self.frontier_summary(anchor.session_id),
            ),
        )

    def _score_primary_actions_with_state(
        self: Any,
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
                    if trace_id in self.traces
                    and self._trace_matches_session(self.traces[trace_id], anchor.session_id)
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
            evaluations.append(
                self._score_action_candidate("ASSIMILATE", trace_id, anchor, c, candidate_ids, pred, observations)
            )
            evaluations.append(
                self._score_action_candidate("ATTACH", trace_id, anchor, c, candidate_ids, pred, observations)
            )
            evaluations.append(
                self._score_action_candidate("SPLIT", trace_id, anchor, c, candidate_ids, pred, observations)
            )
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

    def _anchor_membership_count(self: Any, anchor_id: str) -> int:
        return sum(1 for trace in self.traces.values() if anchor_id in trace.anchor_reservoir)

    def _has_replay_spawn_from_frame(self: Any, frame_id: str) -> bool:
        return any(trace.metadata.get("spawned_from_frame") == frame_id for trace in self.traces.values())

    def _materialize_replay_trace_mutation(
        self: Any,
        decision: ProposalDecision,
        anchor: Anchor,
        *,
        frame: ExperienceFrame,
    ) -> str:
        context = np.asarray(decision.context, dtype=np.float64)
        if decision.action == "BIRTH":
            trace = cast(TraceRecord, self._new_trace(anchor, context))
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
            parent = cast(TraceRecord, self.traces[decision.trace_id])
            sigma = np.minimum(np.full_like(parent.z_sigma_diag, self.config.init_sigma), 0.5 * parent.z_sigma_diag)
            trace = cast(
                TraceRecord,
                self._new_trace(
                    anchor,
                    parent.ctx_mu.copy(),
                    parent_trace_id=parent.trace_id,
                    sigma=sigma,
                    ctx_sigma=parent.ctx_sigma_diag.copy(),
                ),
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
        return str(trace.trace_id)

    def _apply_primary_action(self: Any, decision: ProposalDecision, anchor: Anchor) -> str:
        x = np.asarray(anchor.z, dtype=np.float64)
        c = np.asarray(decision.context, dtype=np.float64)
        if decision.action == "BIRTH":
            trace = cast(TraceRecord, self._new_trace(anchor, c))
            self.trace_store.add(trace)
            self.ann_index.add_or_update(trace)
            return str(trace.trace_id)

        if decision.trace_id is None:
            raise RuntimeError("proposal decision missing trace_id")
        trace = cast(TraceRecord, self.traces[decision.trace_id])
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
                (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag
                + self.config.eta_ctx * np.square(c - trace.ctx_mu),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            trace.evidence = self.config.evidence_decay * trace.evidence + 1.0
        elif decision.action == "ATTACH":
            trace.ctx_mu = trace.ctx_mu + self.config.eta_ctx * (c - trace.ctx_mu)
            trace.ctx_sigma_diag = safe_variance(
                (1.0 - self.config.eta_ctx) * trace.ctx_sigma_diag
                + self.config.eta_ctx * np.square(c - trace.ctx_mu),
                min_value=self.config.sigma_floor,
                max_value=self.config.sigma_ceiling,
            )
            trace.evidence = self.config.evidence_decay * trace.evidence + 1.0
        elif decision.action == "SPLIT":
            sigma = np.minimum(np.full_like(trace.z_sigma_diag, self.config.init_sigma), 0.5 * trace.z_sigma_diag)
            child = cast(
                TraceRecord,
                self._new_trace(
                    anchor,
                    trace.ctx_mu.copy(),
                    parent_trace_id=trace.trace_id,
                    sigma=sigma,
                    ctx_sigma=trace.ctx_sigma_diag.copy(),
                ),
            )
            child.fidelity = max(trace.fidelity, 0.85)
            self.trace_store.add(child)
            self.ann_index.add_or_update(child)
            return str(child.trace_id)
        else:
            raise ValueError(f"unsupported action: {decision.action}")
        trace.metadata["session_id"] = anchor.session_id
        trace.metadata["turn_id"] = anchor.turn_id
        trace.metadata["source"] = anchor.source
        trace.uncertainty = compute_uncertainty(trace, self.config)
        update_stability(trace, self.config)
        self.ann_index.add_or_update(trace)
        return str(trace.trace_id)

    def _bf_separate(self: Any, left_id: str, right_id: str, x: np.ndarray, c: np.ndarray) -> float:
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

    def _context_overlap(self: Any, left_id: str, right_id: str) -> float:
        left = self.traces[left_id]
        right = self.traces[right_id]
        return 0.5 * (1.0 + cosine_similarity(left.ctx_mu, right.ctx_mu))

    def _upsert_edge(
        self: Any,
        src: str,
        dst: str,
        kind: EdgeKind,
        *,
        delta_weight: float,
        ts: float,
    ) -> TraceEdge:
        key = (src, dst, kind)
        edge = cast(TraceEdge | None, self.edges.get(key))
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

    def _find_group_with_members(self: Any, trace_ids: Sequence[str]) -> PosteriorGroup | None:
        target = set(trace_ids)
        for group in cast(dict[str, PosteriorGroup], self.groups).values():
            if target.issubset(set(group.member_ids)):
                return group
        return None

    def _create_or_extend_group(self: Any, left_id: str, right_id: str) -> PosteriorGroup:
        existing = cast(PosteriorGroup | None, self._find_group_with_members([left_id, right_id]))
        if existing is not None:
            return existing
        candidate_group_ids = set(self.traces[left_id].posterior_group_ids) | set(
            self.traces[right_id].posterior_group_ids
        )
        for group_id in candidate_group_ids:
            group = cast(PosteriorGroup | None, self.groups.get(group_id))
            if group is None:
                continue
            for member_id in (left_id, right_id):
                if member_id not in group.member_ids:
                    group.member_ids.append(member_id)
                    member_trace = self.traces[member_id]
                    group.alpha = np.concatenate([group.alpha[:-1], np.asarray([1.0]), group.alpha[-1:]])
                    group.ctx_mu = np.vstack(
                        [group.ctx_mu[:-1], member_trace.ctx_mu[None, :], group.ctx_mu[-1:]]
                    )
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

    def _posterior_slice(self: Any, group: PosteriorGroup, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return posterior_slice(group, self.traces, x, c, self.config)

    def _spawn_trace_from_unresolved(
        self: Any,
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
        return str(trace.trace_id)

    def _maybe_spawn_from_unresolved(self: Any, group: PosteriorGroup, *, ts: float) -> str | None:
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
        if (
            compact > self.config.unresolved_compact_threshold
            or cohesion < self.config.unresolved_context_cohesion_threshold
        ):
            return None
        new_trace_id = str(self._spawn_trace_from_unresolved(group, anchors, cs))
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
        group.pred_success_ema = np.concatenate(
            [group.pred_success_ema[:-1], np.asarray([0.5]), group.pred_success_ema[-1:]]
        )
        self.traces[new_trace_id].posterior_group_ids = tuple(
            sorted(set(self.traces[new_trace_id].posterior_group_ids) | {group.group_id})
        )
        group.unresolved_mass = 0.0
        group.ambiguous_buffer.clear()
        group.ambiguous_ctx_buffer.clear()
        return str(new_trace_id)

    def _update_group(
        self: Any,
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
        group.replay_alignment_ema = decay * group.replay_alignment_ema + (1.0 - decay) * float(
            max(pred_loss or 0.0, 0.0)
        )
        group.replay_tension_ema = decay * group.replay_tension_ema + (1.0 - decay) * float(
            group_entropy + group.unresolved_mass
        )
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
        self: Any,
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
        left_edge = self._upsert_edge(
            left_id,
            right_id,
            "inhib",
            delta_weight=self.config.inhibit_lr * max(0.0, bf),
            ts=ts,
        )
        right_edge = self._upsert_edge(
            right_id,
            left_id,
            "inhib",
            delta_weight=self.config.inhibit_lr * max(0.0, bf),
            ts=ts,
        )
        left_edge.bf_sep_ema = self.config.inhibit_decay * left_edge.bf_sep_ema + (
            1.0 - self.config.inhibit_decay
        ) * bf
        right_edge.bf_sep_ema = self.config.inhibit_decay * right_edge.bf_sep_ema + (
            1.0 - self.config.inhibit_decay
        ) * bf
        ctx_overlap = self._context_overlap(left_id, right_id)
        if (
            left_edge.bf_sep_ema > self.config.group_bf_threshold
            and ctx_overlap > self.config.group_ctx_overlap_threshold
            and self._accept_pair_group_mutation(
                left_id,
                right_id,
                ambiguity=1.0,
                bf=left_edge.bf_sep_ema,
                ctx_overlap=ctx_overlap,
            )
        ):
            group = cast(PosteriorGroup, self._create_or_extend_group(left_id, right_id))
            return float(self._update_group(group, x, c, anchor_id=anchor_id, ts=ts, pred_loss=pred_loss))
        return 0.0
