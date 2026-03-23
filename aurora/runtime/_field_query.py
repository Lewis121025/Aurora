"""Workspace, frontier, and query helpers for AuroraField."""

from __future__ import annotations

import math
import uuid
from typing import Any, Mapping, Sequence, cast

import numpy as np

from aurora.core.math import cosine_similarity, l2_normalize, weighted_mean
from aurora.core.types import Anchor, ExperienceFrame, ProposalDecision, TraceRecord, Workspace
from aurora.readout.workspace import settle_workspace
from aurora.replay.reconsolidate import trace_forget_risk
from aurora.runtime.objective import trace_structural_role_mass
from aurora.runtime.proposals import ACTION_ORDER, action_vector


class FieldQueryMixin:
    def read_workspace(self: Any, cue: str | Mapping[str, Any], k: int | None = None) -> Workspace:
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
        return cast(Workspace, self._settle_workspace(cue_vec, context, pred.mu, session_id, workspace_size=k))

    def _empty_workspace(self: Any) -> Workspace:
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

    def frontier_summary(self: Any, session_id: str = "") -> np.ndarray:
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
        self: Any,
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

    def _refresh_hot_traces(self: Any) -> None:
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

    def _trace_matches_session(self: Any, trace: TraceRecord, session_id: str) -> bool:
        if not session_id:
            return True
        return trace.metadata.get("session_id") == session_id

    def _make_context(self: Any, q: np.ndarray, frontier: np.ndarray, pred_mu: np.ndarray) -> np.ndarray:
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

    def _neighbors(self: Any, trace_id: str) -> set[str]:
        out: set[str] = set()
        for edge in self.edges.values():
            if edge.src == trace_id:
                out.add(edge.dst)
            if edge.dst == trace_id:
                out.add(edge.src)
        return out

    def _make_candidates(self: Any, q: np.ndarray, c: np.ndarray, session_id: str = "") -> tuple[str, ...]:
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

    def _settle_workspace(
        self: Any,
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

    def _update_frontier(self: Any, workspace: Workspace, session_id: str) -> None:
        weights = dict(
            sorted(workspace.activation.items(), key=lambda item: item[1], reverse=True)[: self.config.frontier_size]
        )
        self.frontier_weights = weights
        if session_id:
            self.session_frontiers[session_id] = weights

    def _update_assoc_edges(self: Any, workspace: Workspace, *, ts: float) -> None:
        active_ids = list(workspace.active_trace_ids)
        for i, src in enumerate(active_ids):
            for dst in active_ids[i + 1 :]:
                weight = math.sqrt(max(workspace.activation[src] * workspace.activation[dst], 0.0))
                self._upsert_edge(src, dst, "assoc", delta_weight=0.10 * weight, ts=ts)
                self._upsert_edge(dst, src, "assoc", delta_weight=0.10 * weight, ts=ts)

    def _anchor_contexts(self: Any, anchor_ids: Sequence[str]) -> list[np.ndarray]:
        contexts: list[np.ndarray] = []
        wanted = set(anchor_ids)
        if not wanted:
            return contexts
        for frame in reversed(self.frames):
            if frame.anchor_id in wanted:
                contexts.append(np.asarray(frame.context, dtype=np.float64))
        return contexts

    def _trace_utility(self: Any, trace: TraceRecord) -> float:
        role_mass = trace_structural_role_mass(trace)
        forget = trace_forget_risk(trace, now_ts=self.current_ts)
        return float(trace.evidence + trace.stability + trace.access_ema + 0.5 * role_mass - trace.uncertainty - forget)
