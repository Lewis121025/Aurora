"""Workspace settling for the Aurora v2 trace field."""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

from aurora.core.math import EPS, cosine_similarity, diag_gaussian_logpdf, entmax_bisect, project_top_k, weighted_mean
from aurora.core.types import FieldConfig, PosteriorGroup, TraceEdge, TraceRecord, Workspace


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    row_sums = np.maximum(np.sum(np.abs(mat), axis=1, keepdims=True), 1.0)
    return np.asarray(mat / row_sums, dtype=np.float64)


def _build_edge_matrices(
    candidate_ids: Sequence[str],
    edges: Mapping[tuple[str, str, str], TraceEdge],
) -> tuple[np.ndarray, np.ndarray]:
    index = {trace_id: idx for idx, trace_id in enumerate(candidate_ids)}
    size = len(candidate_ids)
    w_pos = np.zeros((size, size), dtype=np.float64)
    w_neg = np.zeros((size, size), dtype=np.float64)
    for edge in edges.values():
        if edge.src not in index or edge.dst not in index:
            continue
        src_idx = index[edge.src]
        dst_idx = index[edge.dst]
        weight = max(float(edge.weight), 0.0)
        if edge.kind == "inhib":
            w_neg[dst_idx, src_idx] += weight
            w_neg[src_idx, dst_idx] += weight
            continue
        w_pos[dst_idx, src_idx] += weight
        if edge.kind == "assoc":
            w_pos[src_idx, dst_idx] += weight
    return _normalize_matrix(w_pos), _normalize_matrix(w_neg)


def _base_bias(
    trace: TraceRecord,
    cue: np.ndarray,
    frontier: np.ndarray,
    pred_mu: np.ndarray,
    context: np.ndarray,
    *,
    session_id: str,
) -> float:
    evidence_prior = math.log(trace.evidence + EPS) + 0.75 * trace.stability - 0.50 * trace.uncertainty
    cue_term = cosine_similarity(cue, trace.z_mu)
    frontier_term = cosine_similarity(frontier, trace.z_mu)
    pred_term = cosine_similarity(pred_mu, trace.z_mu)
    ctx_term = diag_gaussian_logpdf(context, trace.ctx_mu, trace.ctx_sigma_diag) / max(trace.ctx_mu.size, 1)
    role_bonus = max(trace.role_logits.values())
    session_bonus = 0.25 if session_id and trace.metadata.get("session_id") == session_id else 0.0
    return float(
        evidence_prior
        + cue_term
        + 0.35 * frontier_term
        + 0.25 * pred_term
        + 0.20 * ctx_term
        + 0.10 * role_bonus
        + session_bonus
    )


def settle_workspace(
    *,
    candidate_ids: Sequence[str],
    traces: Mapping[str, TraceRecord],
    edges: Mapping[tuple[str, str, str], TraceEdge],
    groups: Mapping[str, PosteriorGroup],
    cue: np.ndarray,
    context: np.ndarray,
    frontier: np.ndarray,
    pred_mu: np.ndarray,
    config: FieldConfig,
    posterior_slice_fn: Callable[[PosteriorGroup, np.ndarray, np.ndarray], np.ndarray],
    session_id: str = "",
    workspace_size: int | None = None,
) -> Workspace:
    candidate_ids = tuple(candidate_ids)
    if not candidate_ids:
        return Workspace(
            active_trace_ids=tuple(),
            weights=tuple(),
            activation={},
            posterior_groups=tuple(),
            active_procedure_ids=tuple(),
            anchor_refs=tuple(),
            summary_vector=np.zeros(config.latent_dim, dtype=np.float64),
            metadata={"session_id": session_id, "iterations": 0},
        )

    cue = np.asarray(cue, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    frontier = np.asarray(frontier, dtype=np.float64)
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    active_limit = int(workspace_size or config.workspace_size)

    bias = np.asarray(
        [_base_bias(traces[trace_id], cue, frontier, pred_mu, context, session_id=session_id) for trace_id in candidate_ids],
        dtype=np.float64,
    )
    w_pos, w_neg = _build_edge_matrices(candidate_ids, edges)
    u = bias.copy()
    a = project_top_k(
        entmax_bisect(u / max(config.settle_temperature, 1e-4), alpha=config.entmax_alpha),
        active_limit,
    )

    idx = {trace_id: i for i, trace_id in enumerate(candidate_ids)}
    relevant_groups: list[PosteriorGroup] = []
    seen = set()
    for trace_id in candidate_ids:
        for group_id in traces[trace_id].posterior_group_ids:
            if group_id in groups and group_id not in seen:
                group = groups[group_id]
                overlap = sum(1 for member_id in group.member_ids if member_id in idx)
                if overlap >= 2:
                    relevant_groups.append(group)
                    seen.add(group_id)

    iterations = 0
    for step in range(config.settle_steps):
        iterations = step + 1
        u_next = (1.0 - config.settle_eta) * u + config.settle_eta * (bias + w_pos @ a - w_neg @ a)
        a_next = project_top_k(
            entmax_bisect(u_next / max(config.settle_temperature, 1e-4), alpha=config.entmax_alpha),
            active_limit,
        )
        group_payloads: list[dict[str, object]] = []
        for group in relevant_groups:
            slice_probs = posterior_slice_fn(group, cue, context)
            member_probs = slice_probs[:-1]
            present_member_ids = [member_id for member_id in group.member_ids if member_id in idx]
            if not present_member_ids:
                continue
            local_probs = np.asarray(
                [member_probs[group.member_ids.index(member_id)] for member_id in present_member_ids],
                dtype=np.float64,
            )
            total = float(np.sum(local_probs))
            if total > EPS:
                local_probs = local_probs / total
                member_indices = [idx[member_id] for member_id in present_member_ids]
                mass = float(np.sum(a_next[member_indices]))
                if mass > EPS:
                    for member_idx in member_indices:
                        a_next[member_idx] = 0.0
                    for member_id, prob in zip(present_member_ids, local_probs, strict=False):
                        a_next[idx[member_id]] = mass * prob
            group_payloads.append(
                {
                    "group_id": group.group_id,
                    "trace_ids": tuple(present_member_ids),
                    "weights": tuple(float(slice_probs[group.member_ids.index(member_id)]) for member_id in present_member_ids),
                    "null_weight": float(slice_probs[-1]),
                }
            )
        total = float(np.sum(a_next))
        if total <= EPS:
            a_next = np.zeros_like(a_next)
            a_next[int(np.argmax(u_next))] = 1.0
        else:
            a_next = a_next / total
        if float(np.sum(np.abs(a_next - a))) < 1e-4:
            u = u_next
            a = a_next
            break
        u = u_next
        a = a_next

    active_idx = np.flatnonzero(a > 0.0)
    active_ids = tuple(candidate_ids[i] for i in active_idx.tolist())
    activation = {candidate_ids[i]: float(a[i]) for i in active_idx.tolist()}
    weights = tuple(float(activation[trace_id]) for trace_id in active_ids)
    if active_ids:
        summary = weighted_mean([traces[trace_id].z_mu for trace_id in active_ids], weights)
    else:
        summary = np.zeros(config.latent_dim, dtype=np.float64)
    procedure_ids = tuple(
        trace_id
        for trace_id in active_ids
        if traces[trace_id].role_logits.get("procedure", 0.0) >= config.active_role_threshold
    )
    anchor_refs = tuple(
        anchor_id
        for trace_id in active_ids
        for anchor_id in list(traces[trace_id].anchor_reservoir)[-config.reservoir_size :]
    )
    return Workspace(
        active_trace_ids=active_ids,
        weights=weights,
        activation=activation,
        posterior_groups=tuple(group_payloads),
        active_procedure_ids=procedure_ids,
        anchor_refs=anchor_refs,
        summary_vector=summary,
        metadata={
            "session_id": session_id,
            "iterations": iterations,
            "last_cue": cue.tolist(),
            "last_context": context.tolist(),
        },
    )
