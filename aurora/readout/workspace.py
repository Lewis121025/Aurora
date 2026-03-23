"""Workspace settling for the Aurora trace field."""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

from aurora.core.math import EPS, cosine_similarity, diag_gaussian_logpdf, entmax_bisect, project_top_k, weighted_mean
from aurora.core.config import FieldConfig
from aurora.core.types import PosteriorGroup, TraceEdge, TraceRecord, Workspace
from aurora.runtime.objective_terms import safe_kl, trace_structural_role_mass


def _renorm_operator(mat: np.ndarray, *, cap: float) -> np.ndarray:
    if mat.size == 0:
        return mat
    row_sums = np.maximum(np.sum(np.abs(mat), axis=1, keepdims=True), 1.0)
    out = np.asarray(mat / row_sums, dtype=np.float64)
    induced_norm = float(np.max(np.sum(np.abs(out), axis=1)))
    if induced_norm > cap > 0.0:
        out = out * (cap / induced_norm)
    return np.asarray(out, dtype=np.float64)


def _build_edge_matrices(
    candidate_ids: Sequence[str],
    edges: Mapping[tuple[str, str, str], TraceEdge],
    *,
    operator_cap: float,
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
    return _renorm_operator(w_pos, cap=operator_cap), _renorm_operator(w_neg, cap=operator_cap)


def _trace_prior(trace: TraceRecord) -> float:
    return math.log(trace.evidence + EPS) + 0.75 * trace.stability - 0.50 * trace.uncertainty


def _base_bias(
    trace: TraceRecord,
    cue: np.ndarray,
    frontier: np.ndarray,
    pred_mu: np.ndarray,
    context: np.ndarray,
    *,
    session_id: str,
    config: FieldConfig,
) -> float:
    prior_term = _trace_prior(trace)
    cue_term = cosine_similarity(cue, trace.z_mu)
    frontier_term = cosine_similarity(frontier, trace.z_mu)
    pred_term = cosine_similarity(pred_mu, trace.z_mu)
    ctx_term = diag_gaussian_logpdf(context, trace.ctx_mu, trace.ctx_sigma_diag) / max(trace.ctx_mu.size, 1)
    role_bonus = trace_structural_role_mass(trace)
    fidelity_bonus = trace.fidelity
    session_bonus = 0.25 if session_id and trace.metadata.get("session_id") == session_id else 0.0
    return float(
        config.workspace_bias_prior * prior_term
        + config.workspace_bias_cue * cue_term
        + config.workspace_bias_frontier * frontier_term
        + config.workspace_bias_predictor * pred_term
        + 0.20 * ctx_term
        + config.workspace_bias_role * role_bonus
        + config.workspace_bias_fidelity * fidelity_bonus
        + session_bonus
    )


def _sparse_project(logits: np.ndarray, *, config: FieldConfig, active_limit: int) -> np.ndarray:
    return project_top_k(
        entmax_bisect(logits / max(config.settle_temperature, 1e-4), alpha=config.entmax_alpha),
        active_limit,
    )


def _group_projection(
    a: np.ndarray,
    *,
    relevant_groups: Sequence[PosteriorGroup],
    idx: Mapping[str, int],
    cue: np.ndarray,
    context: np.ndarray,
    config: FieldConfig,
    posterior_slice_fn: Callable[[PosteriorGroup, np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[dict[str, object]], float]:
    out = np.asarray(a, dtype=np.float64).copy()
    payloads: list[dict[str, object]] = []
    total_group_kl = 0.0
    for group in relevant_groups:
        slice_probs = posterior_slice_fn(group, cue, context)
        member_probs = slice_probs[:-1]
        present_member_ids = [member_id for member_id in group.member_ids if member_id in idx]
        if not present_member_ids:
            continue
        member_indices = [idx[member_id] for member_id in present_member_ids]
        mass = float(np.sum(out[member_indices]))
        local_probs = np.asarray(
            [member_probs[group.member_ids.index(member_id)] for member_id in present_member_ids],
            dtype=np.float64,
        )
        total = float(np.sum(local_probs))
        if total > EPS and mass > EPS:
            local_probs = local_probs / total
            current = np.asarray([out[member_idx] for member_idx in member_indices], dtype=np.float64) / mass
            projected = (
                (1.0 - config.workspace_group_projection_eta) * current
                + config.workspace_group_projection_eta * local_probs
            )
            projected_total = float(np.sum(projected))
            if projected_total > EPS:
                projected = projected / projected_total
                for member_idx in member_indices:
                    out[member_idx] = 0.0
                for member_id, prob in zip(present_member_ids, projected, strict=False):
                    out[idx[member_id]] = mass * prob
                total_group_kl += mass * safe_kl(projected, local_probs)
        payloads.append(
            {
                "group_id": group.group_id,
                "trace_ids": tuple(present_member_ids),
                "weights": tuple(float(slice_probs[group.member_ids.index(member_id)]) for member_id in present_member_ids),
                "null_weight": float(slice_probs[-1]),
            }
        )
    return out, payloads, float(total_group_kl)


def _workspace_energy(
    a: np.ndarray,
    *,
    bias: np.ndarray,
    w_pos: np.ndarray,
    w_neg: np.ndarray,
    group_kl: float,
    config: FieldConfig,
) -> float:
    a = np.asarray(a, dtype=np.float64)
    linear = -float(np.dot(bias, a))
    positive = -0.5 * float(a @ w_pos @ a)
    negative = 0.5 * float(a @ w_neg @ a)
    l1 = config.workspace_l1 * float(np.sum(np.abs(a)))
    support = config.workspace_l0_proxy * float(np.count_nonzero(a > EPS))
    group = config.lambda_group_kl * float(group_kl)
    return float(linear + positive + negative + group + l1 + support)


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
            metadata={"session_id": session_id, "iterations": 0, "energy": 0.0, "energy_trace": [0.0]},
        )

    cue = np.asarray(cue, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    frontier = np.asarray(frontier, dtype=np.float64)
    pred_mu = np.asarray(pred_mu, dtype=np.float64)
    active_limit = int(workspace_size or config.workspace_size)

    bias = np.asarray(
        [
            _base_bias(traces[trace_id], cue, frontier, pred_mu, context, session_id=session_id, config=config)
            for trace_id in candidate_ids
        ],
        dtype=np.float64,
    )
    w_pos, w_neg = _build_edge_matrices(
        candidate_ids,
        edges,
        operator_cap=max(config.workspace_operator_norm_cap, 1e-3),
    )
    u = bias.copy()
    a = _sparse_project(u, config=config, active_limit=active_limit)

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

    a, group_payloads, group_kl_total = _group_projection(
        a,
        relevant_groups=relevant_groups,
        idx=idx,
        cue=cue,
        context=context,
        config=config,
        posterior_slice_fn=posterior_slice_fn,
    )
    total = float(np.sum(a))
    if total > EPS:
        a = a / total
    else:
        a[int(np.argmax(u))] = 1.0
    energy = _workspace_energy(a, bias=bias, w_pos=w_pos, w_neg=w_neg, group_kl=group_kl_total, config=config)
    energy_trace = [float(energy)]

    iterations = 0
    eta = float(config.settle_eta)
    for step in range(config.settle_steps):
        iterations = step + 1
        drive = bias + w_pos @ a - w_neg @ a
        accepted = False
        eta_try = eta
        best_state = (u, a, group_payloads, group_kl_total, energy)
        for _ in range(max(int(config.workspace_backtrack_steps), 1)):
            u_next = (1.0 - eta_try) * u + eta_try * drive
            a_next = _sparse_project(u_next, config=config, active_limit=active_limit)
            a_next, payloads_next, group_kl_next = _group_projection(
                a_next,
                relevant_groups=relevant_groups,
                idx=idx,
                cue=cue,
                context=context,
                config=config,
                posterior_slice_fn=posterior_slice_fn,
            )
            total = float(np.sum(a_next))
            if total <= EPS:
                a_next = np.zeros_like(a_next)
                a_next[int(np.argmax(u_next))] = 1.0
            else:
                a_next = a_next / total
            energy_next = _workspace_energy(
                a_next,
                bias=bias,
                w_pos=w_pos,
                w_neg=w_neg,
                group_kl=group_kl_next,
                config=config,
            )
            best_state = (u_next, a_next, payloads_next, group_kl_next, energy_next)
            if energy_next <= energy + 1e-8:
                accepted = True
                eta = eta_try
                break
            eta_try = max(eta_try * 0.5, config.workspace_min_eta)
        u_next, a_next, payloads_next, group_kl_next, energy_next = best_state
        if not accepted and eta_try <= config.workspace_min_eta:
            eta = eta_try
        if float(np.sum(np.abs(a_next - a))) < 1e-4:
            u = u_next
            a = a_next
            group_payloads = payloads_next
            group_kl_total = group_kl_next
            energy = energy_next
            energy_trace.append(float(energy))
            break
        u = u_next
        a = a_next
        group_payloads = payloads_next
        group_kl_total = group_kl_next
        energy = energy_next
        energy_trace.append(float(energy))

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
            "group_kl": float(group_kl_total),
            "energy": float(energy),
            "effective_eta": float(eta),
            "energy_trace": [float(value) for value in energy_trace],
            "last_cue": cue.tolist(),
            "last_context": context.tolist(),
        },
    )
