"""Closed-loop objective helpers for the Aurora trace field."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from aurora.core.math import EPS, diag_gaussian_logpdf, logsumexp, safe_variance, softmax, weighted_mean
from aurora.core.config import FieldConfig
from aurora.core.types import PosteriorGroup, TraceEdge, TraceRecord


@dataclass(slots=True)
class ObjectiveObservation:
    x: NDArray[np.float64]
    c: NDArray[np.float64]
    weight: float = 1.0
    target: NDArray[np.float64] | None = None
    target_sigma_diag: NDArray[np.float64] | None = None
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.c = np.asarray(self.c, dtype=np.float64)
        self.weight = float(max(self.weight, 0.0))
        if self.target is not None:
            self.target = np.asarray(self.target, dtype=np.float64)
        if self.target_sigma_diag is not None:
            self.target_sigma_diag = safe_variance(self.target_sigma_diag)


@dataclass(slots=True)
class ObjectiveTerms:
    surprise: float
    storage: float
    complexity: float
    plasticity: float = 0.0
    slow_alignment: float = 0.0
    drift: float = 0.0
    group_kl: float = 0.0
    fidelity: float = 0.0
    role: float = 0.0
    inhibit_bonus: float = 0.0
    local_energy: float = 0.0
    total: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        payload = {
            "surprise": float(self.surprise),
            "storage": float(self.storage),
            "complexity": float(self.complexity),
            "plasticity": float(self.plasticity),
            "slow_alignment": float(self.slow_alignment),
            "drift": float(self.drift),
            "group_kl": float(self.group_kl),
            "fidelity": float(self.fidelity),
            "role": float(self.role),
            "inhibit_bonus": float(self.inhibit_bonus),
            "local_energy": float(self.local_energy),
            "total": float(self.total),
        }
        payload.update({str(key): float(value) for key, value in self.extras.items()})
        return payload


@dataclass(slots=True)
class ActionEvaluation:
    action: str
    trace_id: str | None
    total: float
    terms: ObjectiveTerms
    apply_inhibit: bool = False
    inhibit_pair: tuple[str, str] | None = None


def gaussian_alignment_nll(x: np.ndarray, mean: np.ndarray, sigma_diag: np.ndarray) -> float:
    return float(-diag_gaussian_logpdf(x, mean, sigma_diag) / max(x.size, 1))


def safe_kl(
    p: Sequence[float] | NDArray[np.float64],
    q: Sequence[float] | NDArray[np.float64],
) -> float:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.size == 0 or q_arr.size == 0:
        return 0.0
    p_arr = np.clip(p_arr, EPS, 1.0)
    q_arr = np.clip(q_arr, EPS, 1.0)
    p_arr = p_arr / np.sum(p_arr)
    q_arr = q_arr / np.sum(q_arr)
    return float(np.sum(p_arr * (np.log(p_arr) - np.log(q_arr))))


def trace_log_prior(trace: TraceRecord, c: np.ndarray) -> float:
    base = math.log(trace.evidence + EPS) + 0.75 * trace.stability - 0.50 * trace.uncertainty
    ctx = diag_gaussian_logpdf(c, trace.ctx_mu, trace.ctx_sigma_diag)
    return float(base + ctx)


def null_logprob(x: np.ndarray, c: np.ndarray, config: FieldConfig) -> float:
    return float(
        diag_gaussian_logpdf(x, np.zeros_like(x), np.full_like(x, config.null_sigma))
        + diag_gaussian_logpdf(c, np.zeros_like(c), np.full_like(c, config.null_ctx_sigma))
    )


def mixture_summary(
    trace_ids: Sequence[str],
    traces: Mapping[str, TraceRecord],
    responsibilities: Mapping[str, float],
    shadow: Mapping[str, TraceRecord] | None = None,
    *,
    latent_dim: int,
) -> np.ndarray:
    shadow = shadow or {}
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for trace_id in trace_ids:
        if trace_id not in traces and trace_id not in shadow:
            continue
        weight = float(responsibilities.get(trace_id, 0.0))
        if weight <= 0.0:
            continue
        trace = shadow.get(trace_id, traces.get(trace_id))
        if trace is None:
            continue
        vectors.append(np.asarray(trace.z_mu, dtype=np.float64))
        weights.append(weight)
    for trace_id, trace in shadow.items():
        if trace_id in responsibilities or trace_id in trace_ids:
            continue
        vectors.append(np.asarray(trace.z_mu, dtype=np.float64))
        weights.append(0.5)
    if not vectors:
        return np.zeros(latent_dim, dtype=np.float64)
    return weighted_mean(vectors, weights)


def _effective_trace_ids(
    candidate_ids: Sequence[str],
    shadow: Mapping[str, TraceRecord] | None = None,
) -> tuple[str, ...]:
    shadow = shadow or {}
    local_ids: list[str] = []
    seen: set[str] = set()
    for trace_id in candidate_ids:
        if trace_id in seen:
            continue
        local_ids.append(trace_id)
        seen.add(trace_id)
    for trace_id in shadow:
        if trace_id in seen:
            continue
        local_ids.append(trace_id)
        seen.add(trace_id)
    return tuple(local_ids)


def shadow_responsibilities(
    candidate_ids: Sequence[str],
    traces: Mapping[str, TraceRecord],
    x: np.ndarray,
    c: np.ndarray,
    config: FieldConfig,
    *,
    shadow: Mapping[str, TraceRecord] | None = None,
) -> dict[str, float]:
    shadow = shadow or {}
    local_ids = _effective_trace_ids(candidate_ids, shadow)
    if not local_ids:
        return {}
    kept_ids: list[str] = []
    logits: list[float] = []
    for trace_id in local_ids:
        trace = shadow.get(trace_id, traces.get(trace_id))
        if trace is None:
            continue
        kept_ids.append(trace_id)
        logits.append(trace_log_prior(trace, c) + diag_gaussian_logpdf(x, trace.z_mu, trace.z_sigma_diag))
    if not logits:
        return {}
    probs = softmax(np.asarray(logits, dtype=np.float64))
    return {trace_id: float(prob) for trace_id, prob in zip(kept_ids, probs.tolist(), strict=False)}


def local_surprise(
    candidate_ids: Sequence[str],
    traces: Mapping[str, TraceRecord],
    x: np.ndarray,
    c: np.ndarray,
    config: FieldConfig,
    *,
    shadow: Mapping[str, TraceRecord] | None = None,
) -> float:
    shadow = shadow or {}
    scores: list[float] = []
    for trace_id in _effective_trace_ids(candidate_ids, shadow):
        trace = shadow.get(trace_id, traces.get(trace_id))
        if trace is None:
            continue
        scores.append(trace_log_prior(trace, c) + diag_gaussian_logpdf(x, trace.z_mu, trace.z_sigma_diag))
    scores.append(null_logprob(x, c, config))
    return float(-logsumexp(scores))


def trace_role_cost(trace: TraceRecord) -> float:
    prototype = max(trace.role_logits.get("prototype", 0.0), 0.0)
    procedure = max(trace.role_logits.get("procedure", 0.0), 0.0)
    support = trace.role_support.get("prototype", 0.0) + trace.role_support.get("procedure", 0.0)
    gain = max(trace.role_gain_ema.get("prototype", 0.0), 0.0) + max(trace.role_gain_ema.get("procedure", 0.0), 0.0)
    activation = prototype + procedure
    target = float(np.clip(support + gain, 0.0, 4.0))
    return float((activation - target) ** 2)


def trace_structural_role_mass(trace: TraceRecord) -> float:
    prototype = max(trace.role_logits.get("prototype", 0.0), 0.0)
    procedure = max(trace.role_logits.get("procedure", 0.0), 0.0)
    support = max(trace.role_support.get("prototype", 0.0), trace.role_support.get("procedure", 0.0))
    return float(max(prototype, procedure) + 0.25 * support)


def trace_fidelity_cost(trace: TraceRecord) -> float:
    return float(max(0.0, 1.0 - trace.fidelity) * (1.0 + 0.25 * trace.compression_stage))


def trace_storage_cost(trace: TraceRecord, config: FieldConfig) -> float:
    fidelity_factor = 0.35 + 0.65 * float(np.clip(trace.fidelity, 0.0, 1.0))
    role_factor = 1.0 + 0.05 * len(trace.member_ids) + 0.05 * len(trace.path_signature)
    return float(config.base_trace_cost * fidelity_factor * role_factor)


def edge_storage_cost(edge: TraceEdge, config: FieldConfig) -> float:
    kind_factor = {
        "assoc": 0.90,
        "temporal": 1.00,
        "inhib": 1.10,
        "option": 1.15,
    }[edge.kind]
    return float(config.base_edge_cost * kind_factor)


def group_storage_cost(group: PosteriorGroup, config: FieldConfig) -> float:
    member_factor = 1.0 + 0.08 * max(len(group.member_ids) - 1, 0)
    unresolved_factor = 1.0 + 0.15 * max(group.unresolved_mass, 0.0)
    replay_factor = 1.0 + 0.10 * max(group.replay_tension_ema, 0.0)
    buffer_factor = 1.0 + 0.02 * (len(group.ambiguous_buffer) + len(group.ambiguous_ctx_buffer))
    return float(0.25 * config.base_trace_cost * member_factor * unresolved_factor * replay_factor * buffer_factor)


def trace_future_alignment_cost(trace: TraceRecord) -> float:
    return float(max(trace.future_alignment_ema, 0.0))


def trace_future_drift_cost(trace: TraceRecord) -> float:
    return float(max(trace.future_drift_ema, 0.0))


def group_tension_cost(group: PosteriorGroup) -> float:
    return float(max(group.replay_tension_ema, 0.0) + 0.5 * max(group.unresolved_mass, 0.0))


def posterior_slice(
    group: PosteriorGroup,
    traces: Mapping[str, TraceRecord],
    x: np.ndarray,
    c: np.ndarray,
    config: FieldConfig,
    *,
    shadow: Mapping[str, TraceRecord] | None = None,
) -> np.ndarray:
    shadow = shadow or {}
    scores: list[float] = []
    for index, member_id in enumerate(group.member_ids):
        trace = shadow.get(member_id, traces.get(member_id))
        if trace is None:
            scores.append(float("-inf"))
            continue
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
        + diag_gaussian_logpdf(x, np.zeros_like(x), np.full_like(x, config.null_sigma))
    )
    return softmax(scores, temperature=max(group.temperature, 1e-4))


def group_regularization(
    candidate_ids: Sequence[str],
    traces: Mapping[str, TraceRecord],
    groups: Mapping[str, PosteriorGroup],
    x: np.ndarray,
    c: np.ndarray,
    responsibilities: Mapping[str, float],
    config: FieldConfig,
    *,
    shadow: Mapping[str, TraceRecord] | None = None,
) -> float:
    shadow = shadow or {}
    relevant_group_ids: set[str] = set()
    for trace_id in _effective_trace_ids(candidate_ids, shadow):
        trace = shadow.get(trace_id, traces.get(trace_id))
        if trace is None:
            continue
        relevant_group_ids.update(trace.posterior_group_ids)
    total_kl = 0.0
    for group_id in relevant_group_ids:
        group = groups.get(group_id)
        if group is None:
            continue
        present_member_ids = [member_id for member_id in group.member_ids if member_id in responsibilities]
        if len(present_member_ids) < 2:
            continue
        slice_probs = posterior_slice(group, traces, x, c, config, shadow=shadow)
        target = np.asarray(
            [slice_probs[group.member_ids.index(member_id)] for member_id in present_member_ids] + [slice_probs[-1]],
            dtype=np.float64,
        )
        current_member = np.asarray(
            [responsibilities.get(member_id, 0.0) for member_id in present_member_ids],
            dtype=np.float64,
        )
        current_null = max(1.0 - float(np.sum(current_member)), EPS)
        current = np.concatenate([current_member, np.asarray([current_null], dtype=np.float64)])
        total_kl += safe_kl(current, target) + group_tension_cost(group)
    return float(total_kl)


def weighted_future_alignment(
    trace_ids: Sequence[str],
    traces: Mapping[str, TraceRecord],
    responsibilities: Mapping[str, float],
    shadow: Mapping[str, TraceRecord] | None = None,
) -> float:
    shadow = shadow or {}
    values: list[float] = []
    weights: list[float] = []
    for trace_id in trace_ids:
        trace = shadow.get(trace_id, traces.get(trace_id))
        if trace is None:
            continue
        weight = float(max(responsibilities.get(trace_id, 0.0), 0.0))
        if weight <= 0.0:
            continue
        values.append(trace_future_alignment_cost(trace))
        weights.append(weight)
    if not values:
        return 0.0
    return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def weighted_future_drift(
    trace_ids: Sequence[str],
    traces: Mapping[str, TraceRecord],
    responsibilities: Mapping[str, float],
    shadow: Mapping[str, TraceRecord] | None = None,
) -> float:
    shadow = shadow or {}
    values: list[float] = []
    weights: list[float] = []
    for trace_id in trace_ids:
        trace = shadow.get(trace_id, traces.get(trace_id))
        if trace is None:
            continue
        weight = float(max(responsibilities.get(trace_id, 0.0), 0.0))
        if weight <= 0.0:
            continue
        values.append(trace_future_drift_cost(trace))
        weights.append(weight)
    if not values:
        return 0.0
    return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def empirical_patch_terms(
    *,
    config: FieldConfig,
    candidate_ids: Sequence[str],
    traces: Mapping[str, TraceRecord],
    edges: Mapping[tuple[str, str, str], TraceEdge],
    groups: Mapping[str, PosteriorGroup],
    observations: Sequence[ObjectiveObservation],
    shadow: Mapping[str, TraceRecord] | None = None,
    extra_trace_cost: float = 0.0,
    extra_edge_cost: float = 0.0,
    plasticity: float = 0.0,
    inhibit_bonus: float = 0.0,
    extra_drift: float = 0.0,
    extra_group_kl: float = 0.0,
) -> tuple[ObjectiveTerms, dict[str, float]]:
    shadow = shadow or {}
    local_ids = _effective_trace_ids(candidate_ids, shadow)
    if not observations:
        observations = ()

    total_weight = 0.0
    surprise_acc = 0.0
    group_kl_acc = 0.0
    immediate_alignment_acc = 0.0
    future_alignment_acc = 0.0
    future_drift_acc = 0.0
    resp_acc: dict[str, float] = {}
    target_count = 0.0

    for observation in observations:
        weight = float(max(observation.weight, EPS))
        total_weight += weight
        responsibilities = shadow_responsibilities(local_ids, traces, observation.x, observation.c, config, shadow=shadow)
        surprise_acc += weight * local_surprise(local_ids, traces, observation.x, observation.c, config, shadow=shadow)
        group_kl_acc += weight * group_regularization(
            local_ids,
            traces,
            groups,
            observation.x,
            observation.c,
            responsibilities,
            config,
            shadow=shadow,
        )
        for trace_id, prob in responsibilities.items():
            resp_acc[trace_id] = resp_acc.get(trace_id, 0.0) + weight * float(prob)
        if observation.target is not None:
            summary = mixture_summary(local_ids, traces, responsibilities, shadow, latent_dim=config.latent_dim)
            sigma = (
                np.asarray(observation.target_sigma_diag, dtype=np.float64)
                if observation.target_sigma_diag is not None
                else np.ones_like(observation.target, dtype=np.float64)
            )
            immediate_alignment_acc += weight * gaussian_alignment_nll(observation.target, summary, sigma)
            target_count += weight
        future_alignment_acc += weight * weighted_future_alignment(local_ids, traces, responsibilities, shadow)
        future_drift_acc += weight * weighted_future_drift(local_ids, traces, responsibilities, shadow)

    norm = max(total_weight, EPS)
    mean_resp = {trace_id: value / norm for trace_id, value in resp_acc.items()}

    storage = sum(
        trace_storage_cost(trace, config)
        for trace_id in local_ids
        if (trace := shadow.get(trace_id, traces.get(trace_id))) is not None
    )
    storage += sum(
        edge_storage_cost(edge, config)
        for edge in edges.values()
        if edge.src in local_ids or edge.dst in local_ids
    )
    relevant_group_ids: set[str] = set()
    for trace_id in local_ids:
        trace = shadow.get(trace_id, traces.get(trace_id))
        if trace is None:
            continue
        relevant_group_ids.update(trace.posterior_group_ids)
    storage += sum(group_storage_cost(groups[group_id], config) for group_id in relevant_group_ids if group_id in groups)
    storage += extra_trace_cost + extra_edge_cost

    local_edge_count = sum(1 for edge in edges.values() if edge.src in local_ids or edge.dst in local_ids)
    complexity = sum(
        math.log1p(sum(1 for edge in edges.values() if edge.src == trace_id or edge.dst == trace_id))
        for trace_id in local_ids
    )
    complexity += len(relevant_group_ids) + 0.05 * local_edge_count

    fidelity = sum(
        trace_fidelity_cost(trace)
        for trace_id in local_ids
        if (trace := shadow.get(trace_id, traces.get(trace_id))) is not None
    )
    role = sum(
        trace_role_cost(trace)
        for trace_id in local_ids
        if (trace := shadow.get(trace_id, traces.get(trace_id))) is not None
    )

    immediate_alignment = immediate_alignment_acc / max(target_count, 1.0)
    future_alignment = future_alignment_acc / norm
    future_drift = future_drift_acc / norm

    terms = objective_from_components(
        config=config,
        surprise=surprise_acc / norm,
        storage=storage,
        complexity=complexity,
        plasticity=plasticity,
        slow_alignment=immediate_alignment + config.continuation_weight * future_alignment,
        drift=extra_drift + config.continuation_weight * future_drift,
        group_kl=(group_kl_acc / norm) + float(extra_group_kl),
        fidelity=fidelity,
        role=role,
        inhibit_bonus=inhibit_bonus,
        extras={
            "observation_count": float(len(observations)),
            "candidate_count": float(len(local_ids)),
            "immediate_alignment": float(immediate_alignment),
            "future_alignment": float(future_alignment),
            "future_drift": float(future_drift),
            "group_count": float(len(relevant_group_ids)),
            "local_edge_count": float(local_edge_count),
        },
    )
    return terms, mean_resp


def combine_terms(config: FieldConfig, terms: ObjectiveTerms) -> ObjectiveTerms:
    local = (
        float(terms.surprise)
        + config.lambda_storage * float(terms.storage)
        + config.lambda_complexity * float(terms.complexity)
        + config.lambda_plasticity * float(terms.plasticity)
    )
    total = (
        local
        + config.lambda_predictor * float(terms.slow_alignment)
        + config.lambda_drift * float(terms.drift)
        + config.lambda_group_kl * float(terms.group_kl)
        + config.lambda_fidelity * float(terms.fidelity)
        + config.lambda_role * float(terms.role)
        + float(terms.inhibit_bonus)
    )
    terms.local_energy = float(local)
    terms.total = float(total)
    return terms


def objective_from_components(
    *,
    config: FieldConfig,
    surprise: float,
    storage: float,
    complexity: float,
    plasticity: float = 0.0,
    slow_alignment: float = 0.0,
    drift: float = 0.0,
    group_kl: float = 0.0,
    fidelity: float = 0.0,
    role: float = 0.0,
    inhibit_bonus: float = 0.0,
    extras: dict[str, float] | None = None,
) -> ObjectiveTerms:
    terms = ObjectiveTerms(
        surprise=float(surprise),
        storage=float(storage),
        complexity=float(complexity),
        plasticity=float(plasticity),
        slow_alignment=float(slow_alignment),
        drift=float(drift),
        group_kl=float(group_kl),
        fidelity=float(fidelity),
        role=float(role),
        inhibit_bonus=float(inhibit_bonus),
        extras=dict(extras or {}),
    )
    return combine_terms(config, terms)
