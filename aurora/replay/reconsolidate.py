"""Reconsolidation helpers for the Aurora v2 kernel."""

from __future__ import annotations

import math

import numpy as np

from aurora.core.math import EPS, safe_variance
from aurora.core.config import FieldConfig
from aurora.core.types import TraceRecord
from aurora.runtime.objective_terms import trace_structural_role_mass


def compute_uncertainty(trace: TraceRecord, config: FieldConfig) -> float:
    dispersion = float(
        np.mean(safe_variance(trace.z_sigma_diag, min_value=config.sigma_floor, max_value=config.sigma_ceiling))
    )
    lack_of_evidence = float(config.kappa_evidence / (trace.evidence + EPS))
    predictive_instability = float(config.kappa_pred * trace.pred_loss_ema)
    return dispersion + lack_of_evidence + predictive_instability


def update_stability(trace: TraceRecord, config: FieldConfig) -> float:
    logit = math.log(trace.evidence + EPS) - trace.uncertainty - trace.pred_loss_ema
    target = 1.0 / (1.0 + math.exp(-logit))
    trace.stability = (1.0 - config.eta_stability) * trace.stability + config.eta_stability * target
    trace.stability = float(np.clip(trace.stability, 0.0, 1.0))
    return trace.stability


def push_anchor_reference(trace: TraceRecord, anchor_id: str, *, reservoir_size: int) -> None:
    if anchor_id in trace.anchor_reservoir:
        trace.anchor_reservoir.remove(anchor_id)
    trace.anchor_reservoir.append(anchor_id)
    while len(trace.anchor_reservoir) > reservoir_size:
        trace.anchor_reservoir.popleft()


def reconsolidate_trace(
    trace: TraceRecord,
    x: np.ndarray,
    c: np.ndarray,
    *,
    config: FieldConfig,
    responsibility: float = 1.0,
    pred_loss: float = 0.0,
    ts: float | None = None,
    anchor_id: str | None = None,
) -> TraceRecord:
    r = float(np.clip(responsibility, 0.0, 1.0))
    if r <= EPS:
        return trace

    trace.evidence = config.evidence_decay * trace.evidence + r

    eta_mu = config.eta_mu * r / (1.0 + config.lambda_stability * trace.stability)
    old_mu = trace.z_mu.copy()
    trace.z_mu = trace.z_mu + eta_mu * (x - trace.z_mu)
    trace.z_sigma_diag = safe_variance(
        (1.0 - config.eta_sigma * r) * trace.z_sigma_diag + config.eta_sigma * r * np.square(x - old_mu),
        min_value=config.sigma_floor,
        max_value=config.sigma_ceiling,
    )

    trace.ctx_mu = trace.ctx_mu + config.eta_ctx * r * (c - trace.ctx_mu)
    trace.ctx_sigma_diag = safe_variance(
        (1.0 - config.eta_sigma * r) * trace.ctx_sigma_diag + config.eta_sigma * r * np.square(c - trace.ctx_mu),
        min_value=config.sigma_floor,
        max_value=config.sigma_ceiling,
    )

    trace.pred_loss_ema = 0.95 * trace.pred_loss_ema + 0.05 * float(max(pred_loss, 0.0))
    trace.uncertainty = compute_uncertainty(trace, config)
    update_stability(trace, config)

    if ts is not None:
        trace.t_end = max(trace.t_end, float(ts))
        trace.last_access_ts = float(ts)
    if anchor_id is not None:
        push_anchor_reference(trace, anchor_id, reservoir_size=config.reservoir_size)
    return trace


def trace_forget_risk(trace: TraceRecord, now_ts: float) -> float:
    age = max(float(now_ts - trace.last_access_ts), 0.0)
    return float((1.0 + age) * (trace.uncertainty + 1.0) / (trace.evidence + 1.0))


def trace_utility(trace: TraceRecord, now_ts: float) -> float:
    age = max(float(now_ts - trace.last_access_ts), 0.0)
    recency = 1.0 / (1.0 + age)
    role_bonus = trace_structural_role_mass(trace)
    return float(
        0.45 * trace.evidence
        + 0.25 * trace.access_ema
        + 0.20 * trace.stability
        + 0.10 * role_bonus
        + recency
        - trace.uncertainty
    )
