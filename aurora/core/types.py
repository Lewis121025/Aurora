"""Canonical runtime types for the Aurora v2.1 closed-loop trace field."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Literal

import numpy as np
from numpy.typing import NDArray

from aurora.core.math import Array, safe_variance

PayloadType = Literal["text", "tool_call", "tool_result", "state_delta", "reward", "observation"]
SourceType = Literal["user", "assistant", "tool", "env"]
RoleName = Literal["episodic", "prototype", "procedure"]
EdgeKind = Literal["temporal", "assoc", "inhib", "option"]
ActionName = Literal["ASSIMILATE", "ATTACH", "SPLIT", "BIRTH", "INHIBIT"]


@dataclass(slots=True)
class Anchor:
    anchor_id: str
    packet_id: str
    session_id: str
    turn_id: str
    source: SourceType
    z: Array
    z_hv: NDArray[np.int64] | None = None
    ts: float = 0.0
    residual_ref: str | None = None
    source_quality: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=np.float64)
        if self.z_hv is not None:
            self.z_hv = np.asarray(self.z_hv, dtype=np.int64)


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    z_mu: Array
    z_sigma_diag: Array
    ctx_mu: Array
    ctx_sigma_diag: Array
    evidence: float = 1.0
    stability: float = 0.0
    uncertainty: float = 1.0
    fidelity: float = 1.0
    compression_stage: int = 0
    pred_loss_ema: float = 0.0
    future_alignment_ema: float = 0.0
    future_drift_ema: float = 0.0
    access_ema: float = 0.0
    last_access_ts: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    anchor_reservoir: Deque[str] = field(default_factory=deque)
    role_logits: dict[str, float] = field(
        default_factory=lambda: {"episodic": 1.0, "prototype": 0.0, "procedure": 0.0}
    )
    role_support: dict[str, float] = field(default_factory=lambda: {"prototype": 0.0, "procedure": 0.0})
    role_gain_ema: dict[str, float] = field(default_factory=lambda: {"prototype": 0.0, "procedure": 0.0})
    member_ids: tuple[str, ...] = field(default_factory=tuple)
    path_signature: tuple[str, ...] = field(default_factory=tuple)
    posterior_group_ids: tuple[str, ...] = field(default_factory=tuple)
    parent_trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z_mu = np.asarray(self.z_mu, dtype=np.float64)
        self.z_sigma_diag = safe_variance(self.z_sigma_diag)
        self.ctx_mu = np.asarray(self.ctx_mu, dtype=np.float64)
        self.ctx_sigma_diag = safe_variance(self.ctx_sigma_diag)
        if not isinstance(self.anchor_reservoir, deque):
            self.anchor_reservoir = deque(self.anchor_reservoir)

    def clone(self) -> "TraceRecord":
        return TraceRecord(
            trace_id=self.trace_id,
            z_mu=self.z_mu.copy(),
            z_sigma_diag=self.z_sigma_diag.copy(),
            ctx_mu=self.ctx_mu.copy(),
            ctx_sigma_diag=self.ctx_sigma_diag.copy(),
            evidence=float(self.evidence),
            stability=float(self.stability),
            uncertainty=float(self.uncertainty),
            fidelity=float(self.fidelity),
            compression_stage=int(self.compression_stage),
            pred_loss_ema=float(self.pred_loss_ema),
            future_alignment_ema=float(self.future_alignment_ema),
            future_drift_ema=float(self.future_drift_ema),
            access_ema=float(self.access_ema),
            last_access_ts=float(self.last_access_ts),
            t_start=float(self.t_start),
            t_end=float(self.t_end),
            anchor_reservoir=deque(self.anchor_reservoir),
            role_logits=dict(self.role_logits),
            role_support=dict(self.role_support),
            role_gain_ema=dict(self.role_gain_ema),
            member_ids=tuple(self.member_ids),
            path_signature=tuple(self.path_signature),
            posterior_group_ids=tuple(self.posterior_group_ids),
            parent_trace_id=self.parent_trace_id,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class TraceEdge:
    src: str
    dst: str
    kind: EdgeKind
    weight: float
    support_ema: float = 0.0
    last_update_ts: float = 0.0
    bf_sep_ema: float = 0.0
    success_alpha: float = 1.0
    success_beta: float = 1.0

    @property
    def key(self) -> tuple[str, str, EdgeKind]:
        return (self.src, self.dst, self.kind)


@dataclass(slots=True)
class PosteriorGroup:
    group_id: str
    member_ids: list[str]
    alpha: Array
    ctx_mu: Array
    ctx_sigma_diag: Array
    pred_success_ema: Array
    temperature: float = 1.0
    unresolved_mass: float = 0.0
    replay_alignment_ema: float = 0.0
    replay_tension_ema: float = 0.0
    ambiguous_buffer: Deque[str] = field(default_factory=deque)
    ambiguous_ctx_buffer: Deque[Array] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.alpha = np.asarray(self.alpha, dtype=np.float64)
        self.ctx_mu = np.asarray(self.ctx_mu, dtype=np.float64)
        self.ctx_sigma_diag = safe_variance(self.ctx_sigma_diag)
        self.pred_success_ema = np.asarray(self.pred_success_ema, dtype=np.float64)
        if not isinstance(self.ambiguous_buffer, deque):
            self.ambiguous_buffer = deque(self.ambiguous_buffer)
        if not isinstance(self.ambiguous_ctx_buffer, deque):
            self.ambiguous_ctx_buffer = deque(self.ambiguous_ctx_buffer)


@dataclass(slots=True)
class PredictorState:
    h: Array
    theta: dict[str, Array]
    theta_target: dict[str, Array]


@dataclass(slots=True)
class PredictorPeek:
    mu: Array
    sigma_diag: Array
    h: Array


@dataclass(slots=True)
class ProposalDecision:
    action: ActionName
    trace_id: str | None
    delta_energy: float
    candidate_ids: tuple[str, ...]
    context: Array
    base_energy: float
    top_responsibilities: dict[str, float] = field(default_factory=dict)
    apply_inhibit: bool = False
    inhibit_pair: tuple[str, str] | None = None
    objective_terms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExperienceFrame:
    frame_id: str
    anchor_id: str
    trace_id: str
    session_id: str
    turn_id: str
    action: ActionName
    x: Array
    context: Array
    workspace_vec: Array
    frontier_vec: Array
    action_vec: Array
    ts: float
    active_ids: tuple[str, ...]
    activation: dict[str, float] = field(default_factory=dict)
    group_entropy: float = 0.0
    next_x: Array | None = None
    next_trace_id: str | None = None
    next_ts: float | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.context = np.asarray(self.context, dtype=np.float64)
        self.workspace_vec = np.asarray(self.workspace_vec, dtype=np.float64)
        self.frontier_vec = np.asarray(self.frontier_vec, dtype=np.float64)
        self.action_vec = np.asarray(self.action_vec, dtype=np.float64)
        if self.next_x is not None:
            self.next_x = np.asarray(self.next_x, dtype=np.float64)


@dataclass(slots=True)
class Workspace:
    active_trace_ids: tuple[str, ...]
    weights: tuple[float, ...]
    activation: dict[str, float]
    posterior_groups: tuple[dict[str, Any], ...]
    active_procedure_ids: tuple[str, ...]
    anchor_refs: tuple[str, ...]
    summary_vector: Array
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.summary_vector = np.asarray(self.summary_vector, dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecoderRequest:
    cue: str
    workspace: Workspace
    active_trace_ids: tuple[str, ...]
    anchor_refs: tuple[str, ...]
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecoderOutput:
    text: str
    token_count: int
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InjectResult:
    packet_ids: list[str]
    anchor_ids: list[str]
    trace_ids: list[str]
    proposal_kinds: list[str] = field(default_factory=list)
    touched_trace_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MaintenanceStats:
    replayed_trace_ids: list[str] = field(default_factory=list)
    structural_trace_ids: list[str] = field(default_factory=list)
    prototype_trace_ids: list[str] = field(default_factory=list)
    procedure_trace_ids: list[str] = field(default_factory=list)
    pruned_trace_ids: list[str] = field(default_factory=list)
    compressed_trace_ids: list[str] = field(default_factory=list)
    rehydrated_trace_ids: list[str] = field(default_factory=list)
    demoted_trace_ids: list[str] = field(default_factory=list)
    replayed_group_ids: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    replay_batch: int = 0
    predictor_loss: float = 0.0
    predictor_nll: float = 0.0
    objective_total: float = 0.0


@dataclass(frozen=True, slots=True)
class ResponseResult:
    response_text: str
    workspace: Workspace
    trace_ids: list[str] = field(default_factory=list)
    anchor_ids: list[str] = field(default_factory=list)
    response_vector: Array = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    snapshot_path: str
    trace_count: int
    anchor_count: int
    edge_count: int


__all__ = [
    "ActionName",
    "Anchor",
    "DecoderOutput",
    "DecoderRequest",
    "EdgeKind",
    "ExperienceFrame",
    "InjectResult",
    "MaintenanceStats",
    "PayloadType",
    "PosteriorGroup",
    "PredictorPeek",
    "PredictorState",
    "ProposalDecision",
    "ResponseResult",
    "RoleName",
    "SnapshotMeta",
    "SourceType",
    "TraceEdge",
    "TraceRecord",
    "Workspace",
]
