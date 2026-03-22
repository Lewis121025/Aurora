"""Canonical runtime types for the Aurora v2 trace field."""

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
class FieldConfig:
    data_dir: str = ".aurora"
    db_path: str = ".aurora/aurora.sqlite"
    blob_dir: str = ".aurora/blobs"
    max_snapshots: int = 256
    default_encoder_model: str = "intfloat/multilingual-e5-small"
    default_decoder_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_decoder_backend: str = "transformers"
    latent_dim: int = 128
    context_dim: int | None = None
    packet_chars: int = 512
    candidate_size: int = 64
    workspace_size: int = 12
    frontier_size: int = 16
    settle_steps: int = 8
    settle_eta: float = 0.35
    settle_temperature: float = 1.0
    entmax_alpha: float = 1.5
    maintenance_ms_budget: int = 20
    trace_budget: int = 256
    edge_budget: int = 2048
    anchor_budget: int = 1024
    evidence_decay: float = 0.995
    access_decay: float = 0.99
    eta_mu: float = 0.20
    eta_ctx: float = 0.10
    eta_sigma: float = 0.10
    eta_stability: float = 0.05
    lambda_stability: float = 4.0
    lambda_storage: float = 0.05
    lambda_complexity: float = 0.03
    lambda_plasticity: float = 0.10
    lambda_inhibit_gain: float = 0.30
    lambda_group_kl: float = 0.25
    gamma_stability: float = 1.0
    gamma_uncertainty: float = 0.7
    kappa_evidence: float = 0.3
    kappa_pred: float = 0.8
    base_trace_cost: float = 1.0
    base_edge_cost: float = 0.1
    init_sigma: float = 0.40
    init_ctx_sigma: float = 0.60
    null_sigma: float = 1.20
    null_ctx_sigma: float = 1.20
    sigma_floor: float = 1e-4
    sigma_ceiling: float = 16.0
    split_residual_threshold: float = 2.5
    attach_stability_threshold: float = 0.65
    inhibit_margin: float = 0.15
    inhibit_decay: float = 0.95
    inhibit_lr: float = 0.25
    group_bf_threshold: float = 0.2
    group_ctx_overlap_threshold: float = 0.55
    group_alpha_decay: float = 0.995
    group_ctx_lr: float = 0.10
    group_temp_lr: float = 0.05
    group_temp_min: float = 0.5
    group_temp_max: float = 2.0
    group_target_entropy: float = 0.75
    group_null_trigger: float = 0.35
    unresolved_buffer_maxlen: int = 16
    unresolved_compact_threshold: float = 0.35
    unresolved_context_cohesion_threshold: float = 0.5
    predictor_context_weight: float = 0.25
    frontier_context_weight: float = 0.35
    cue_context_weight: float = 1.0
    active_role_threshold: float = 0.75
    reservoir_size: int = 8

    def __post_init__(self) -> None:
        if self.context_dim is None:
            self.context_dim = self.latent_dim

    @property
    def packet_max_chars(self) -> int:
        return self.packet_chars


@dataclass(slots=True)
class ReplayConfig:
    batch_size: int = 32
    train_steps: int = 1
    trace_mix: float = 0.6
    conflict_mix: float = 0.2
    path_mix: float = 0.2
    priority_alpha_pred: float = 1.0
    priority_alpha_uncertainty: float = 0.8
    priority_alpha_group: float = 0.5
    priority_alpha_centrality: float = 0.4
    priority_alpha_forget: float = 0.4
    predictor_lr: float = 3e-3
    predictor_weight_decay: float = 1e-5
    theta_target_ema: float = 0.995
    reconsolidate_lr: float = 0.10
    edge_lr: float = 0.15
    prototype_support_threshold: float = 6.0
    prototype_dispersion_threshold: float = 0.30
    prototype_gain_threshold: float = 0.05
    procedure_support_threshold: float = 5.0
    procedure_success_threshold: float = 0.70
    procedure_entropy_threshold: float = 0.80
    procedure_gain_threshold: float = 0.05


@dataclass(slots=True)
class BudgetConfig:
    max_traces: int = 4096
    max_edges: int = 8192
    max_groups: int = 512
    pressure_prune_fraction: float = 0.10
    min_trace_utility: float = 0.05
    min_edge_utility: float = 0.02
    min_group_utility: float = 0.05


@dataclass(slots=True)
class PredictorConfig:
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    min_log_sigma: float = -4.0
    max_log_sigma: float = 2.0
    device: str = "cpu"


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
    pred_loss_ema: float = 0.0
    access_ema: float = 0.0
    last_access_ts: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    anchor_reservoir: Deque[str] = field(default_factory=deque)
    role_logits: dict[str, float] = field(
        default_factory=lambda: {"episodic": 1.0, "prototype": 0.0, "procedure": 0.0}
    )
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
            pred_loss_ema=float(self.pred_loss_ema),
            access_ema=float(self.access_ema),
            last_access_ts=float(self.last_access_ts),
            t_start=float(self.t_start),
            t_end=float(self.t_end),
            anchor_reservoir=deque(self.anchor_reservoir),
            role_logits=dict(self.role_logits),
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
    prototype_trace_ids: list[str] = field(default_factory=list)
    procedure_trace_ids: list[str] = field(default_factory=list)
    pruned_trace_ids: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    replay_batch: int = 0
    predictor_loss: float = 0.0
    predictor_nll: float = 0.0


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
