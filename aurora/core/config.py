"""Canonical runtime configuration types."""

from __future__ import annotations

from dataclasses import dataclass


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
    lambda_predictor: float = 0.18
    lambda_drift: float = 0.08
    lambda_fidelity: float = 0.05
    lambda_role: float = 0.03
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
    unresolved_spawn_min_anchors: int = 3
    group_uncertainty_inflate: float = 0.03
    predictor_context_weight: float = 0.25
    frontier_context_weight: float = 0.35
    cue_context_weight: float = 1.0
    active_role_threshold: float = 0.75
    reservoir_size: int = 8

    workspace_bias_prior: float = 1.0
    workspace_bias_cue: float = 1.0
    workspace_bias_frontier: float = 0.35
    workspace_bias_predictor: float = 0.25
    workspace_bias_role: float = 0.10
    workspace_bias_fidelity: float = 0.08
    workspace_group_projection_eta: float = 0.65
    workspace_operator_norm_cap: float = 0.95
    workspace_l1: float = 0.015
    workspace_l0_proxy: float = 0.02
    workspace_backtrack_steps: int = 4
    workspace_min_eta: float = 0.05

    continuation_decay: float = 0.97
    continuation_group_decay: float = 0.98
    continuation_weight: float = 0.50
    replay_mutation_min_delta: float = 0.08
    replay_mutation_max_ratio: float = 0.35
    maintenance_structural_passes: int = 1
    objective_local_window: int = 12
    objective_replay_window: int = 24

    fidelity_min: float = 0.15
    fidelity_compress_step: float = 0.20
    fidelity_expand_step: float = 0.15
    fidelity_sigma_inflate: float = 1.25
    fidelity_restore_activation: float = 0.18
    fidelity_prune_floor: float = 0.22
    fidelity_reservoir_keep_ratio: float = 0.50

    role_support_decay: float = 0.98
    role_gain_decay: float = 0.95
    role_logit_step: float = 0.20
    role_demote_pressure: float = 1.10
    prototype_demote_threshold: float = 1.00
    procedure_demote_threshold: float = 1.00

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


__all__ = ["BudgetConfig", "FieldConfig", "PredictorConfig", "ReplayConfig"]
