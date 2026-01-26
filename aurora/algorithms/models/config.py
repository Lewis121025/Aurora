"""
AURORA Configuration Models
============================

Configuration dataclasses for the AURORA memory system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AlgorithmConfig:
    """
    Algorithm hyperparameters extracted from hard-coded values.

    These parameters control the behavior of learnable components
    and can be tuned for different use cases.
    """

    # --- OnlineKDE parameters ---
    kde_k_sigma: int = 25  # k-nearest neighbors for bandwidth estimation

    # --- LowRankMetric parameters ---
    metric_init_noise: float = 0.01  # Initialization noise scale
    metric_window_size: int = 10000  # Adagrad sliding window size
    metric_decay_factor: float = 0.5  # Decay factor on window reset

    # --- ThompsonBernoulliGate parameters ---
    gate_init_precision: float = 1e-2  # Initial precision for weights
    gate_forgetting_factor: float = 0.99  # Forgetting factor for precision
    gate_rms_decay: float = 0.99  # RMSprop decay for adaptive learning

    # --- StoryModel parameters ---
    story_dirichlet_beta: float = 1.0  # Dirichlet-multinomial smoothing

    # --- FieldRetriever parameters ---
    retrieval_damping: float = 0.85  # PageRank damping factor
    retrieval_mean_shift_steps: int = 8  # Mean-shift iteration count
    retrieval_initial_seed_k: int = 50  # Initial seed vectors for search
    retrieval_reseed_k: int = 60  # Reseed around attractor
    retrieval_pagerank_max_iter: int = 50  # PageRank maximum iterations

    # --- Sliding window ---
    recent_plots_window: int = 200  # Recent plots window size

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "kde_k_sigma": self.kde_k_sigma,
            "metric_init_noise": self.metric_init_noise,
            "metric_window_size": self.metric_window_size,
            "metric_decay_factor": self.metric_decay_factor,
            "gate_init_precision": self.gate_init_precision,
            "gate_forgetting_factor": self.gate_forgetting_factor,
            "gate_rms_decay": self.gate_rms_decay,
            "story_dirichlet_beta": self.story_dirichlet_beta,
            "retrieval_damping": self.retrieval_damping,
            "retrieval_mean_shift_steps": self.retrieval_mean_shift_steps,
            "retrieval_initial_seed_k": self.retrieval_initial_seed_k,
            "retrieval_reseed_k": self.retrieval_reseed_k,
            "retrieval_pagerank_max_iter": self.retrieval_pagerank_max_iter,
            "recent_plots_window": self.recent_plots_window,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "AlgorithmConfig":
        """Reconstruct from state dict."""
        return cls(
            kde_k_sigma=d.get("kde_k_sigma", 25),
            metric_init_noise=d.get("metric_init_noise", 0.01),
            metric_window_size=d.get("metric_window_size", 10000),
            metric_decay_factor=d.get("metric_decay_factor", 0.5),
            gate_init_precision=d.get("gate_init_precision", 1e-2),
            gate_forgetting_factor=d.get("gate_forgetting_factor", 0.99),
            gate_rms_decay=d.get("gate_rms_decay", 0.99),
            story_dirichlet_beta=d.get("story_dirichlet_beta", 1.0),
            retrieval_damping=d.get("retrieval_damping", 0.85),
            retrieval_mean_shift_steps=d.get("retrieval_mean_shift_steps", 8),
            retrieval_initial_seed_k=d.get("retrieval_initial_seed_k", 50),
            retrieval_reseed_k=d.get("retrieval_reseed_k", 60),
            retrieval_pagerank_max_iter=d.get("retrieval_pagerank_max_iter", 50),
            recent_plots_window=d.get("recent_plots_window", 200),
        )


@dataclass
class MemoryConfig:
    """
    Core memory configuration.

    These parameters control the structure and capacity of the memory system.
    """

    dim: int = 384  # Embedding dimension
    metric_rank: int = 64  # Low-rank metric dimension

    # Bounded memory pressures (resource constraints are first-principles reality)
    max_plots: int = 5000  # Maximum number of plots to retain
    kde_reservoir: int = 4096  # KDE reservoir sampling size

    # CRP concentration priors
    story_alpha: float = 1.0  # Story CRP concentration
    theme_alpha: float = 0.5  # Theme CRP concentration

    # Encode gate feature dimension
    gate_feature_dim: int = 6  # Number of features for encoding gate

    # Retrieval preferences
    retrieval_kinds: Tuple[str, ...] = ("theme", "story", "plot")

    # Algorithm config (nested, optional for backward compatibility)
    algo: Optional[AlgorithmConfig] = field(default=None)

    def __post_init__(self):
        if self.algo is None:
            object.__setattr__(self, 'algo', AlgorithmConfig())

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "dim": self.dim,
            "metric_rank": self.metric_rank,
            "max_plots": self.max_plots,
            "kde_reservoir": self.kde_reservoir,
            "story_alpha": self.story_alpha,
            "theme_alpha": self.theme_alpha,
            "gate_feature_dim": self.gate_feature_dim,
            "retrieval_kinds": list(self.retrieval_kinds),
            "algo": self.algo.to_state_dict() if self.algo else None,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "MemoryConfig":
        """Reconstruct from state dict."""
        algo_dict = d.get("algo")
        algo = AlgorithmConfig.from_state_dict(algo_dict) if algo_dict else AlgorithmConfig()
        return cls(
            dim=d.get("dim", 384),
            metric_rank=d.get("metric_rank", 64),
            max_plots=d.get("max_plots", 5000),
            kde_reservoir=d.get("kde_reservoir", 4096),
            story_alpha=d.get("story_alpha", 1.0),
            theme_alpha=d.get("theme_alpha", 0.5),
            gate_feature_dim=d.get("gate_feature_dim", 6),
            retrieval_kinds=tuple(d.get("retrieval_kinds", ["theme", "story", "plot"])),
            algo=algo,
        )
