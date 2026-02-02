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
    
    Benchmark optimization notes:
    - AR: Needs precise retrieval, less smoothing
    - TTL: Needs fast learning, higher gate precision
    - LRU: Needs good aggregation, moderate KDE bandwidth
    - CR: Needs conflict detection, sensitive metric learning
    """

    # --- OnlineKDE parameters ---
    # Lower k_sigma = narrower bandwidth = sharper surprise peaks
    # Helps AR by making novel info more distinguishable
    kde_k_sigma: int = 20  # Reduced from 25 for sharper surprise detection

    # --- LowRankMetric parameters ---
    metric_init_noise: float = 0.01  # Initialization noise scale
    metric_window_size: int = 8000   # Reduced from 10000 for faster adaptation
    metric_decay_factor: float = 0.6  # Increased from 0.5 for slower forgetting

    # --- ThompsonBernoulliGate parameters ---
    # Higher initial precision = less exploration = faster convergence
    # Helps TTL by learning rules faster
    gate_init_precision: float = 5e-2  # Increased from 1e-2 for faster learning
    gate_forgetting_factor: float = 0.98  # Reduced from 0.99 for more plasticity
    gate_rms_decay: float = 0.95  # Reduced from 0.99 for faster adaptation

    # --- StoryModel parameters ---
    # Lower beta = less smoothing = sharper actor likelihood
    # Helps AR by distinguishing conversations with different actors
    story_dirichlet_beta: float = 0.8  # Reduced from 1.0

    # --- FieldRetriever parameters ---
    # Optimized for benchmark: balance precision (AR) with coverage (LRU)
    retrieval_damping: float = 0.80  # Reduced from 0.85 for better direct matches
    retrieval_mean_shift_steps: int = 6  # Reduced from 8 for less smoothing
    retrieval_initial_seed_k: int = 60  # Increased from 50 for better recall
    retrieval_reseed_k: int = 50  # Reduced from 60 for precision
    retrieval_pagerank_max_iter: int = 40  # Reduced from 50, converges faster

    # --- Sliding window ---
    # Larger window helps LRU by considering more context
    recent_plots_window: int = 250  # Increased from 200

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
    # Benchmark optimization:
    # - Lower story_alpha = fewer stories = better LRU aggregation
    # - Higher theme_alpha = more themes = better identity dimension capture
    story_alpha: float = 0.7  # Reduced from 1.0: prefer fewer, richer stories for LRU
    theme_alpha: float = 0.8  # Increased from 0.5: capture more identity patterns

    # Encode gate feature dimension
    gate_feature_dim: int = 6  # Number of features for encoding gate

    # Retrieval preferences
    retrieval_kinds: Tuple[str, ...] = ("theme", "story", "plot")

    # Vector index: "auto" (FAISS if available), "faiss", "brute"
    vector_backend: str = "auto"
    
    # FAISS HNSW parameters (only used when vector_backend is "faiss" or "auto")
    faiss_m: int = 32  # HNSW connections per layer (8-64, higher = better recall)
    faiss_ef_construction: int = 64  # Construction-time search depth
    faiss_ef_search: int = 32  # Query-time search depth (increase for better recall)

    # Benchmark mode: force store all plots, bypass VOI gating
    # Essential for benchmarks like LongMemEval where every turn may contain critical information
    benchmark_mode: bool = False

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
            "vector_backend": self.vector_backend,
            "faiss_m": self.faiss_m,
            "faiss_ef_construction": self.faiss_ef_construction,
            "faiss_ef_search": self.faiss_ef_search,
            "benchmark_mode": self.benchmark_mode,
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
            vector_backend=d.get("vector_backend", "auto"),
            faiss_m=d.get("faiss_m", 32),
            faiss_ef_construction=d.get("faiss_ef_construction", 64),
            faiss_ef_search=d.get("faiss_ef_search", 32),
            benchmark_mode=d.get("benchmark_mode", False),
            algo=algo,
        )
