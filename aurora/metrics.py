"""
Aurora Prometheus Metrics
=========================

Provides white-box monitoring of memory algorithm behavior:
- Narrative tension (user engagement/interest)
- Cluster entropy (story/theme diversity)
- Gate pass rate (memory encoding selectivity)
- Performance metrics (latency, throughput)

Usage:
    from aurora.metrics import (
        aurora_tension_avg,
        aurora_cluster_entropy,
        aurora_gate_pass_rate,
    )
    
    # Update metrics in algorithm code
    aurora_tension_avg.labels(user_id="user_123").set(0.65)

For FastAPI integration:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional

# Lazy import to avoid hard dependency
try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if TYPE_CHECKING:
    from aurora.core.memory import AuroraMemory


# =============================================================================
# Metric Definitions
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # -------------------------------------------------------------------------
    # Algorithm Health Metrics
    # -------------------------------------------------------------------------
    
    # Narrative tension: measures user engagement/interest level
    # Higher tension = more engaging content being processed
    aurora_tension = Histogram(
        "aurora_tension",
        "Distribution of narrative tension values",
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0],
    )
    
    aurora_tension_avg = Gauge(
        "aurora_tension_avg",
        "Average narrative tension (higher = more engaging content)",
        ["user_id"],
    )
    
    # Cluster entropy: measures story/theme diversity
    # Low entropy = "super story" absorbing all content (potential problem)
    # High entropy = healthy diversity of narratives
    aurora_cluster_entropy = Gauge(
        "aurora_cluster_entropy",
        "Story cluster entropy (higher = more diverse narratives)",
        ["user_id"],
    )
    
    aurora_story_count = Gauge(
        "aurora_story_count",
        "Number of active stories",
        ["user_id"],
    )
    
    aurora_theme_count = Gauge(
        "aurora_theme_count",
        "Number of emerged themes",
        ["user_id"],
    )
    
    # Gate pass rate: measures memory encoding selectivity
    # Too high = remembering too much (clutter)
    # Too low = remembering too little (amnesia)
    aurora_gate_pass_rate = Gauge(
        "aurora_gate_pass_rate",
        "Memory gate pass rate (fraction of inputs encoded)",
        ["user_id"],
    )
    
    aurora_gate_decisions = Counter(
        "aurora_gate_decisions_total",
        "Total gate decisions by type",
        ["user_id", "decision"],  # decision: "encode" or "skip"
    )
    
    # -------------------------------------------------------------------------
    # Performance Metrics
    # -------------------------------------------------------------------------
    
    aurora_ingest_latency = Histogram(
        "aurora_ingest_latency_seconds",
        "Ingest operation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    
    aurora_query_latency = Histogram(
        "aurora_query_latency_seconds",
        "Query operation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    
    aurora_evolve_latency = Histogram(
        "aurora_evolve_latency_seconds",
        "Evolution operation latency",
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    
    # -------------------------------------------------------------------------
    # Storage Metrics
    # -------------------------------------------------------------------------
    
    aurora_plot_count = Gauge(
        "aurora_plot_count",
        "Number of stored plots",
        ["user_id"],
    )
    
    aurora_vector_count = Gauge(
        "aurora_vector_count",
        "Number of vectors in index",
        ["user_id", "kind"],  # kind: "plot", "story", "theme"
    )
    
    aurora_snapshot_size_bytes = Gauge(
        "aurora_snapshot_size_bytes",
        "Size of last snapshot in bytes",
        ["user_id"],
    )
    
    # -------------------------------------------------------------------------
    # Learning Metrics
    # -------------------------------------------------------------------------
    
    aurora_metric_loss = Gauge(
        "aurora_metric_loss_avg",
        "Average triplet loss for metric learning",
        ["user_id"],
    )
    
    aurora_feedback_count = Counter(
        "aurora_feedback_total",
        "Total feedback events by outcome",
        ["user_id", "success"],  # success: "true" or "false"
    )
    
    # -------------------------------------------------------------------------
    # System Info
    # -------------------------------------------------------------------------
    
    aurora_info = Info(
        "aurora",
        "Aurora memory system information",
    )

else:
    # Dummy implementations when prometheus_client is not installed
    class _DummyMetric:
        def labels(self, **kwargs): return self
        def set(self, value): pass
        def inc(self, value=1): pass
        def observe(self, value): pass
        def time(self): return _DummyContextManager()
        def info(self, info_dict): pass
    
    class _DummyContextManager:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    aurora_tension = _DummyMetric()
    aurora_tension_avg = _DummyMetric()
    aurora_cluster_entropy = _DummyMetric()
    aurora_story_count = _DummyMetric()
    aurora_theme_count = _DummyMetric()
    aurora_gate_pass_rate = _DummyMetric()
    aurora_gate_decisions = _DummyMetric()
    aurora_ingest_latency = _DummyMetric()
    aurora_query_latency = _DummyMetric()
    aurora_evolve_latency = _DummyMetric()
    aurora_plot_count = _DummyMetric()
    aurora_vector_count = _DummyMetric()
    aurora_snapshot_size_bytes = _DummyMetric()
    aurora_metric_loss = _DummyMetric()
    aurora_feedback_count = _DummyMetric()
    aurora_info = _DummyMetric()


# =============================================================================
# Metric Computation Helpers
# =============================================================================

def compute_cluster_entropy(cluster_sizes: Dict[str, int]) -> float:
    """Compute Shannon entropy of cluster distribution.
    
    Higher entropy = more uniform distribution (healthy)
    Lower entropy = one cluster dominating (potential "super story")
    
    Args:
        cluster_sizes: Dict mapping cluster_id -> number of items
        
    Returns:
        Entropy in bits (log2)
    """
    if not cluster_sizes:
        return 0.0
    
    total = sum(cluster_sizes.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for size in cluster_sizes.values():
        if size > 0:
            p = size / total
            entropy -= p * math.log2(p)
    
    return entropy


def compute_tension_avg(tension_values: list) -> float:
    """Compute average tension from recent values.
    
    Args:
        tension_values: List of tension floats
        
    Returns:
        Average tension, or 0.0 if empty
    """
    if not tension_values:
        return 0.0
    return sum(tension_values) / len(tension_values)


def update_memory_metrics(memory: "AuroraMemory", user_id: str) -> Dict[str, float]:
    """Update all metrics from an AuroraMemory instance.
    
    Call this periodically (e.g., after ingest or evolve) to update metrics.
    
    Args:
        memory: AuroraMemory instance
        user_id: User identifier for metric labels
        
    Returns:
        Dict of computed metric values
    """
    metrics_data = {}
    
    # Story cluster entropy
    story_sizes = {sid: len(s.plot_ids) for sid, s in memory.stories.items()}
    entropy = compute_cluster_entropy(story_sizes)
    aurora_cluster_entropy.labels(user_id=user_id).set(entropy)
    metrics_data["cluster_entropy"] = entropy
    
    # Counts
    aurora_story_count.labels(user_id=user_id).set(len(memory.stories))
    aurora_theme_count.labels(user_id=user_id).set(len(memory.themes))
    aurora_plot_count.labels(user_id=user_id).set(len(memory.plots))
    metrics_data["story_count"] = len(memory.stories)
    metrics_data["theme_count"] = len(memory.themes)
    metrics_data["plot_count"] = len(memory.plots)
    
    # Gate pass rate
    pass_rate = memory.gate.pass_rate()
    aurora_gate_pass_rate.labels(user_id=user_id).set(pass_rate)
    metrics_data["gate_pass_rate"] = pass_rate
    
    # Average tension from recent plots
    recent_plots = [memory.plots[pid] for pid in memory._recent_encoded_plot_ids 
                   if pid in memory.plots]
    if recent_plots:
        tensions = [p.tension for p in recent_plots[-50:]]  # Last 50
        avg_tension = compute_tension_avg(tensions)
        aurora_tension_avg.labels(user_id=user_id).set(avg_tension)
        metrics_data["tension_avg"] = avg_tension
    
    # Metric learning loss
    if hasattr(memory.metric, "average_loss"):
        loss = memory.metric.average_loss()
        aurora_metric_loss.labels(user_id=user_id).set(loss)
        metrics_data["metric_loss"] = loss
    
    return metrics_data


def record_ingest_metrics(
    user_id: str,
    tension: float,
    encoded: bool,
    latency_seconds: float,
) -> None:
    """Record metrics for an ingest operation.
    
    Args:
        user_id: User identifier
        tension: Computed tension value
        encoded: Whether the plot was encoded (gate decision)
        latency_seconds: Operation latency
    """
    aurora_tension.observe(tension)
    aurora_ingest_latency.observe(latency_seconds)
    
    decision = "encode" if encoded else "skip"
    aurora_gate_decisions.labels(user_id=user_id, decision=decision).inc()


def record_query_metrics(user_id: str, latency_seconds: float) -> None:
    """Record metrics for a query operation."""
    aurora_query_latency.observe(latency_seconds)


def record_feedback_metrics(user_id: str, success: bool) -> None:
    """Record metrics for a feedback event."""
    aurora_feedback_count.labels(user_id=user_id, success=str(success).lower()).inc()


def record_evolve_metrics(user_id: str, latency_seconds: float) -> None:
    """Record metrics for an evolve operation."""
    aurora_evolve_latency.observe(latency_seconds)


def set_system_info(version: str, config: Optional[Dict] = None) -> None:
    """Set system information metrics."""
    info_dict = {"version": version}
    if config:
        info_dict.update({f"config_{k}": str(v) for k, v in config.items()})
    aurora_info.info(info_dict)


# =============================================================================
# Metric Exposition
# =============================================================================

def get_metrics_text() -> str:
    """Get metrics in Prometheus text format.
    
    Returns:
        Prometheus text format metrics string
    """
    if PROMETHEUS_AVAILABLE:
        from prometheus_client import generate_latest
        return generate_latest().decode("utf-8")
    return "# Prometheus client not installed\n"


def get_metrics_json(memory: "AuroraMemory", user_id: str) -> Dict[str, float]:
    """Get current metrics as JSON-serializable dict.
    
    Useful for API responses or debugging.
    
    Args:
        memory: AuroraMemory instance
        user_id: User identifier
        
    Returns:
        Dict of metric name -> value
    """
    return update_memory_metrics(memory, user_id)
