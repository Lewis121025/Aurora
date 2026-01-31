"""
AURORA Benchmark Module
=======================

Benchmark adapters for evaluating AURORA memory system against academic benchmarks.

Supported Benchmarks:
- MemoryAgentBench (2025.07): LLM Agent memory evaluation
- LOCOMO (ACL 2024): Long context memory evaluation (planned)

Usage:
    from aurora.benchmark import MemoryAgentBenchAdapter, BenchmarkCapability
    from aurora.algorithms.aurora_core import AuroraMemory
    
    adapter = MemoryAgentBenchAdapter()
    memory = AuroraMemory()
    
    results, metrics = adapter.run_benchmark_with_config(
        memory=memory,
        source="ai-hyz/MemoryAgentBench",
    )
"""

from aurora.benchmark.interface import (
    AURORABenchmarkRunner,
    BenchmarkAdapter,
    BenchmarkCapability,
    BenchmarkInstance,
    BenchmarkResult,
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationMethod,
)

__all__ = [
    "AURORABenchmarkRunner",
    "BenchmarkAdapter",
    "BenchmarkCapability",
    "BenchmarkInstance",
    "BenchmarkResult",
    "EvaluationConfig",
    "EvaluationMetrics",
    "EvaluationMethod",
]


# Lazy imports for adapters
def __getattr__(name: str):
    if name == "MemoryAgentBenchAdapter":
        from aurora.benchmark.adapters.memoryagentbench import MemoryAgentBenchAdapter
        return MemoryAgentBenchAdapter
    if name == "LOCOMOAdapter":
        from aurora.benchmark.adapters.locomo import LOCOMOAdapter
        return LOCOMOAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
