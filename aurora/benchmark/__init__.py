"""
AURORA 基准测试模块
=======================

用于针对学术基准评估 AURORA 内存系统的基准适配器。

支持的基准:
- MemoryAgentBench (2025.07): LLM 代理内存评估
- LOCOMO (ACL 2024): 长上下文内存评估（计划中）

使用方法:
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


# 适配器的延迟导入
def __getattr__(name: str):
    if name == "MemoryAgentBenchAdapter":
        from aurora.benchmark.adapters.memoryagentbench import MemoryAgentBenchAdapter
        return MemoryAgentBenchAdapter
    if name == "LOCOMOAdapter":
        from aurora.benchmark.adapters.locomo import LOCOMOAdapter
        return LOCOMOAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
