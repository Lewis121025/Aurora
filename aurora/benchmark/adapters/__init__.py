"""
AURORA Benchmark Adapters
==========================

Adapters for transforming external benchmark formats to AURORA's interface.

Each benchmark has its own adapter that handles:
    - Dataset loading and parsing
    - Instance evaluation
    - Result aggregation

Supported Benchmarks:
    - MemoryAgentBench: HuggingFace ai-hyz/MemoryAgentBench
    - LOCOMO (planned): GitHub snap-research/locomo

Example:
    from aurora.benchmark.adapters import MemoryAgentBenchAdapter
    
    adapter = MemoryAgentBenchAdapter()
    instances = adapter.load_dataset("ai-hyz/MemoryAgentBench")
"""

from typing import List

# Import implemented adapters
from aurora.benchmark.adapters.memoryagentbench import MemoryAgentBenchAdapter

__all__: List[str] = [
    "MemoryAgentBenchAdapter",
]
