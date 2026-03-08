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
    - LOCOMO: GitHub snap-research/locomo (ACL 2024)

Example:
    from aurora.benchmarks.adapters import MemoryAgentBenchAdapter, LOCOMOAdapter
    
    # MemoryAgentBench
    adapter = MemoryAgentBenchAdapter()
    instances = adapter.load_dataset("ai-hyz/MemoryAgentBench")
    
    # LOCOMO
    locomo_adapter = LOCOMOAdapter()
    instances = locomo_adapter.load_dataset("./data/locomo/")
"""

from typing import List

# Import implemented adapters
from aurora.benchmarks.adapters.memoryagentbench import MemoryAgentBenchAdapter
from aurora.benchmarks.adapters.locomo import LOCOMOAdapter

__all__: List[str] = [
    "MemoryAgentBenchAdapter",
    "LOCOMOAdapter",
]
