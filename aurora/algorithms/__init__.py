"""
AURORA Memory Algorithms
========================

Core algorithmic components for the AURORA memory system.

Public API:
    AuroraMemory - Main memory system

Submodules:
    models - Data structures (Plot, StoryArc, Theme, etc.)
    components - Learnable building blocks (KDE, Metric, Bandit, etc.)
    graph - Graph structures (MemoryGraph, VectorIndex, etc.)
    retrieval - Retrieval algorithms (FieldRetriever)
    causal - Causal inference
    coherence - Coherence maintenance
    self_narrative - Self-narrative management
"""

# Main system - the only top-level export
from aurora.algorithms.aurora_core import AuroraMemory

__all__ = ["AuroraMemory"]
