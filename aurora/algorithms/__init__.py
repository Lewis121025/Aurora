"""
AURORA Memory Algorithms
========================

Core algorithmic components for the AURORA memory system.

Public API:
    AuroraMemory - Main memory system
    TensionManager - Functional contradiction management

Submodules:
    models - Data structures (Plot, StoryArc, Theme, etc.)
    components - Learnable building blocks (KDE, Metric, Bandit, etc.)
    graph - Graph structures (MemoryGraph, VectorIndex, etc.)
    retrieval - Retrieval algorithms (FieldRetriever)
    causal - Causal inference
    coherence - Coherence maintenance
    tension - Tension/contradiction management
    self_narrative - Self-narrative management
"""

# Main system
from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.tension import TensionManager, Tension, TensionType, TensionResolution

__all__ = [
    "AuroraMemory",
    "TensionManager",
    "Tension",
    "TensionType",
    "TensionResolution",
]
