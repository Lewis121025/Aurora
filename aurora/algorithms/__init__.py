"""
AURORA Memory Algorithms
========================

Core algorithmic components for the AURORA memory system.

Public API:
    AuroraMemory - Main memory system
    TensionManager - Functional contradiction management
    NarratorEngine - Story reconstruction and narrative generation

Submodules:
    models - Data structures (Plot, StoryArc, Theme, etc.)
    components - Learnable building blocks (KDE, Metric, Bandit, etc.)
    graph - Graph structures (MemoryGraph, VectorIndex, etc.)
    retrieval - Retrieval algorithms (FieldRetriever)
    causal - Causal inference
    coherence - Coherence maintenance
    tension - Tension/contradiction management
    self_narrative - Self-narrative management
    narrator - Narrative reconstruction engine
"""

# Main system
from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.tension import TensionManager, Tension, TensionType, TensionResolution
from aurora.algorithms.narrator import (
    NarratorEngine,
    NarrativePerspective,
    NarrativeTrace,
    NarrativeElement,
    NarrativeRole,
)

__all__ = [
    "AuroraMemory",
    "TensionManager",
    "Tension",
    "TensionType",
    "TensionResolution",
    "NarratorEngine",
    "NarrativePerspective",
    "NarrativeTrace",
    "NarrativeElement",
    "NarrativeRole",
]
