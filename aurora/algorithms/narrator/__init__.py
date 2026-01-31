"""
AURORA Narrator Module
======================

Storytelling engine for memory reconstruction and narrative generation.

Core responsibilities:
1. Story Reconstruction: Reorganize memory fragments based on query context
2. Perspective Selection: Choose optimal narrative perspective (probabilistic)
3. Context Recovery: Causal chain tracing and turning point identification

Design principles:
- Zero hard-coded thresholds: All decisions use Bayesian/stochastic policies
- Deterministic reproducibility: All random operations support seed
- Complete type annotations
- Serializable state

Usage:
    from aurora.algorithms.narrator import NarratorEngine, NarrativePerspective
    
    narrator = NarratorEngine(metric=metric, seed=42)
    trace = narrator.reconstruct_story(query, plots)
"""

from aurora.algorithms.narrator.context import (
    ContextRecovery,
    TurningPointDetector,
)
from aurora.algorithms.narrator.perspective import (
    NarrativePerspective,
    NarrativeRole,
    PerspectiveOrganizer,
    PerspectiveScore,
    PerspectiveSelector,
)
from aurora.algorithms.narrator.reconstruction import (
    NarrativeElement,
    NarrativeTrace,
    NarratorEngine,
)

__all__ = [
    # Main engine
    "NarratorEngine",
    # Data classes
    "NarrativeElement",
    "NarrativeTrace",
    # Enums
    "NarrativePerspective",
    "NarrativeRole",
    # Perspective components
    "PerspectiveScore",
    "PerspectiveSelector",
    "PerspectiveOrganizer",
    # Context components
    "ContextRecovery",
    "TurningPointDetector",
]
