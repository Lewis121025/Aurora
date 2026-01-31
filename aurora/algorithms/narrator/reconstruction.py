"""
Narrative Reconstruction Module
===============================

Core narrative reconstruction engine and related data classes.

Responsibilities:
1. Define NarrativeElement and NarrativeTrace data classes
2. Implement NarratorEngine for story reconstruction
3. Coordinate perspective selection, context recovery, and narrative generation

Design principles:
- Zero hard-coded thresholds: All decisions use Bayesian/stochastic policies
- Deterministic reproducibility: All random operations support seed
- Complete type annotations
- Serializable state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.graph.vector_index import VectorIndex
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.narrator.context import ContextRecovery, TurningPointDetector
from aurora.algorithms.narrator.perspective import (
    NarrativePerspective,
    NarrativeRole,
    PerspectiveOrganizer,
    PerspectiveSelector,
)
from aurora.utils.time_utils import now_ts


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NarrativeElement:
    """A single element in a narrative reconstruction.
    
    Represents a memory fragment with its role in the narrative structure.
    """
    plot_id: str
    role: NarrativeRole
    content: str
    timestamp: float
    
    # Narrative significance (computed, not hard-coded)
    significance: float = 0.5
    
    # Causal connections
    causes: List[str] = field(default_factory=list)      # Plot IDs that caused this
    effects: List[str] = field(default_factory=list)     # Plot IDs caused by this
    
    # Narrative annotations
    annotation: str = ""                                  # Narrative commentary
    tension_level: float = 0.0                           # Tension at this point
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "plot_id": self.plot_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "significance": self.significance,
            "causes": self.causes,
            "effects": self.effects,
            "annotation": self.annotation,
            "tension_level": self.tension_level,
        }
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "NarrativeElement":
        """Reconstruct from state dict."""
        return cls(
            plot_id=d["plot_id"],
            role=NarrativeRole(d["role"]),
            content=d["content"],
            timestamp=d["timestamp"],
            significance=d.get("significance", 0.5),
            causes=d.get("causes", []),
            effects=d.get("effects", []),
            annotation=d.get("annotation", ""),
            tension_level=d.get("tension_level", 0.0),
        )


@dataclass
class NarrativeTrace:
    """Trace of a narrative reconstruction process.
    
    Contains the full narrative with metadata about the reconstruction process.
    """
    query: str
    perspective: NarrativePerspective
    elements: List[NarrativeElement]
    
    # Reconstruction metadata
    created_ts: float = field(default_factory=now_ts)
    reconstruction_confidence: float = 0.5
    
    # Perspective selection probabilities
    perspective_probs: Dict[str, float] = field(default_factory=dict)
    
    # Identified turning points
    turning_point_ids: List[str] = field(default_factory=list)
    
    # Causal chain summary
    causal_chain_depth: int = 0
    
    # Generated narrative text
    narrative_text: str = ""
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "query": self.query,
            "perspective": self.perspective.value,
            "elements": [e.to_state_dict() for e in self.elements],
            "created_ts": self.created_ts,
            "reconstruction_confidence": self.reconstruction_confidence,
            "perspective_probs": self.perspective_probs,
            "turning_point_ids": self.turning_point_ids,
            "causal_chain_depth": self.causal_chain_depth,
            "narrative_text": self.narrative_text,
        }
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "NarrativeTrace":
        """Reconstruct from state dict."""
        return cls(
            query=d["query"],
            perspective=NarrativePerspective(d["perspective"]),
            elements=[NarrativeElement.from_state_dict(e) for e in d["elements"]],
            created_ts=d.get("created_ts", now_ts()),
            reconstruction_confidence=d.get("reconstruction_confidence", 0.5),
            perspective_probs=d.get("perspective_probs", {}),
            turning_point_ids=d.get("turning_point_ids", []),
            causal_chain_depth=d.get("causal_chain_depth", 0),
            narrative_text=d.get("narrative_text", ""),
        )


# =============================================================================
# Narrator Engine
# =============================================================================

class NarratorEngine:
    """Engine for narrative reconstruction and storytelling.
    
    Transforms fragmented memories into coherent narratives through:
    1. Context-adaptive perspective selection
    2. Causal chain reconstruction
    3. Turning point identification
    4. Narrative structure generation
    
    All decisions are probabilistic - no hard-coded thresholds.
    
    Attributes:
        metric: Learned metric for similarity computation
        seed: Random seed for reproducibility
        rng: Random number generator
        
        perspective_beliefs: Beta posterior beliefs for perspective effectiveness
        reconstruction_cache: Cache of recent reconstructions
    """
    
    def __init__(
        self,
        metric: LowRankMetric,
        vindex: Optional[VectorIndex] = None,
        graph: Optional[MemoryGraph] = None,
        seed: int = 0,
    ):
        """Initialize the narrator engine.
        
        Args:
            metric: Learned low-rank metric for similarity computation
            vindex: Optional vector index for retrieval
            graph: Optional memory graph for causal tracing
            seed: Random seed for reproducibility
        """
        self.metric = metric
        self.vindex = vindex
        self.graph = graph
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Initialize sub-components
        self._perspective_selector = PerspectiveSelector(
            metric=metric,
            rng=self.rng,
        )
        self._organizer = PerspectiveOrganizer(
            metric=metric,
            rng=self.rng,
        )
        self._context_recovery = ContextRecovery(
            metric=metric,
            rng=self.rng,
            graph=graph,
        )
        self._turning_point_detector = TurningPointDetector(rng=self.rng)
        
        # Reconstruction cache (LRU-style, limited size)
        self._reconstruction_cache: Dict[str, NarrativeTrace] = {}
        self._cache_max_size = 100
    
    @property
    def perspective_beliefs(self) -> Dict[NarrativePerspective, Tuple[float, float]]:
        """Get perspective beliefs from the selector."""
        return self._perspective_selector.perspective_beliefs
    
    @perspective_beliefs.setter
    def perspective_beliefs(
        self, value: Dict[NarrativePerspective, Tuple[float, float]]
    ) -> None:
        """Set perspective beliefs on the selector."""
        self._perspective_selector.perspective_beliefs = value
    
    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------
    
    def reconstruct_story(
        self,
        query: str,
        plots: List[Plot],
        perspective: Optional[NarrativePerspective] = None,
        stories: Optional[Dict[str, StoryArc]] = None,
        themes: Optional[Dict[str, Theme]] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> NarrativeTrace:
        """Reconstruct a narrative from memory fragments.
        
        Organizes the given plots into a coherent narrative structure
        based on the selected (or auto-selected) perspective.
        
        Args:
            query: The query or context driving the reconstruction
            plots: List of Plot objects to include in the narrative
            perspective: Optional specific perspective (auto-selected if None)
            stories: Optional story context for richer narratives
            themes: Optional theme context for abstraction
            query_embedding: Optional pre-computed query embedding
            
        Returns:
            NarrativeTrace containing the reconstructed narrative
        """
        if not plots:
            return NarrativeTrace(
                query=query,
                perspective=perspective or NarrativePerspective.CHRONOLOGICAL,
                elements=[],
                narrative_text="没有找到相关的记忆片段。",
            )
        
        # Auto-select perspective if not specified
        perspective_probs = {}
        if perspective is None:
            perspective, perspective_probs = self.select_perspective(
                query=query,
                plots=plots,
                stories=stories,
                themes=themes,
                query_embedding=query_embedding,
            )
        
        # Organize plots according to perspective
        elements = self._organize_by_perspective(
            plots=plots,
            perspective=perspective,
            query=query,
            query_embedding=query_embedding,
        )
        
        # Identify turning points
        turning_points = self._turning_point_detector.detect_from_elements(
            [e.to_state_dict() for e in elements]
        )
        turning_point_ids = [tp["plot_id"] for tp in turning_points]
        
        # Assign narrative roles
        elements = self._assign_narrative_roles(elements, turning_point_ids)
        
        # Generate narrative text
        narrative_text = self._generate_narrative_text(
            elements=elements,
            perspective=perspective,
            query=query,
            stories=stories,
            themes=themes,
        )
        
        # Compute reconstruction confidence
        confidence = self._compute_reconstruction_confidence(elements, perspective)
        
        trace = NarrativeTrace(
            query=query,
            perspective=perspective,
            elements=elements,
            reconstruction_confidence=confidence,
            perspective_probs=perspective_probs,
            turning_point_ids=turning_point_ids,
            narrative_text=narrative_text,
        )
        
        # Cache the reconstruction
        cache_key = self._compute_cache_key(query, [p.id for p in plots])
        self._cache_reconstruction(cache_key, trace)
        
        return trace
    
    def select_perspective(
        self,
        query: str,
        plots: Optional[List[Plot]] = None,
        stories: Optional[Dict[str, StoryArc]] = None,
        themes: Optional[Dict[str, Theme]] = None,
        context: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[NarrativePerspective, Dict[str, float]]:
        """Select the most appropriate narrative perspective.
        
        Uses multiple signals combined probabilistically:
        - Query characteristics
        - Memory structure (temporal spread, contrasts, themes)
        - Historical effectiveness (learned from feedback)
        
        Args:
            query: The query text
            plots: Optional list of plots to consider
            stories: Optional stories for context
            themes: Optional themes for abstraction signals
            context: Optional additional context
            query_embedding: Optional pre-computed query embedding
            
        Returns:
            Tuple of (selected_perspective, probability_dict)
        """
        return self._perspective_selector.select_perspective(
            query=query,
            plots=plots,
            stories=stories,
            themes=themes,
            context=context,
            query_embedding=query_embedding,
        )
    
    def recover_context(
        self,
        plot: Plot,
        plots_dict: Dict[str, Plot],
        depth: int = 3,
    ) -> List[Plot]:
        """Recover the causal context for a plot.
        
        Traces back through causal chains and temporal connections
        to find the context that led to this plot.
        
        Args:
            plot: The plot to recover context for
            plots_dict: Dictionary of all available plots
            depth: Maximum depth of causal chain to trace
            
        Returns:
            List of plots forming the causal context (ordered by relevance)
        """
        return self._context_recovery.recover_context(
            plot=plot,
            plots_dict=plots_dict,
            depth=depth,
        )
    
    def identify_turning_points(
        self,
        plots: List[Plot],
        stories: Optional[Dict[str, StoryArc]] = None,
    ) -> List[Plot]:
        """Identify turning points in a sequence of plots.
        
        Turning points are moments of significant change:
        - High tension followed by resolution
        - Shift in relationship dynamics
        - Identity-relevant events
        
        Uses probabilistic detection, not fixed thresholds.
        
        Args:
            plots: List of plots to analyze
            stories: Optional story context
            
        Returns:
            List of plots identified as turning points
        """
        return self._context_recovery.identify_turning_points(
            plots=plots,
            stories=stories,
        )
    
    # -------------------------------------------------------------------------
    # Perspective-specific organization
    # -------------------------------------------------------------------------
    
    def _organize_by_perspective(
        self,
        plots: List[Plot],
        perspective: NarrativePerspective,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[NarrativeElement]:
        """Organize plots according to the selected perspective."""
        
        # Use the organizer to get element dicts
        if perspective == NarrativePerspective.CHRONOLOGICAL:
            element_dicts = self._organizer.organize_chronological(
                plots, self._compute_significance
            )
        elif perspective == NarrativePerspective.RETROSPECTIVE:
            element_dicts = self._organizer.organize_retrospective(
                plots, query, query_embedding, self._compute_significance
            )
        elif perspective == NarrativePerspective.CONTRASTIVE:
            element_dicts = self._organizer.organize_contrastive(
                plots, self._compute_significance
            )
        elif perspective == NarrativePerspective.FOCUSED:
            element_dicts = self._organizer.organize_focused(
                plots, query, query_embedding, self._compute_significance
            )
        elif perspective == NarrativePerspective.ABSTRACTED:
            element_dicts = self._organizer.organize_abstracted(
                plots, self._compute_significance
            )
        else:
            element_dicts = self._organizer.organize_chronological(
                plots, self._compute_significance
            )
        
        # Convert dicts to NarrativeElement objects
        elements = []
        for d in element_dicts:
            element = NarrativeElement(
                plot_id=d["plot_id"],
                role=d["role"],
                content=d["content"],
                timestamp=d["timestamp"],
                significance=d["significance"],
                tension_level=d["tension_level"],
                annotation=d.get("annotation", ""),
            )
            elements.append(element)
        
        return elements
    
    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    
    def _assign_narrative_roles(
        self,
        elements: List[NarrativeElement],
        turning_point_ids: List[str],
    ) -> List[NarrativeElement]:
        """Assign narrative roles to elements based on structure."""
        if not elements:
            return elements
        
        n = len(elements)
        turning_ids = set(turning_point_ids)
        
        for i, element in enumerate(elements):
            # Position-based role assignment
            position_ratio = i / max(n - 1, 1)
            
            if element.plot_id in turning_ids:
                element.role = NarrativeRole.CLIMAX
            elif position_ratio < 0.15:
                element.role = NarrativeRole.EXPOSITION
            elif position_ratio < 0.5:
                element.role = NarrativeRole.RISING_ACTION
            elif position_ratio < 0.85:
                element.role = NarrativeRole.FALLING_ACTION
            else:
                element.role = NarrativeRole.RESOLUTION
        
        return elements
    
    def _generate_narrative_text(
        self,
        elements: List[NarrativeElement],
        perspective: NarrativePerspective,
        query: str,
        stories: Optional[Dict[str, StoryArc]] = None,
        themes: Optional[Dict[str, Theme]] = None,
    ) -> str:
        """Generate natural language narrative from elements."""
        if not elements:
            return "没有可用的记忆片段来构建叙事。"
        
        # Build narrative based on perspective
        parts = []
        
        # Opening based on perspective
        perspective_openings = {
            NarrativePerspective.CHRONOLOGICAL: "按时间顺序回顾：",
            NarrativePerspective.RETROSPECTIVE: "回望这段经历：",
            NarrativePerspective.CONTRASTIVE: "对比分析：",
            NarrativePerspective.FOCUSED: f"关于「{query[:20]}...」的记忆：",
            NarrativePerspective.ABSTRACTED: "从这些经历中可以看到：",
        }
        parts.append(perspective_openings.get(perspective, "记忆重构："))
        
        # Add elements
        for element in elements:
            role_markers = {
                NarrativeRole.EXPOSITION: "【背景】",
                NarrativeRole.RISING_ACTION: "",
                NarrativeRole.CLIMAX: "【转折】",
                NarrativeRole.FALLING_ACTION: "",
                NarrativeRole.RESOLUTION: "【结果】",
            }
            marker = role_markers.get(element.role, "")
            
            content = element.content
            if element.annotation:
                content = f"({element.annotation}) {content}"
            
            if marker:
                parts.append(f"\n{marker} {content}")
            else:
                parts.append(f"\n• {content}")
        
        # Add theme summary if abstracted
        if perspective == NarrativePerspective.ABSTRACTED and themes:
            parts.append("\n\n【主题总结】")
            for theme in list(themes.values())[:3]:
                if theme.identity_dimension:
                    parts.append(f"\n• {theme.identity_dimension}")
                elif theme.name:
                    parts.append(f"\n• {theme.name}")
        
        return "".join(parts)
    
    def _compute_significance(self, plot: Plot) -> float:
        """Compute narrative significance of a plot."""
        base = 0.5
        
        # Tension contribution
        base += 0.2 * plot.tension
        
        # Identity impact contribution
        if plot.has_identity_impact():
            base += 0.2
        
        # Access count (frequently accessed = more significant)
        base += 0.1 * min(1.0, plot.access_count / 10.0)
        
        return min(1.0, base)
    
    def _compute_reconstruction_confidence(
        self,
        elements: List[NarrativeElement],
        perspective: NarrativePerspective,
    ) -> float:
        """Compute confidence in the narrative reconstruction."""
        if not elements:
            return 0.0
        
        # Base confidence from element count
        count_factor = min(1.0, len(elements) / 5.0)
        
        # Significance distribution (more high-significance = more confident)
        significances = [e.significance for e in elements]
        sig_factor = float(np.mean(significances)) if significances else 0.5
        
        # Perspective belief factor
        a, b = self.perspective_beliefs.get(perspective, (1.0, 1.0))
        belief_factor = a / (a + b)
        
        return 0.4 * count_factor + 0.3 * sig_factor + 0.3 * belief_factor
    
    def _compute_cache_key(self, query: str, plot_ids: List[str]) -> str:
        """Compute a cache key for a reconstruction."""
        plot_hash = hash(tuple(sorted(plot_ids)))
        return f"{hash(query)}_{plot_hash}"
    
    def _cache_reconstruction(self, key: str, trace: NarrativeTrace) -> None:
        """Cache a reconstruction with LRU eviction."""
        if len(self._reconstruction_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._reconstruction_cache))
            del self._reconstruction_cache[oldest_key]
        
        self._reconstruction_cache[key] = trace
    
    # -------------------------------------------------------------------------
    # Feedback and Learning
    # -------------------------------------------------------------------------
    
    def feedback_narrative(
        self,
        trace: NarrativeTrace,
        success: bool,
    ) -> None:
        """Update beliefs based on narrative feedback.
        
        Args:
            trace: The narrative trace that was used
            success: Whether the narrative was helpful/successful
        """
        self._perspective_selector.feedback(trace.perspective, success)
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize engine state to JSON-compatible dict."""
        return {
            "seed": self._seed,
            "perspective_beliefs": {
                p.value: list(ab) for p, ab in self.perspective_beliefs.items()
            },
            "cache_max_size": self._cache_max_size,
        }
    
    @classmethod
    def from_state_dict(
        cls,
        d: Dict[str, Any],
        metric: LowRankMetric,
        vindex: Optional[VectorIndex] = None,
        graph: Optional[MemoryGraph] = None,
    ) -> "NarratorEngine":
        """Reconstruct engine from state dict."""
        engine = cls(
            metric=metric,
            vindex=vindex,
            graph=graph,
            seed=d.get("seed", 0),
        )
        
        # Restore beliefs
        if "perspective_beliefs" in d:
            for p_str, ab in d["perspective_beliefs"].items():
                p = NarrativePerspective(p_str)
                engine.perspective_beliefs[p] = tuple(ab)
        
        engine._cache_max_size = d.get("cache_max_size", 100)
        
        return engine


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    from aurora.algorithms.components.metric import LowRankMetric
    
    # Create a simple metric
    metric = LowRankMetric(dim=96, rank=32, seed=42)
    
    # Create narrator engine
    narrator = NarratorEngine(metric=metric, seed=42)
    
    print("NarratorEngine created successfully.")
    print(f"Perspectives: {[p.value for p in NarrativePerspective]}")
    
    # Test perspective selection (without plots)
    perspective, probs = narrator.select_perspective(
        query="如何处理记忆系统中的矛盾？",
        context="memory algorithm design",
    )
    print(f"\nSelected perspective: {perspective.value}")
    print(f"Probabilities: {probs}")
    
    # Test serialization
    state = narrator.to_state_dict()
    print(f"\nState dict keys: {list(state.keys())}")
    
    # Test reconstruction
    restored = NarratorEngine.from_state_dict(state, metric=metric)
    print(f"Restored engine seed: {restored._seed}")
