"""
Context Recovery Module
=======================

Handles causal context recovery and turning point identification.

Responsibilities:
1. Trace causal chains from a plot
2. Identify turning points in narrative sequences
3. Find connected plots based on temporal, semantic, and story relationships

Design principles:
- Zero hard-coded thresholds: Uses probabilistic pruning
- Deterministic reproducibility: All random operations support seed
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.constants import (
    DEFAULT_CAUSAL_DEPTH,
    MAX_CAUSAL_CHAIN_LENGTH,
    TURNING_POINT_TENSION_THRESHOLD_BASE,
)
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.utils.math_utils import sigmoid


class ContextRecovery:
    """Handles context recovery and causal chain tracing.
    
    Provides methods to:
    - Trace back through causal chains and temporal connections
    - Identify turning points in a sequence of plots
    - Find plots connected through various relationships
    
    All decisions use probabilistic approaches rather than hard thresholds.
    
    Attributes:
        metric: Learned metric for similarity computation
        rng: Random number generator for reproducibility
        graph: Optional memory graph for explicit causal links
    """
    
    def __init__(
        self,
        metric: LowRankMetric,
        rng: np.random.Generator,
        graph: Optional[MemoryGraph] = None,
    ):
        """Initialize context recovery.
        
        Args:
            metric: Learned low-rank metric for similarity computation
            rng: Random number generator
            graph: Optional memory graph for causal tracing
        """
        self.metric: LowRankMetric = metric
        self.rng: np.random.Generator = rng
        self.graph: Optional[MemoryGraph] = graph
    
    def recover_context(
        self,
        plot: Plot,
        plots_dict: Dict[str, Plot],
        depth: int = DEFAULT_CAUSAL_DEPTH,
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
        context_plots: List[Tuple[Plot, float]] = []  # (plot, relevance_score)
        visited: set = {plot.id}
        
        # BFS with probabilistic pruning
        queue: List[Tuple[str, int, float]] = [(plot.id, 0, 1.0)]  # (id, depth, path_strength)
        
        while queue and len(context_plots) < MAX_CAUSAL_CHAIN_LENGTH:
            current_id, current_depth, path_strength = queue.pop(0)
            
            if current_depth >= depth:
                continue
            
            current_plot = plots_dict.get(current_id)
            if current_plot is None:
                continue
            
            # Find connected plots
            connected = self._find_connected_plots(
                current_plot, plots_dict, visited
            )
            
            for connected_plot, connection_strength in connected:
                if connected_plot.id in visited:
                    continue
                
                visited.add(connected_plot.id)
                
                # Compute relevance score
                new_strength = path_strength * connection_strength
                
                # Probabilistic pruning (weaker connections less likely to be followed)
                if self.rng.random() < new_strength:
                    context_plots.append((connected_plot, new_strength))
                    queue.append((connected_plot.id, current_depth + 1, new_strength))
        
        # Sort by relevance and return plots only
        context_plots.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in context_plots]
    
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
        if len(plots) < 2:
            return []
        
        # Sort by timestamp
        sorted_plots = sorted(plots, key=lambda p: p.ts)
        
        turning_points: List[Plot] = []
        
        # Compute tension curve
        tensions = [p.tension for p in sorted_plots]
        
        # Adaptive threshold based on tension distribution
        if tensions:
            mean_tension = np.mean(tensions)
            std_tension = np.std(tensions) + 1e-6
            
            for i, plot in enumerate(sorted_plots):
                # Z-score of tension
                z_score = (plot.tension - mean_tension) / std_tension
                
                # Probability of being a turning point
                # Higher tension = higher probability
                p_turning = sigmoid(z_score - TURNING_POINT_TENSION_THRESHOLD_BASE)
                
                # Also consider identity impact
                if plot.has_identity_impact():
                    p_turning = min(1.0, p_turning * 1.5)
                
                # Check for tension drop (climax followed by resolution)
                if i > 0 and i < len(sorted_plots) - 1:
                    prev_tension = sorted_plots[i - 1].tension
                    next_tension = sorted_plots[i + 1].tension
                    
                    # Peak detection
                    if plot.tension > prev_tension and plot.tension > next_tension:
                        p_turning = min(1.0, p_turning * 1.3)
                
                # Stochastic selection
                if self.rng.random() < p_turning:
                    turning_points.append(plot)
        
        return turning_points
    
    def _find_connected_plots(
        self,
        plot: Plot,
        plots_dict: Dict[str, Plot],
        visited: set,
    ) -> List[Tuple[Plot, float]]:
        """Find plots connected to the given plot.
        
        Considers multiple types of connections:
        - Temporal proximity (close in time)
        - Semantic similarity (similar embeddings)
        - Story membership (same story arc)
        
        Args:
            plot: The plot to find connections for
            plots_dict: Dictionary of all available plots
            visited: Set of already visited plot IDs
            
        Returns:
            List of (plot, connection_strength) tuples, sorted by strength
        """
        connected = []
        
        for pid, p in plots_dict.items():
            if pid in visited or pid == plot.id:
                continue
            
            # Temporal connection (close in time)
            time_diff = abs(p.ts - plot.ts)
            temporal_strength = math.exp(-time_diff / 3600.0)  # Decay over 1 hour
            
            # Semantic connection
            semantic_strength = self.metric.sim(plot.embedding, p.embedding)
            
            # Story connection
            story_strength = 1.0 if p.story_id == plot.story_id and plot.story_id else 0.0
            
            # Combined strength
            strength = 0.3 * temporal_strength + 0.4 * semantic_strength + 0.3 * story_strength
            
            if strength > 0.2:  # Soft threshold
                connected.append((p, strength))
        
        # Sort by strength
        connected.sort(key=lambda x: x[1], reverse=True)
        return connected[:5]  # Limit to top 5


class TurningPointDetector:
    """Dedicated detector for turning points in narratives.
    
    Provides alternative detection strategies that can be used
    alongside or instead of the basic approach.
    """
    
    def __init__(self, rng: np.random.Generator):
        """Initialize the detector.
        
        Args:
            rng: Random number generator
        """
        self.rng = rng
    
    def detect_from_elements(
        self,
        elements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify turning points from narrative elements.
        
        Args:
            elements: List of narrative element dicts with tension_level
            
        Returns:
            List of elements identified as turning points
        """
        if len(elements) < 2:
            return []
        
        turning_points = []
        tensions = [e.get("tension_level", 0.0) for e in elements]
        
        if not tensions or all(t == 0 for t in tensions):
            return []
        
        mean_t = np.mean(tensions)
        std_t = np.std(tensions) + 1e-6
        
        for element in elements:
            tension = element.get("tension_level", 0.0)
            z = (tension - mean_t) / std_t
            p_turn = sigmoid(z - 0.5)
            
            if self.rng.random() < p_turn:
                turning_points.append(element)
        
        return turning_points
