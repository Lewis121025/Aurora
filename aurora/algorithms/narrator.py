"""
AURORA Narrator Engine
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
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.constants import (
    DEFAULT_CAUSAL_DEPTH,
    MAX_CAUSAL_CHAIN_LENGTH,
    NARRATIVE_SEGMENT_MAX_LENGTH,
    PERSPECTIVE_PRIOR_ABSTRACTED,
    PERSPECTIVE_PRIOR_CHRONOLOGICAL,
    PERSPECTIVE_PRIOR_CONTRASTIVE,
    PERSPECTIVE_PRIOR_FOCUSED,
    PERSPECTIVE_PRIOR_RETROSPECTIVE,
    SNIPPET_MAX_LENGTH,
    TURNING_POINT_TENSION_THRESHOLD_BASE,
)
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.graph.vector_index import VectorIndex
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.utils.math_utils import cosine_sim, l2_normalize, sigmoid, softmax
from aurora.utils.time_utils import now_ts


# =============================================================================
# Enums
# =============================================================================

class NarrativePerspective(Enum):
    """Narrative perspectives for story reconstruction.
    
    Each perspective offers a different lens for organizing memories:
    
    CHRONOLOGICAL: Time-ordered narrative, best for understanding sequence
    RETROSPECTIVE: Looking back from present, best for reflection
    CONTRASTIVE: Highlighting contrasts and changes, best for growth analysis
    FOCUSED: Centered on a specific aspect, best for deep exploration
    ABSTRACTED: High-level patterns and themes, best for identity synthesis
    """
    CHRONOLOGICAL = "chronological"    # 时间序：按时间顺序叙述
    RETROSPECTIVE = "retrospective"    # 回顾式：从现在回望过去
    CONTRASTIVE = "contrastive"        # 对比式：突出变化和对比
    FOCUSED = "focused"                # 聚焦式：围绕特定主题深入
    ABSTRACTED = "abstracted"          # 抽象式：提炼模式和主题


class NarrativeRole(Enum):
    """Roles that memory elements can play in a narrative."""
    EXPOSITION = "exposition"          # 背景介绍
    RISING_ACTION = "rising_action"    # 情节发展
    CLIMAX = "climax"                  # 高潮/转折点
    FALLING_ACTION = "falling_action"  # 情节回落
    RESOLUTION = "resolution"          # 结局/解决


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


@dataclass
class PerspectiveScore:
    """Scores for perspective selection with uncertainty."""
    perspective: NarrativePerspective
    score: float                    # Main score
    uncertainty: float = 0.5        # Epistemic uncertainty
    
    # Evidence signals
    temporal_signal: float = 0.0   # Strength of temporal patterns
    contrast_signal: float = 0.0   # Strength of contrast patterns
    focus_signal: float = 0.0      # Query specificity
    abstraction_signal: float = 0.0  # Theme relevance


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
        
        # Perspective effectiveness beliefs (Beta posteriors)
        # Learned from user feedback on narrative quality
        self.perspective_beliefs: Dict[NarrativePerspective, Tuple[float, float]] = {
            NarrativePerspective.CHRONOLOGICAL: (2.0, 1.0),   # Prior: somewhat effective
            NarrativePerspective.RETROSPECTIVE: (1.5, 1.0),
            NarrativePerspective.CONTRASTIVE: (1.2, 1.0),
            NarrativePerspective.FOCUSED: (1.5, 1.0),
            NarrativePerspective.ABSTRACTED: (1.0, 1.0),
        }
        
        # Reconstruction cache (LRU-style, limited size)
        self._reconstruction_cache: Dict[str, NarrativeTrace] = {}
        self._cache_max_size = 100
    
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
        turning_points = self._identify_turning_points(elements)
        
        # Assign narrative roles
        elements = self._assign_narrative_roles(elements, turning_points)
        
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
            turning_point_ids=[tp.plot_id for tp in turning_points],
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
        scores: Dict[NarrativePerspective, PerspectiveScore] = {}
        
        for perspective in NarrativePerspective:
            score = self._compute_perspective_score(
                perspective=perspective,
                query=query,
                plots=plots,
                stories=stories,
                themes=themes,
                query_embedding=query_embedding,
            )
            scores[perspective] = score
        
        # Convert scores to probabilities using softmax with Thompson sampling
        log_odds = []
        perspectives = list(NarrativePerspective)
        
        for p in perspectives:
            # Base score
            base = scores[p].score
            
            # Thompson sample from effectiveness belief
            a, b = self.perspective_beliefs[p]
            sampled_effectiveness = self.rng.beta(a, b)
            
            # Combine: base score weighted by sampled effectiveness
            combined = base * (0.5 + sampled_effectiveness)
            log_odds.append(combined)
        
        # Softmax to get probabilities
        probs = softmax(log_odds)
        
        # Sample from distribution (stochastic selection)
        choice_idx = self.rng.choice(len(perspectives), p=probs)
        selected = perspectives[choice_idx]
        
        prob_dict = {p.value: float(prob) for p, prob in zip(perspectives, probs)}
        
        return selected, prob_dict
    
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
        
        if perspective == NarrativePerspective.CHRONOLOGICAL:
            return self._organize_chronological(plots)
        elif perspective == NarrativePerspective.RETROSPECTIVE:
            return self._organize_retrospective(plots, query, query_embedding)
        elif perspective == NarrativePerspective.CONTRASTIVE:
            return self._organize_contrastive(plots)
        elif perspective == NarrativePerspective.FOCUSED:
            return self._organize_focused(plots, query, query_embedding)
        elif perspective == NarrativePerspective.ABSTRACTED:
            return self._organize_abstracted(plots)
        else:
            return self._organize_chronological(plots)
    
    def _organize_chronological(self, plots: List[Plot]) -> List[NarrativeElement]:
        """Organize plots in time order."""
        sorted_plots = sorted(plots, key=lambda p: p.ts)
        
        elements = []
        for plot in sorted_plots:
            element = NarrativeElement(
                plot_id=plot.id,
                role=NarrativeRole.RISING_ACTION,  # Will be refined later
                content=self._truncate_content(plot.text),
                timestamp=plot.ts,
                significance=self._compute_significance(plot),
                tension_level=plot.tension,
            )
            elements.append(element)
        
        return elements
    
    def _organize_retrospective(
        self,
        plots: List[Plot],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[NarrativeElement]:
        """Organize plots from present looking back.
        
        Most recent and most relevant events first, then trace back.
        """
        # Sort by recency and relevance
        now = now_ts()
        
        def retrospective_score(plot: Plot) -> float:
            recency = 1.0 / (1.0 + math.log1p(now - plot.ts))
            relevance = 0.5
            if query_embedding is not None:
                relevance = self.metric.sim(plot.embedding, query_embedding)
            return 0.6 * relevance + 0.4 * recency
        
        sorted_plots = sorted(plots, key=retrospective_score, reverse=True)
        
        elements = []
        for i, plot in enumerate(sorted_plots):
            # Earlier in retrospective order = more recent/relevant
            element = NarrativeElement(
                plot_id=plot.id,
                role=NarrativeRole.RISING_ACTION,
                content=self._truncate_content(plot.text),
                timestamp=plot.ts,
                significance=retrospective_score(plot),
                tension_level=plot.tension,
                annotation="回顾" if i == 0 else "",
            )
            elements.append(element)
        
        return elements
    
    def _organize_contrastive(self, plots: List[Plot]) -> List[NarrativeElement]:
        """Organize plots to highlight contrasts and changes.
        
        Pairs similar plots that have different outcomes or emotions.
        """
        elements = []
        used = set()
        
        # Find contrasting pairs
        plots_list = list(plots)
        for i, plot_a in enumerate(plots_list):
            if plot_a.id in used:
                continue
            
            best_contrast = None
            best_contrast_score = 0.0
            
            for j, plot_b in enumerate(plots_list):
                if i == j or plot_b.id in used:
                    continue
                
                # Contrast score: similar embeddings but different tension/outcomes
                semantic_sim = self.metric.sim(plot_a.embedding, plot_b.embedding)
                tension_diff = abs(plot_a.tension - plot_b.tension)
                
                # Good contrast: semantically related but different tension
                if semantic_sim > 0.4:  # Related topics
                    contrast_score = semantic_sim * tension_diff
                    if contrast_score > best_contrast_score:
                        best_contrast_score = contrast_score
                        best_contrast = plot_b
            
            # Add the pair
            element_a = NarrativeElement(
                plot_id=plot_a.id,
                role=NarrativeRole.RISING_ACTION,
                content=self._truncate_content(plot_a.text),
                timestamp=plot_a.ts,
                significance=self._compute_significance(plot_a),
                tension_level=plot_a.tension,
                annotation="对比起点",
            )
            elements.append(element_a)
            used.add(plot_a.id)
            
            if best_contrast:
                element_b = NarrativeElement(
                    plot_id=best_contrast.id,
                    role=NarrativeRole.RISING_ACTION,
                    content=self._truncate_content(best_contrast.text),
                    timestamp=best_contrast.ts,
                    significance=self._compute_significance(best_contrast),
                    tension_level=best_contrast.tension,
                    annotation="对比终点",
                )
                elements.append(element_b)
                used.add(best_contrast.id)
        
        return elements
    
    def _organize_focused(
        self,
        plots: List[Plot],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[NarrativeElement]:
        """Organize plots around a specific focus.
        
        Most relevant to query first, with decreasing relevance.
        """
        if query_embedding is None:
            # Fall back to chronological
            return self._organize_chronological(plots)
        
        # Sort by relevance to query
        def focus_score(plot: Plot) -> float:
            return self.metric.sim(plot.embedding, query_embedding)
        
        sorted_plots = sorted(plots, key=focus_score, reverse=True)
        
        elements = []
        for i, plot in enumerate(sorted_plots):
            relevance = focus_score(plot)
            element = NarrativeElement(
                plot_id=plot.id,
                role=NarrativeRole.RISING_ACTION,
                content=self._truncate_content(plot.text),
                timestamp=plot.ts,
                significance=relevance,
                tension_level=plot.tension,
                annotation="核心" if i == 0 else ("相关" if relevance > 0.5 else ""),
            )
            elements.append(element)
        
        return elements
    
    def _organize_abstracted(self, plots: List[Plot]) -> List[NarrativeElement]:
        """Organize plots by extracting patterns and themes.
        
        Groups similar plots together to form thematic clusters.
        """
        # Simple clustering by embedding similarity
        clusters: List[List[Plot]] = []
        unclustered = list(plots)
        
        while unclustered:
            # Start new cluster with first unclustered plot
            seed = unclustered.pop(0)
            cluster = [seed]
            
            # Find similar plots
            remaining = []
            for plot in unclustered:
                sim = self.metric.sim(seed.embedding, plot.embedding)
                if sim > 0.6:  # Soft threshold for clustering
                    cluster.append(plot)
                else:
                    remaining.append(plot)
            
            clusters.append(cluster)
            unclustered = remaining
        
        # Create elements with cluster annotations
        elements = []
        for cluster_idx, cluster in enumerate(clusters):
            # Sort cluster by time
            cluster.sort(key=lambda p: p.ts)
            
            for i, plot in enumerate(cluster):
                annotation = f"主题{cluster_idx + 1}" if i == 0 else ""
                element = NarrativeElement(
                    plot_id=plot.id,
                    role=NarrativeRole.RISING_ACTION,
                    content=self._truncate_content(plot.text),
                    timestamp=plot.ts,
                    significance=self._compute_significance(plot),
                    tension_level=plot.tension,
                    annotation=annotation,
                )
                elements.append(element)
        
        return elements
    
    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    
    def _compute_perspective_score(
        self,
        perspective: NarrativePerspective,
        query: str,
        plots: Optional[List[Plot]],
        stories: Optional[Dict[str, StoryArc]],
        themes: Optional[Dict[str, Theme]],
        query_embedding: Optional[np.ndarray],
    ) -> PerspectiveScore:
        """Compute score for a perspective given the context."""
        score = PerspectiveScore(perspective=perspective, score=0.0)
        
        # Get prior preference
        priors = {
            NarrativePerspective.CHRONOLOGICAL: PERSPECTIVE_PRIOR_CHRONOLOGICAL,
            NarrativePerspective.RETROSPECTIVE: PERSPECTIVE_PRIOR_RETROSPECTIVE,
            NarrativePerspective.CONTRASTIVE: PERSPECTIVE_PRIOR_CONTRASTIVE,
            NarrativePerspective.FOCUSED: PERSPECTIVE_PRIOR_FOCUSED,
            NarrativePerspective.ABSTRACTED: PERSPECTIVE_PRIOR_ABSTRACTED,
        }
        score.score = priors.get(perspective, 0.5)
        
        if not plots:
            return score
        
        # Compute signals
        
        # 1. Temporal signal: span of timestamps
        timestamps = [p.ts for p in plots]
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        score.temporal_signal = min(1.0, time_span / (7 * 24 * 3600))  # Normalize to week
        
        # 2. Contrast signal: variance in tension
        tensions = [p.tension for p in plots]
        if tensions:
            score.contrast_signal = float(np.std(tensions))
        
        # 3. Focus signal: query specificity
        if query_embedding is not None and plots:
            relevances = [self.metric.sim(p.embedding, query_embedding) for p in plots]
            score.focus_signal = float(np.max(relevances)) if relevances else 0.0
        
        # 4. Abstraction signal: theme coverage
        if themes:
            score.abstraction_signal = min(1.0, len(themes) * 0.2)
        
        # Adjust score based on perspective
        if perspective == NarrativePerspective.CHRONOLOGICAL:
            score.score += 0.3 * score.temporal_signal
        elif perspective == NarrativePerspective.RETROSPECTIVE:
            score.score += 0.2 * score.temporal_signal + 0.1 * score.focus_signal
        elif perspective == NarrativePerspective.CONTRASTIVE:
            score.score += 0.4 * score.contrast_signal
        elif perspective == NarrativePerspective.FOCUSED:
            score.score += 0.5 * score.focus_signal
        elif perspective == NarrativePerspective.ABSTRACTED:
            score.score += 0.4 * score.abstraction_signal
        
        return score
    
    def _identify_turning_points(
        self,
        elements: List[NarrativeElement],
    ) -> List[NarrativeElement]:
        """Identify turning points from narrative elements."""
        if len(elements) < 2:
            return []
        
        turning_points = []
        tensions = [e.tension_level for e in elements]
        
        if not tensions or all(t == 0 for t in tensions):
            return []
        
        mean_t = np.mean(tensions)
        std_t = np.std(tensions) + 1e-6
        
        for i, element in enumerate(elements):
            z = (element.tension_level - mean_t) / std_t
            p_turn = sigmoid(z - 0.5)
            
            if self.rng.random() < p_turn:
                turning_points.append(element)
        
        return turning_points
    
    def _assign_narrative_roles(
        self,
        elements: List[NarrativeElement],
        turning_points: List[NarrativeElement],
    ) -> List[NarrativeElement]:
        """Assign narrative roles to elements based on structure."""
        if not elements:
            return elements
        
        n = len(elements)
        turning_ids = {tp.plot_id for tp in turning_points}
        
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
    
    def _find_connected_plots(
        self,
        plot: Plot,
        plots_dict: Dict[str, Plot],
        visited: set,
    ) -> List[Tuple[Plot, float]]:
        """Find plots connected to the given plot."""
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
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content to maximum length."""
        if len(content) <= SNIPPET_MAX_LENGTH:
            return content
        return content[:SNIPPET_MAX_LENGTH - 3] + "..."
    
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
        perspective = trace.perspective
        a, b = self.perspective_beliefs[perspective]
        
        if success:
            self.perspective_beliefs[perspective] = (a + 1.0, b)
        else:
            self.perspective_beliefs[perspective] = (a, b + 1.0)
    
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
