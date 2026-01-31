"""
Narrative Perspective Module
============================

Defines narrative perspectives and perspective selection logic.

Responsibilities:
1. Define narrative perspective types (chronological, retrospective, etc.)
2. Define narrative roles for story structure
3. Perspective selection with Bayesian/stochastic policies
4. Perspective-specific organization of memory elements

Design principles:
- Zero hard-coded thresholds: All decisions use Bayesian/stochastic policies
- Deterministic reproducibility: All random operations support seed
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.constants import (
    PERSPECTIVE_PRIOR_ABSTRACTED,
    PERSPECTIVE_PRIOR_CHRONOLOGICAL,
    PERSPECTIVE_PRIOR_CONTRASTIVE,
    PERSPECTIVE_PRIOR_FOCUSED,
    PERSPECTIVE_PRIOR_RETROSPECTIVE,
    SNIPPET_MAX_LENGTH,
)
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.utils.math_utils import softmax
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
# Perspective Selector
# =============================================================================

class PerspectiveSelector:
    """Selects narrative perspective using Bayesian approach.
    
    Uses multiple signals combined probabilistically:
    - Query characteristics
    - Memory structure (temporal spread, contrasts, themes)
    - Historical effectiveness (learned from feedback)
    
    Attributes:
        metric: Learned metric for similarity computation
        rng: Random number generator for reproducibility
        perspective_beliefs: Beta posterior beliefs for perspective effectiveness
    """
    
    def __init__(
        self,
        metric: LowRankMetric,
        rng: np.random.Generator,
        perspective_beliefs: Optional[Dict[NarrativePerspective, Tuple[float, float]]] = None,
    ):
        """Initialize the perspective selector.
        
        Args:
            metric: Learned low-rank metric for similarity computation
            rng: Random number generator
            perspective_beliefs: Optional pre-existing beliefs (Beta posteriors)
        """
        self.metric: LowRankMetric = metric
        self.rng: np.random.Generator = rng
        
        # Default beliefs if not provided
        self.perspective_beliefs = perspective_beliefs or {
            NarrativePerspective.CHRONOLOGICAL: (2.0, 1.0),
            NarrativePerspective.RETROSPECTIVE: (1.5, 1.0),
            NarrativePerspective.CONTRASTIVE: (1.2, 1.0),
            NarrativePerspective.FOCUSED: (1.5, 1.0),
            NarrativePerspective.ABSTRACTED: (1.0, 1.0),
        }
    
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
    
    def feedback(self, perspective: NarrativePerspective, success: bool) -> None:
        """Update beliefs based on feedback.
        
        Args:
            perspective: The perspective that was used
            success: Whether it was helpful/successful
        """
        a, b = self.perspective_beliefs[perspective]
        if success:
            self.perspective_beliefs[perspective] = (a + 1.0, b)
        else:
            self.perspective_beliefs[perspective] = (a, b + 1.0)


# =============================================================================
# Perspective-specific Organization
# =============================================================================

class PerspectiveOrganizer:
    """Organizes plots according to different narrative perspectives.
    
    Provides methods to arrange memory elements in ways that suit
    different storytelling approaches.
    """
    
    def __init__(self, metric: LowRankMetric, rng: np.random.Generator):
        """Initialize the organizer.
        
        Args:
            metric: Learned metric for similarity computation
            rng: Random number generator
        """
        self.metric: LowRankMetric = metric
        self.rng: np.random.Generator = rng
    
    def organize_chronological(
        self,
        plots: List[Plot],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """Organize plots in time order.
        
        Args:
            plots: Plots to organize
            compute_significance: Function to compute plot significance
            
        Returns:
            List of element dicts with plot info and role
        """
        sorted_plots = sorted(plots, key=lambda p: p.ts)
        
        elements = []
        for plot in sorted_plots:
            element = {
                "plot_id": plot.id,
                "role": NarrativeRole.RISING_ACTION,
                "content": self._truncate_content(plot.text),
                "timestamp": plot.ts,
                "significance": compute_significance(plot),
                "tension_level": plot.tension,
                "annotation": "",
            }
            elements.append(element)
        
        return elements
    
    def organize_retrospective(
        self,
        plots: List[Plot],
        query: str,
        query_embedding: Optional[np.ndarray],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """Organize plots from present looking back.
        
        Most recent and most relevant events first, then trace back.
        """
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
            element = {
                "plot_id": plot.id,
                "role": NarrativeRole.RISING_ACTION,
                "content": self._truncate_content(plot.text),
                "timestamp": plot.ts,
                "significance": retrospective_score(plot),
                "tension_level": plot.tension,
                "annotation": "回顾" if i == 0 else "",
            }
            elements.append(element)
        
        return elements
    
    def organize_contrastive(
        self,
        plots: List[Plot],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """Organize plots to highlight contrasts and changes.
        
        Pairs similar plots that have different outcomes or emotions.
        """
        elements = []
        used = set()
        
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
                if semantic_sim > 0.4:
                    contrast_score = semantic_sim * tension_diff
                    if contrast_score > best_contrast_score:
                        best_contrast_score = contrast_score
                        best_contrast = plot_b
            
            # Add the pair
            element_a = {
                "plot_id": plot_a.id,
                "role": NarrativeRole.RISING_ACTION,
                "content": self._truncate_content(plot_a.text),
                "timestamp": plot_a.ts,
                "significance": compute_significance(plot_a),
                "tension_level": plot_a.tension,
                "annotation": "对比起点",
            }
            elements.append(element_a)
            used.add(plot_a.id)
            
            if best_contrast:
                element_b = {
                    "plot_id": best_contrast.id,
                    "role": NarrativeRole.RISING_ACTION,
                    "content": self._truncate_content(best_contrast.text),
                    "timestamp": best_contrast.ts,
                    "significance": compute_significance(best_contrast),
                    "tension_level": best_contrast.tension,
                    "annotation": "对比终点",
                }
                elements.append(element_b)
                used.add(best_contrast.id)
        
        return elements
    
    def organize_focused(
        self,
        plots: List[Plot],
        query: str,
        query_embedding: Optional[np.ndarray],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """Organize plots around a specific focus.
        
        Most relevant to query first, with decreasing relevance.
        """
        if query_embedding is None:
            return self.organize_chronological(plots, compute_significance)
        
        def focus_score(plot: Plot) -> float:
            return self.metric.sim(plot.embedding, query_embedding)
        
        sorted_plots = sorted(plots, key=focus_score, reverse=True)
        
        elements = []
        for i, plot in enumerate(sorted_plots):
            relevance = focus_score(plot)
            element = {
                "plot_id": plot.id,
                "role": NarrativeRole.RISING_ACTION,
                "content": self._truncate_content(plot.text),
                "timestamp": plot.ts,
                "significance": relevance,
                "tension_level": plot.tension,
                "annotation": "核心" if i == 0 else ("相关" if relevance > 0.5 else ""),
            }
            elements.append(element)
        
        return elements
    
    def organize_abstracted(
        self,
        plots: List[Plot],
        compute_significance: Callable[[Plot], float],
    ) -> List[Dict[str, Any]]:
        """Organize plots by extracting patterns and themes.
        
        Groups similar plots together to form thematic clusters.
        """
        # Simple clustering by embedding similarity
        clusters: List[List[Plot]] = []
        unclustered = list(plots)
        
        while unclustered:
            seed = unclustered.pop(0)
            cluster = [seed]
            
            remaining = []
            for plot in unclustered:
                sim = self.metric.sim(seed.embedding, plot.embedding)
                if sim > 0.6:
                    cluster.append(plot)
                else:
                    remaining.append(plot)
            
            clusters.append(cluster)
            unclustered = remaining
        
        # Create elements with cluster annotations
        elements = []
        for cluster_idx, cluster in enumerate(clusters):
            cluster.sort(key=lambda p: p.ts)
            
            for i, plot in enumerate(cluster):
                annotation = f"主题{cluster_idx + 1}" if i == 0 else ""
                element = {
                    "plot_id": plot.id,
                    "role": NarrativeRole.RISING_ACTION,
                    "content": self._truncate_content(plot.text),
                    "timestamp": plot.ts,
                    "significance": compute_significance(plot),
                    "tension_level": plot.tension,
                    "annotation": annotation,
                }
                elements.append(element)
        
        return elements
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content to maximum length."""
        if len(content) <= SNIPPET_MAX_LENGTH:
            return content
        return content[:SNIPPET_MAX_LENGTH - 3] + "..."
