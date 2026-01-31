"""
AURORA Pressure Management Module
=================================

Growth-oriented forgetting and pressure management.

Key responsibilities:
- Decide what to forget based on growth, not just capacity
- Compute identity, relationship, and growth contributions of memories
- Manage memory pressure through selective forgetting

Philosophy: "Forgetting is not losing information, it's choosing what to become."
"""

from __future__ import annotations

import logging
import math

import numpy as np

from aurora.algorithms.constants import (
    GROWTH_HINDRANCE_AGE_SECONDS,
)
from aurora.algorithms.models.plot import Plot
from aurora.utils.math_utils import softmax
from aurora.utils.time_utils import now_ts

logger = logging.getLogger(__name__)


class PressureMixin:
    """Mixin providing growth-oriented pressure management functionality."""

    # -------------------------------------------------------------------------
    # Growth-Oriented Forgetting
    # -------------------------------------------------------------------------

    def _pressure_manage(self) -> None:
        """
        Growth-oriented forgetting: decide what to forget based on growth, not just capacity.
        
        Key philosophy: "Forgetting is choosing what to become."
        """
        max_plots = self.cfg.max_plots
        if len(self.plots) <= max_plots:
            return

        # Get candidates for removal
        candidates = [plot for plot in self.plots.values() if plot.status == "active" and plot.story_id is not None]
        if not candidates:
            return

        # Score candidates and select for removal
        self._score_candidates_for_removal(candidates)
        remove_ids = self._select_plots_to_forget(candidates, len(self.plots) - max_plots)
        
        # Forget selected plots
        for pid in remove_ids:
            if pid in self.plots:
                self._forget_plot(pid)
        
        logger.debug(
            f"Pressure managed: removed {len(remove_ids)} plots, "
            f"remaining={len([plot for plot in self.plots.values() if plot.status == 'active'])}"
        )

    def _score_candidates_for_removal(self, candidates: list) -> None:
        """Score each candidate plot based on growth contribution."""
        for plot in candidates:
            identity_contribution = self._compute_identity_contribution(plot)
            relationship_contribution = self._compute_relationship_contribution(plot)
            growth_contribution = self._compute_growth_contribution(plot)
            
            plot._keep_score = (
                0.4 * identity_contribution +
                0.3 * relationship_contribution +
                0.3 * growth_contribution
            )

    def _select_plots_to_forget(self, candidates: list, excess: int) -> set:
        """Select plots to forget based on keep scores.
        
        Uses softmax over negative keep scores to probabilistically select
        plots to forget. Lower keep scores = higher probability of forgetting.
        """
        if not candidates or excess <= 0:
            return set()
        
        # Clamp excess to available candidates to avoid sampling error
        actual_excess = min(excess, len(candidates))
        
        keep_scores = np.array([getattr(plot, '_keep_score', plot.mass()) for plot in candidates], dtype=np.float32)
        logits = (-keep_scores).tolist()
        probs = np.array(softmax(logits), dtype=np.float64)
        
        return set(self.rng.choice([plot.id for plot in candidates], size=actual_excess, replace=False, p=probs))

    # -------------------------------------------------------------------------
    # Contribution Computations
    # -------------------------------------------------------------------------

    def _compute_identity_contribution(self, plot: Plot) -> float:
        """
        Compute how much this plot contributes to current identity.
        
        A plot contributes to identity if:
        - It's evidence for an identity dimension (Theme)
        - It reinforces who I am
        
        Args:
            plot: The plot to evaluate
            
        Returns:
            Identity contribution score between 0 and 1
        """
        if not plot.identity_impact:
            return 0.3  # Baseline for plots without explicit identity impact
        
        contribution = 0.0
        
        # Check if this plot is evidence for active identity dimensions
        for dim in plot.identity_impact.identity_dimensions_affected:
            dim_strength = self._identity_dimensions.get(dim, 0.0)
            contribution += dim_strength * 0.3
        
        # Check if the plot's story is connected to important themes
        if plot.story_id and plot.story_id in self.stories:
            story = self.stories[plot.story_id]
            # Find themes this story supports
            for theme in self.themes.values():
                if plot.story_id in theme.story_ids:
                    contribution += theme.confidence() * 0.2
        
        return min(1.0, contribution + 0.2)  # Baseline of 0.2
    
    def _compute_relationship_contribution(self, plot: Plot) -> float:
        """
        Compute how much this plot maintains an important relationship.
        
        A plot contributes to relationships if:
        - It's an anchor point in an important relationship
        - The relationship is healthy and ongoing
        
        Args:
            plot: The plot to evaluate
            
        Returns:
            Relationship contribution score between 0 and 1
        """
        if not plot.relational:
            return 0.3  # Baseline for non-relational plots
        
        relationship_entity = plot.relational.with_whom
        story_id = self._relationship_story_index.get(relationship_entity)
        
        if not story_id or story_id not in self.stories:
            return 0.3
        
        story = self.stories[story_id]
        
        # Factor 1: Relationship health
        health_factor = story.relationship_health
        
        # Factor 2: Relationship recency (ongoing relationships matter more)
        recency = 1.0 / (1.0 + math.log1p(now_ts() - story.updated_ts) / 10)
        
        # Factor 3: Is this plot recent in the relationship?
        if story.plot_ids:
            plot_position = story.plot_ids.index(plot.id) if plot.id in story.plot_ids else -1
            if plot_position >= 0:
                # More recent plots matter more
                recency_in_story = plot_position / len(story.plot_ids)
            else:
                recency_in_story = 0.5
        else:
            recency_in_story = 0.5
        
        return 0.3 * health_factor + 0.4 * recency + 0.3 * recency_in_story
    
    def _compute_growth_contribution(self, plot: Plot) -> float:
        """
        Compute how much keeping this plot helps growth.
        
        Questions:
        - Does this provide ongoing learning value?
        - Does this influence future behavior?
        - Does this hinder growth (e.g., reinforce negative self-image)?
        
        Args:
            plot: The plot to evaluate
            
        Returns:
            Growth contribution score between 0 and 1
        """
        # Factor 1: Learning value - high surprise/tension plots often have learning value
        learning_value = min(1.0, plot.tension * 0.5) if plot.tension > 0 else 0.3
        
        # Factor 2: Future influence - recent, accessed plots influence future
        age_factor = 1.0 / math.log1p(max(1.0, now_ts() - plot.ts))
        access_factor = math.log1p(plot.access_count + 1) / 5.0
        future_influence = 0.5 * age_factor + 0.5 * min(1.0, access_factor)
        
        # Factor 3: Growth hindrance check
        # (In a full implementation, this would use sentiment/content analysis)
        # For now, use a simple heuristic: very old, never-accessed plots may hinder
        if plot.access_count == 0 and (now_ts() - plot.ts) > GROWTH_HINDRANCE_AGE_SECONDS:
            growth_hindrance = 0.3  # Some penalty
        else:
            growth_hindrance = 0.0
        
        return 0.4 * learning_value + 0.4 * future_influence - growth_hindrance

    # -------------------------------------------------------------------------
    # Forgetting
    # -------------------------------------------------------------------------

    def _forget_plot(self, plot_id: str) -> None:
        """
        Forget a plot - not deletion, but letting go.
        
        Philosophy: "Forgetting is not losing information, 
        it's choosing what to become."
        
        The plot's essence is preserved in:
        - The Story's centroid (aggregate meaning)
        - The relationship trajectory (if relational)
        - The identity dimensions it affected
        
        Args:
            plot_id: ID of the plot to forget
        """
        p = self.plots.get(plot_id)
        if p is None or p.story_id is None:
            return
        
        p.status = "absorbed"

        # Remove from vector index to reduce retrieval noise
        self.vindex.remove(plot_id)
        
        # If this plot had identity impact, the impact is preserved in the dimensions
        # (The _identity_dimensions dict retains the accumulated effect)
        
        # If this plot had relational context, the relationship trajectory preserves it
        # (The story's relationship_arc retains the pattern)
