"""
AURORA Assignment Models
=========================

Nonparametric hierarchical assignment using Chinese Restaurant Process (CRP)
and probabilistic generative models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from aurora.utils.math_utils import softmax
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.components.metric import LowRankMetric


class CRPAssigner:
    """Generic CRP-like assigner for (item -> cluster) with probabilistic sampling.

    Implements the Chinese Restaurant Process for nonparametric clustering,
    where the number of clusters grows with the data but at a rate controlled
    by the concentration parameter alpha.

    Attributes:
        alpha: CRP concentration parameter (higher = more new clusters)
    """

    def __init__(self, alpha: float = 1.0, seed: int = 0):
        """Initialize the CRP assigner.

        Args:
            alpha: Concentration parameter
            seed: Random seed
        """
        self.alpha = alpha
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, logps: Dict[str, float]) -> Tuple[Optional[str], Dict[str, float]]:
        """Sample a cluster assignment using CRP.

        Args:
            logps: Log probabilities for existing clusters

        Returns:
            Tuple of (chosen cluster ID or None for new cluster, posterior probabilities)
        """
        # Add new cluster option
        logs = dict(logps)
        logs["__new__"] = math.log(self.alpha)
        keys = list(logs.keys())
        probs = softmax([logs[k] for k in keys])
        choice = self.rng.choice(keys, p=np.array(probs, dtype=np.float64))
        post = {k: p for k, p in zip(keys, probs)}
        if choice == "__new__":
            return None, post
        return choice, post

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "alpha": self.alpha,
            "seed": self._seed,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "CRPAssigner":
        """Reconstruct from state dict."""
        return cls(alpha=d["alpha"], seed=d.get("seed", 0))


class StoryModel:
    """Likelihood model p(plot | story) with interpretable factors.

    No fixed weights: all components are generative log-likelihood terms.
    The model computes the probability of observing a plot given a story
    based on semantic, temporal, and actor features.

    Components:
        - Semantic likelihood: Gaussian in metric space using story's dispersion
        - Temporal likelihood: Exponential around learned typical gap
        - Actor likelihood: Dirichlet-multinomial predictive

    Attributes:
        metric: The learned metric for semantic similarity
    """

    def __init__(self, metric: LowRankMetric):
        """Initialize the story model.

        Args:
            metric: Learned low-rank metric for semantic similarity
        """
        self.metric = metric

    def loglik(self, plot: Plot, story: StoryArc) -> float:
        """Compute log likelihood of plot given story.

        Args:
            plot: The plot to evaluate
            story: The story to condition on

        Returns:
            Log likelihood value
        """
        # Semantic likelihood: Gaussian in metric space using story's dispersion
        ll_sem = 0.0
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            var = max(story.dist_var(), 1e-3)
            ll_sem = -0.5 * d2 / var

        # Temporal likelihood: exponential around learned typical gap
        ll_time = 0.0
        if story.plot_ids:
            gap = max(0.0, plot.ts - story.updated_ts)
            tau = story.gap_mean_safe()
            lam = 1.0 / max(tau, 1e-6)
            ll_time = math.log(lam + 1e-12) - lam * gap

        # Actor likelihood: Dirichlet-multinomial predictive
        ll_actor = 0.0
        beta = 1.0
        total = sum(story.actor_counts.values())
        denom = total + beta * max(len(story.actor_counts), 1)
        for a in plot.actors:
            ll_actor += math.log(story.actor_counts.get(a, 0) + beta) - math.log(denom + 1e-12)

        return ll_sem + ll_time + ll_actor


class ThemeModel:
    """Likelihood model p(story | theme).

    Computes the probability of a story belonging to a theme based on
    semantic similarity in the learned metric space.

    Attributes:
        metric: The learned metric for semantic similarity
    """

    def __init__(self, metric: LowRankMetric):
        """Initialize the theme model.

        Args:
            metric: Learned low-rank metric for semantic similarity
        """
        self.metric = metric

    def loglik(self, story: StoryArc, theme: Theme) -> float:
        """Compute log likelihood of story given theme.

        Args:
            story: The story to evaluate
            theme: The theme to condition on

        Returns:
            Log likelihood value
        """
        if theme.prototype is None or story.centroid is None:
            return 0.0
        d2 = self.metric.d2(story.centroid, theme.prototype)
        # Theme dispersion is not stored; we use a robust default scale = 1
        return -0.5 * d2
