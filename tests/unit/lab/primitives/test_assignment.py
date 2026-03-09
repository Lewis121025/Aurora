"""分配模型（CRP、StoryModel、ThemeModel）的测试。"""

import math
import numpy as np
import pytest

from aurora.lab.primitives.assignment import CRPAssigner, StoryModel, ThemeModel
from aurora.lab.primitives.metric import LowRankMetric
from aurora.lab.models.plot import Plot
from aurora.lab.models.story import StoryArc
from aurora.lab.models.theme import Theme
from aurora.utils.time_utils import now_ts


class TestCRPAssigner:
    """CRPAssigner 类的测试。"""

    def test_initial_state(self):
        """测试 CRP 分配器的初始状态。"""
        crp = CRPAssigner(alpha=1.5, seed=42)
        assert crp.alpha == 1.5
        assert crp._seed == 42

    def test_sample_empty_returns_new(self):
        """Test that sampling from empty set returns new cluster."""
        crp = CRPAssigner(alpha=1.0, seed=0)

        choice, post = crp.sample({})

        # Should always choose new cluster when no existing options
        assert choice is None
        assert "__new__" in post
        assert np.isclose(post["__new__"], 1.0)  # Only option (with tolerance)

    def test_sample_includes_new_option(self):
        """Test that new cluster option is always included."""
        crp = CRPAssigner(alpha=2.0, seed=0)

        logps = {"cluster1": math.log(5.0), "cluster2": math.log(3.0)}
        choice, post = crp.sample(logps)

        # Posterior should include all options + new
        assert "cluster1" in post
        assert "cluster2" in post
        assert "__new__" in post

    def test_alpha_affects_new_cluster_probability(self):
        """Test that higher alpha increases new cluster probability."""
        crp_low = CRPAssigner(alpha=0.1, seed=42)
        crp_high = CRPAssigner(alpha=10.0, seed=42)

        logps = {"existing": math.log(5.0)}

        # Run many samples
        new_count_low = 0
        new_count_high = 0

        for _ in range(1000):
            choice, _ = crp_low.sample(logps)
            if choice is None:
                new_count_low += 1

            choice, _ = crp_high.sample(logps)
            if choice is None:
                new_count_high += 1

        # Higher alpha should result in more new clusters
        assert new_count_high > new_count_low

    def test_log_probabilities_affect_choice(self):
        """Test that higher log probabilities are chosen more often."""
        crp = CRPAssigner(alpha=0.1, seed=42)

        # Make cluster1 much more likely
        logps = {"cluster1": math.log(100.0), "cluster2": math.log(1.0)}

        counts = {"cluster1": 0, "cluster2": 0, "new": 0}
        for _ in range(1000):
            choice, _ = crp.sample(logps)
            if choice is None:
                counts["new"] += 1
            else:
                counts[choice] += 1

        # cluster1 should be chosen much more often
        assert counts["cluster1"] > counts["cluster2"]

    def test_serialization(self):
        """Test state dict serialization and deserialization."""
        crp1 = CRPAssigner(alpha=2.5, seed=123)

        state = crp1.to_state_dict()
        crp2 = CRPAssigner.from_state_dict(state)

        assert crp2.alpha == crp1.alpha
        assert crp2._seed == crp1._seed


class TestStoryModel:
    """Tests for StoryModel class."""

    def test_loglik_no_centroid(self):
        """Test log likelihood when story has no centroid."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        model = StoryModel(metric)

        plot = Plot(
            id="plot1",
            ts=now_ts(),
            text="test",
            actors=("user",),
            embedding=np.random.randn(32).astype(np.float32),
        )

        story = StoryArc(id="story1", created_ts=now_ts(), updated_ts=now_ts())
        # Story has no centroid

        ll = model.loglik(plot, story)
        # Should return approximately 0 for semantic component when no centroid
        # (may have small floating point errors from temporal/actor components)
        assert np.isclose(ll, 0.0, atol=1e-10)

    def test_loglik_with_centroid(self):
        """Test log likelihood when story has centroid."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        model = StoryModel(metric)

        embedding = np.random.randn(32).astype(np.float32)
        plot = Plot(
            id="plot1",
            ts=now_ts(),
            text="test",
            actors=("user",),
            embedding=embedding,
        )

        story = StoryArc(id="story1", created_ts=now_ts(), updated_ts=now_ts())
        story.plot_ids = ["prev1"]
        story.centroid = embedding + 0.1 * np.random.randn(32).astype(np.float32)
        story._n_dist = 10
        story._mean_dist = 0.5
        story._m2_dist = 1.0

        ll = model.loglik(plot, story)
        # Should be a finite number
        assert np.isfinite(ll)

    def test_loglik_closer_plots_higher(self):
        """Test that closer plots have higher log likelihood."""
        metric = LowRankMetric(dim=32, rank=8, seed=42)
        model = StoryModel(metric)

        story = StoryArc(id="story1", created_ts=now_ts(), updated_ts=now_ts())
        story.centroid = np.random.randn(32).astype(np.float32)
        story.plot_ids = ["p1", "p2", "p3"]
        story._n_dist = 3
        story._mean_dist = 0.5
        story._m2_dist = 0.25

        # Close plot
        close_emb = story.centroid + 0.01 * np.random.randn(32).astype(np.float32)
        close_plot = Plot(
            id="close", ts=now_ts(), text="close",
            actors=("user",), embedding=close_emb
        )

        # Far plot
        far_emb = story.centroid + 10 * np.random.randn(32).astype(np.float32)
        far_plot = Plot(
            id="far", ts=now_ts(), text="far",
            actors=("user",), embedding=far_emb
        )

        ll_close = model.loglik(close_plot, story)
        ll_far = model.loglik(far_plot, story)

        assert ll_close > ll_far


class TestThemeModel:
    """Tests for ThemeModel class."""

    def test_loglik_no_prototype(self):
        """Test log likelihood when theme has no prototype."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        model = ThemeModel(metric)

        story = StoryArc(id="story1", created_ts=now_ts(), updated_ts=now_ts())
        story.centroid = np.random.randn(32).astype(np.float32)

        theme = Theme(id="theme1", created_ts=now_ts(), updated_ts=now_ts())
        # Theme has no prototype

        ll = model.loglik(story, theme)
        assert ll == 0.0

    def test_loglik_no_centroid(self):
        """Test log likelihood when story has no centroid."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        model = ThemeModel(metric)

        story = StoryArc(id="story1", created_ts=now_ts(), updated_ts=now_ts())
        # Story has no centroid

        theme = Theme(id="theme1", created_ts=now_ts(), updated_ts=now_ts())
        theme.prototype = np.random.randn(32).astype(np.float32)

        ll = model.loglik(story, theme)
        assert ll == 0.0

    def test_loglik_closer_stories_higher(self):
        """Test that closer stories have higher log likelihood."""
        metric = LowRankMetric(dim=32, rank=8, seed=42)
        model = ThemeModel(metric)

        theme = Theme(id="theme1", created_ts=now_ts(), updated_ts=now_ts())
        theme.prototype = np.random.randn(32).astype(np.float32)

        # Close story
        close_story = StoryArc(id="close", created_ts=now_ts(), updated_ts=now_ts())
        close_story.centroid = theme.prototype + 0.01 * np.random.randn(32).astype(np.float32)

        # Far story
        far_story = StoryArc(id="far", created_ts=now_ts(), updated_ts=now_ts())
        far_story.centroid = theme.prototype + 10 * np.random.randn(32).astype(np.float32)

        ll_close = model.loglik(close_story, theme)
        ll_far = model.loglik(far_story, theme)

        assert ll_close > ll_far

    def test_loglik_returns_negative(self):
        """Test that log likelihood is negative (as distance increases it)."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        model = ThemeModel(metric)

        story = StoryArc(id="story1", created_ts=now_ts(), updated_ts=now_ts())
        story.centroid = np.random.randn(32).astype(np.float32)

        theme = Theme(id="theme1", created_ts=now_ts(), updated_ts=now_ts())
        theme.prototype = np.random.randn(32).astype(np.float32)

        ll = model.loglik(story, theme)
        # Based on Gaussian likelihood, should be negative
        assert ll <= 0
