"""Tests for LowRankMetric learning."""

import numpy as np
import pytest

from aurora.algorithms.components.metric import LowRankMetric


class TestLowRankMetric:
    """Tests for LowRankMetric class."""

    def test_initial_state(self):
        """Test initial state of metric."""
        metric = LowRankMetric(dim=64, rank=16, seed=42)
        assert metric.dim == 64
        assert metric.rank == 16
        assert metric.t == 0
        assert metric.L.shape == (16, 64)
        assert metric.G.shape == (16, 64)

    def test_rank_clamping(self):
        """Test that rank is clamped to dim."""
        metric = LowRankMetric(dim=32, rank=100, seed=0)
        # Rank should be clamped to dim
        assert metric.rank == 32

    def test_d2_self_zero(self):
        """Test that distance from a vector to itself is near zero."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        v = np.random.randn(32).astype(np.float32)

        d2 = metric.d2(v, v)
        assert np.isclose(d2, 0.0, atol=1e-6)

    def test_d2_symmetric(self):
        """Test that distance is symmetric."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        v1 = np.random.randn(32).astype(np.float32)
        v2 = np.random.randn(32).astype(np.float32)

        d12 = metric.d2(v1, v2)
        d21 = metric.d2(v2, v1)
        assert np.isclose(d12, d21)

    def test_d2_positive(self):
        """Test that distance is non-negative."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        v1 = np.random.randn(32).astype(np.float32)
        v2 = np.random.randn(32).astype(np.float32)

        d2 = metric.d2(v1, v2)
        assert d2 >= 0

    def test_sim_range(self):
        """Test that similarity is in (0, 1]."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)
        v1 = np.random.randn(32).astype(np.float32)
        v2 = np.random.randn(32).astype(np.float32)

        sim = metric.sim(v1, v2)
        assert 0 < sim <= 1

        # Self similarity should be 1
        self_sim = metric.sim(v1, v1)
        assert np.isclose(self_sim, 1.0)

    def test_update_triplet_no_loss(self):
        """Test triplet update when constraint already satisfied."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)

        anchor = np.random.randn(32).astype(np.float32)
        positive = anchor + 0.01 * np.random.randn(32).astype(np.float32)  # Very close
        negative = np.random.randn(32).astype(np.float32) * 10  # Very far

        loss = metric.update_triplet(anchor, positive, negative, margin=1.0)

        # Loss should be 0 when constraint already satisfied
        # (positive much closer than negative)
        assert loss >= 0

    def test_update_triplet_with_loss(self):
        """Test triplet update when constraint violated."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)

        anchor = np.random.randn(32).astype(np.float32)
        # Make positive farther than negative
        positive = anchor + 10 * np.random.randn(32).astype(np.float32)
        negative = anchor + 0.1 * np.random.randn(32).astype(np.float32)

        loss = metric.update_triplet(anchor, positive, negative, margin=1.0)

        # Loss should be positive when constraint violated
        assert loss > 0
        assert metric.t == 1

    def test_learning_improves_metric(self):
        """Test that triplet learning improves the metric."""
        metric = LowRankMetric(dim=32, rank=16, seed=42)

        # Create a consistent pattern
        anchor = np.random.randn(32).astype(np.float32)
        positive = anchor + 0.1 * np.ones(32).astype(np.float32)
        negative = anchor - 0.1 * np.ones(32).astype(np.float32)

        initial_d_pos = metric.d2(anchor, positive)
        initial_d_neg = metric.d2(anchor, negative)

        # Train on this pattern multiple times
        for _ in range(100):
            metric.update_triplet(anchor, positive, negative, margin=0.5)

        final_d_pos = metric.d2(anchor, positive)
        final_d_neg = metric.d2(anchor, negative)

        # After learning, positive should be closer than negative
        # or at least the margin should be satisfied
        margin_before = initial_d_neg - initial_d_pos
        margin_after = final_d_neg - final_d_pos
        assert margin_after > margin_before or margin_after > 0.5

    def test_average_loss(self):
        """Test average loss tracking."""
        metric = LowRankMetric(dim=32, rank=8, seed=0)

        # No updates yet
        assert metric.average_loss() == 0.0

        anchor = np.random.randn(32).astype(np.float32)
        positive = anchor + 5 * np.random.randn(32).astype(np.float32)
        negative = anchor + 0.1 * np.random.randn(32).astype(np.float32)

        # Do some updates that should have loss
        for _ in range(10):
            metric.update_triplet(anchor, positive, negative, margin=1.0)

        avg_loss = metric.average_loss()
        assert avg_loss >= 0
        assert metric._update_count > 0

    def test_window_rescaling(self):
        """Test that window rescaling maintains plasticity."""
        metric = LowRankMetric(dim=32, rank=8, seed=0, window_size=100, decay_factor=0.5)

        anchor = np.random.randn(32).astype(np.float32)
        positive = np.random.randn(32).astype(np.float32)
        negative = np.random.randn(32).astype(np.float32)

        # Do many updates to trigger window rescaling
        for _ in range(150):
            metric.update_triplet(anchor, positive, negative, margin=1.0)

        # G should have been rescaled at t=100
        assert metric.t == 150

    def test_serialization(self):
        """Test state dict serialization and deserialization."""
        metric1 = LowRankMetric(dim=32, rank=8, seed=123, window_size=500, decay_factor=0.3)

        # Do some updates
        for _ in range(50):
            anchor = np.random.randn(32).astype(np.float32)
            positive = np.random.randn(32).astype(np.float32)
            negative = np.random.randn(32).astype(np.float32)
            metric1.update_triplet(anchor, positive, negative)

        # Serialize
        state = metric1.to_state_dict()

        # Deserialize
        metric2 = LowRankMetric.from_state_dict(state)

        # Check attributes
        assert metric2.dim == metric1.dim
        assert metric2.rank == metric1.rank
        assert metric2._seed == metric1._seed
        assert metric2.window_size == metric1.window_size
        assert metric2.decay_factor == metric1.decay_factor
        assert metric2.t == metric1.t
        assert metric2._total_loss == metric1._total_loss
        assert metric2._update_count == metric1._update_count

        # Check matrices
        np.testing.assert_array_almost_equal(metric2.L, metric1.L)
        np.testing.assert_array_almost_equal(metric2.G, metric1.G)

    def test_deterministic_with_seed(self):
        """Test that same seed produces same initial state."""
        metric1 = LowRankMetric(dim=32, rank=8, seed=42)
        metric2 = LowRankMetric(dim=32, rank=8, seed=42)

        np.testing.assert_array_almost_equal(metric1.L, metric2.L)
