"""Thompson Bernoulli Gate bandit 的测试。"""

import numpy as np
import pytest

from aurora.algorithms.components.bandit import ThompsonBernoulliGate


class TestThompsonBernoulliGate:
    """ThompsonBernoulliGate 类的测试。"""

    def test_initial_state(self):
        """测试门的初始状态。"""
        gate = ThompsonBernoulliGate(feature_dim=6, seed=42, forgetting_factor=0.99)
        assert gate.d == 6
        assert gate._seed == 42
        assert gate.lambda_ == 0.99
        assert gate.t == 0
        assert gate._encode_count == 0
        assert gate._skip_count == 0
        assert gate.w_mean.shape == (6,)
        assert gate.prec.shape == (6,)
        assert gate.grad2.shape == (6,)

    def test_prob_range(self):
        """测试概率在 [0, 1] 范围内。"""
        gate = ThompsonBernoulliGate(feature_dim=6, seed=0)
        x = np.random.randn(6).astype(np.float32)

        prob = gate.prob(x)
        assert 0 <= prob <= 1

    def test_decide_returns_bool(self):
        """测试 decide 返回布尔值。"""
        gate = ThompsonBernoulliGate(feature_dim=6, seed=0)
        x = np.random.randn(6).astype(np.float32)

        decision = gate.decide(x)
        assert isinstance(decision, bool)

    def test_decide_updates_counts(self):
        """测试 decide 更新编码/跳过计数。"""
        gate = ThompsonBernoulliGate(feature_dim=6, seed=42)

        # Run many decisions
        for _ in range(100):
            x = np.random.randn(6).astype(np.float32)
            gate.decide(x)

        # Total should equal 100
        assert gate._encode_count + gate._skip_count == 100

    def test_pass_rate(self):
        """测试通过率计算。"""
        gate = ThompsonBernoulliGate(feature_dim=6, seed=0)

        # No decisions yet
        assert gate.pass_rate() == 0.5  # Default

        # Force some decisions
        gate._encode_count = 30
        gate._skip_count = 70
        assert gate.pass_rate() == 0.3

    def test_update_increments_t(self):
        """测试 update 增加时间计数器。"""
        gate = ThompsonBernoulliGate(feature_dim=6, seed=0)
        x = np.random.randn(6).astype(np.float32)

        gate.update(x, reward=1.0)
        assert gate.t == 1

        gate.update(x, reward=-1.0)
        assert gate.t == 2

    def test_update_changes_weights(self):
        """测试 update 改变权重均值。"""
        gate = ThompsonBernoulliGate(feature_dim=6, seed=0)
        x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        initial_w = gate.w_mean.copy()

        # Positive reward
        gate.update(x, reward=1.0)

        # Weights should change (at least slightly)
        assert not np.allclose(gate.w_mean, initial_w)

    def test_learning_from_rewards(self):
        """测试一致的奖励影响概率。"""
        gate = ThompsonBernoulliGate(feature_dim=4, seed=42)

        # Create a simple pattern: high feature 0 -> should encode
        x_positive = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Train with positive rewards
        for _ in range(50):
            gate.update(x_positive, reward=1.0)

        # After training, probability should be higher for this pattern
        # (Thompson sampling makes this stochastic, but trend should be there)
        probs = [gate.prob(x_positive) for _ in range(10)]
        avg_prob = np.mean(probs)
        assert avg_prob > 0.5  # Should lean toward encoding

    def test_forgetting_factor_maintains_plasticity(self):
        """测试遗忘因子防止精度爆炸。"""
        gate = ThompsonBernoulliGate(feature_dim=6, seed=0, forgetting_factor=0.99)
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # Many updates
        for _ in range(1000):
            gate.update(x, reward=1.0)

        # Precision is bounded due to forgetting factor
        # With lambda=0.99, precision stabilizes around a finite value
        # (it doesn't explode to infinity)
        final_prec = gate.prec.copy()

        # Precision should be bounded (not explode to very large values)
        assert np.all(final_prec < 1e6)
        # Precision should be positive
        assert np.all(final_prec > 0)

    def test_serialization(self):
        """测试状态字典的序列化和反序列化。"""
        gate1 = ThompsonBernoulliGate(feature_dim=6, seed=123, forgetting_factor=0.95)

        # Do some updates
        for i in range(50):
            x = np.random.randn(6).astype(np.float32)
            gate1.decide(x)
            gate1.update(x, reward=1.0 if i % 2 == 0 else -1.0)

        # Serialize
        state = gate1.to_state_dict()

        # Deserialize
        gate2 = ThompsonBernoulliGate.from_state_dict(state)

        # Check attributes
        assert gate2.d == gate1.d
        assert gate2._seed == gate1._seed
        assert gate2.lambda_ == gate1.lambda_
        assert gate2.t == gate1.t
        assert gate2._encode_count == gate1._encode_count
        assert gate2._skip_count == gate1._skip_count

        # Check arrays
        np.testing.assert_array_almost_equal(gate2.w_mean, gate1.w_mean)
        np.testing.assert_array_almost_equal(gate2.prec, gate1.prec)
        np.testing.assert_array_almost_equal(gate2.grad2, gate1.grad2)

    def test_sample_w_uses_precision(self):
        """测试采样权重使用精度计算方差。"""
        gate = ThompsonBernoulliGate(feature_dim=4, seed=42)

        # With weak initial precision, samples should vary more
        samples_weak = [gate._sample_w() for _ in range(100)]
        var_weak = np.var(samples_weak, axis=0)

        # Increase precision
        gate.prec = np.ones(4, dtype=np.float32) * 100

        samples_strong = [gate._sample_w() for _ in range(100)]
        var_strong = np.var(samples_strong, axis=0)

        # Higher precision = lower variance
        assert np.all(var_strong < var_weak)
