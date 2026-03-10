"""OnlineKDE 密度估计的测试。"""

import numpy as np

from aurora.lab.primitives.density import OnlineKDE


class TestOnlineKDE:
    """OnlineKDE 类的测试。"""

    def test_initial_state(self):
        """测试 KDE 的初始状态。"""
        kde = OnlineKDE(dim=64, reservoir=100, k_sigma=10, seed=42)
        assert kde.dim == 64
        assert kde.reservoir == 100
        assert kde.k_sigma == 10
        assert len(kde._vecs) == 0

    def test_add_vectors(self):
        """测试向向量库添加向量。"""
        kde = OnlineKDE(dim=32, reservoir=10, seed=0)

        for i in range(5):
            v = np.random.randn(32).astype(np.float32)
            kde.add(v)

        assert len(kde._vecs) == 5

    def test_reservoir_sampling(self):
        """测试超过容量时的向量库采样。"""
        kde = OnlineKDE(dim=16, reservoir=10, seed=0)

        for i in range(100):
            v = np.random.randn(16).astype(np.float32)
            kde.add(v)

        # Should maintain reservoir size
        assert len(kde._vecs) == 10

    def test_log_density_empty(self):
        """测试空向量库的对数密度。"""
        kde = OnlineKDE(dim=32, seed=0)
        v = np.random.randn(32).astype(np.float32)

        # Should return weak prior
        log_d = kde.log_density(v)
        assert log_d == -10.0

    def test_log_density_with_vectors(self):
        """测试向量库中有向量时的对数密度。"""
        kde = OnlineKDE(dim=32, reservoir=100, seed=0)

        # Add some vectors
        for i in range(20):
            v = np.random.randn(32).astype(np.float32)
            kde.add(v)

        # Query with a vector
        q = np.random.randn(32).astype(np.float32)
        log_d = kde.log_density(q)

        # Should be a finite negative number
        assert np.isfinite(log_d)
        assert log_d < 0

    def test_surprise(self):
        """测试惊奇度计算。"""
        kde = OnlineKDE(dim=32, reservoir=100, seed=0)

        # Add vectors
        for i in range(20):
            v = np.random.randn(32).astype(np.float32)
            kde.add(v)

        q = np.random.randn(32).astype(np.float32)
        surprise = kde.surprise(q)
        log_d = kde.log_density(q)

        # Surprise should be negative of log density
        assert np.isclose(surprise, -log_d)

    def test_density_changes_with_data(self):
        """测试随着数据添加对数密度会改变。"""
        np.random.seed(42)
        kde = OnlineKDE(dim=8, reservoir=100, seed=42)

        # Query point
        query = np.zeros(8, dtype=np.float32)

        # Initial density (empty) should be weak prior
        assert kde.log_density(query) == -10.0

        # Add some vectors near the query
        for i in range(20):
            v = query + 0.1 * np.random.randn(8).astype(np.float32)
            kde.add(v)

        # Density should now be higher (less negative)
        log_d = kde.log_density(query)
        assert log_d > -10.0
        assert np.isfinite(log_d)

    def test_serialization(self):
        """测试状态字典的序列化和反序列化。"""
        kde1 = OnlineKDE(dim=16, reservoir=50, k_sigma=5, seed=123)

        # Add some vectors
        for i in range(30):
            v = np.random.randn(16).astype(np.float32)
            kde1.add(v)

        # Serialize
        state = kde1.to_state_dict()

        # Deserialize
        kde2 = OnlineKDE.from_state_dict(state)

        # Check attributes
        assert kde2.dim == kde1.dim
        assert kde2.reservoir == kde1.reservoir
        assert kde2.k_sigma == kde1.k_sigma
        assert kde2._seed == kde1._seed
        assert len(kde2._vecs) == len(kde1._vecs)

        # Check vectors are equal
        for v1, v2 in zip(kde1._vecs, kde2._vecs):
            np.testing.assert_array_almost_equal(v1, v2)

    def test_deterministic_with_seed(self):
        """测试相同的 seed 产生相同的结果。"""
        kde1 = OnlineKDE(dim=32, reservoir=10, seed=42)
        kde2 = OnlineKDE(dim=32, reservoir=10, seed=42)

        vectors = [np.random.randn(32).astype(np.float32) for _ in range(20)]

        for v in vectors:
            kde1.add(v.copy())
            kde2.add(v.copy())

        # After same sequence of adds, should have same state
        assert len(kde1._vecs) == len(kde2._vecs)
